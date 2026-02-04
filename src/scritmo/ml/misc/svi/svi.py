import torch
import pyro
from pyro.poutine import trace, replay
from ..marginalization import marginalize_theta_svi, log_like_loss


# This implementation si different from before in that it doesnt use the posterior
# but the prior for the elbo.
class SVI_model(torch.nn.Module):
    def __init__(self, model, guide):
        super().__init__()
        self.model = model
        self.guide = guide
        print("the multiplication by the prior of beta is still not there")

    def forward(self, y, mp, num_samples: int = 1):
        """
        Custom ELBO with the log likelihood
        Returns negative ELBO for minimization
        """
        elbo_sum = 0.0

        for _ in range(num_samples):
            guide_trace = trace(self.guide).get_trace(y, mp=mp)
            model_trace = (
                trace(replay(self.model, guide_trace)).get_trace(y, mp=mp).nodes
            )

            # Initialize ELBO for this sample
            elbo_sample = 0.0

            # KL term: -log(q(beta)), entropy of the guide
            for site in guide_trace.nodes.values():
                if site["type"] == "sample":
                    elbo_sample = elbo_sample - site["fn"].log_prob(site["value"]).sum()

            # log of the prior evaluated at the posterior
            for site in model_trace.values():
                if site["type"] == "sample" and not site["is_observed"]:
                    elbo_sample = elbo_sample + site["fn"].log_prob(site["value"]).sum()

            # Getting the scanned log likelihood
            ll_xcg = model_trace["obs"]["fn"].log_prob(y)

            # Expected marginalized log likelihood P(D | beta) NOT MULTIPLIED BY P(beta)
            l_xc, max_c = marginalize_theta_svi(ll_xcg, method="simpson")
            # log and add back the max
            elbo_sample = elbo_sample + log_like_loss(l_xc, max_c)

            elbo_sum += elbo_sample

        elbo = elbo_sum / num_samples

        # Return negative ELBO as the loss to minimize
        return -elbo


def get_svi_marginalized_posterior(model, guide, y, mp, num_samples=100):
    """
    To use after training svi.SVI_model !
    Directly sample from guide distributions rather than using Pyro's sampling mechanisms.
    It samples beta values from the posterior to give a marginalized posterior for the
    phases
    """
    with torch.no_grad():
        accumulated_l_xc = None

        # First, capture the guide trace to understand the structure
        guide_trace = pyro.poutine.trace(guide).get_trace(y, mp=mp)

        for i in range(num_samples):
            # Manually create samples from each distribution in the guide
            samples = {}

            # Extract the distributions from the guide trace
            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample" and not site["is_observed"]:
                    # Get the distribution object
                    dist_obj = site["fn"]
                    # Directly sample from it
                    samples[name] = dist_obj.sample()
                    # print(f"Manual sample {i}, {name}: {samples[name].sum().item()}")

            # Run the model with these manual samples
            conditioned_model = pyro.poutine.condition(model, data=samples)
            model_trace = (
                pyro.poutine.trace(conditioned_model).get_trace(y, mp=mp).nodes
            )

            ll_xcg = model_trace["obs"]["fn"].log_prob(y)
            ll_xc = ll_xcg.sum(dim=2)

            maxs, _ = torch.max(ll_xc, dim=0, keepdim=True)
            l_xc = torch.exp(ll_xc - maxs)
            l_xc = l_xc / l_xc.sum(dim=0, keepdim=True)

            if accumulated_l_xc is None:
                accumulated_l_xc = l_xc
            else:
                accumulated_l_xc += l_xc

        posterior_xc = accumulated_l_xc / num_samples
        return posterior_xc
