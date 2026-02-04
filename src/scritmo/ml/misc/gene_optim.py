import numpy as np
import torch
from torch import tensor as tt
import torch.jit
from tqdm import tqdm
from torch import nn
from sklearn.preprocessing import OneHotEncoder
from ..marginalization import (
    vectorized_simpson,
    marginalize_theta,
    log_like_loss,
    cell_prior,
)
import anndata
from ..context_model import ContextModel
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from ..utils import nmp
import pandas as pd
from collections import defaultdict


# Your subclass
class GeneWeightOptimizer(ContextModel):
    def __init__(self, weights_g=None, phase_estimator="mode", *args, **kwargs):
        """
        Initializes the GeneWeightOptimizer.

        Args:
            weights_g: A Tensor of weights for each gene.
            phase_estimator: The method used to estimate the phase (e.g., "mode" or "mean").
            regularization_strength: Your second new parameter (e.g., a float).
            *args, **kwargs: All arguments required by the parent ContextModel.
        """
        # First, call the parent class's __init__ with its arguments
        super().__init__(*args, **kwargs)

        # Now, handle the new parameters specific to this subclass
        print("GeneWeightOptimizer initialized")
        self.phase_estimator = phase_estimator

        if weights_g is None:
            self.weights_g = nn.Parameter(torch.ones(self.Ng))
        else:
            self.weights_g = nn.Parameter(weights_g)

        for param in self.parameters():
            param.requires_grad = False
        # only weights will be optimized
        self.weights_g.requires_grad = True

    def forward(self, y, indices=slice(None), y_e=None, **kwargs):
        """
        Forward pass with category-specific intercepts.
        the data y is already been batched. But the indices are still
        needed for the celltypes.
        """

        dist = self.nb_dist(indices=indices)
        ll_xcg = dist.log_prob(y)

        # max mode
        log_loss_xc = (ll_xcg * self.weights_g).sum(2)
        # Apply softmax to get probabilities
        phase_prob_xc = F.softmax(log_loss_xc, dim=0)
        phi_cos, phi_sin = self.X[:, indices, 0], self.X[:, indices, 1]

        avg_x = (phase_prob_xc * phi_cos).sum(dim=0)
        avg_y = (phase_prob_xc * phi_sin).sum(dim=0)

        # 3. Convert the average (x, y) vector back to an angle using atan2.
        #    atan2 is fully differentiable and correctly handles all quadrants.
        phi_c = torch.atan2(avg_y, avg_x)

        return phi_c

    def get_weighted_posterior(self, y, indices=slice(None), counts=None, n_theta=None):
        """
        Forward pass with category-specific intercepts.
        the data y is already been batched. But the indices are still
        needed for the celltypes.
        """

        dist = self.nb_dist_posterior(indices=indices, counts=counts, n_theta=n_theta)

        if n_theta is not None:
            y = y[0, :, :].unsqueeze(0).repeat(n_theta, 1, 1)

        ll_xcg = dist.log_prob(y)

        # max mode
        log_loss_xc = (ll_xcg * self.weights_g).sum(2)
        # Apply softmax to get probabilities
        phase_prob_xc = F.softmax(log_loss_xc, dim=0)
        post_xc = nmp(phase_prob_xc)

        return post_xc

    # def get_weighted_posterior(


def circular_phase_loss(
    phi_pred,
    phi_true,
    weights_for_reg=None,
    l1_lambda=0.0,
    l2_lambda=0.0,
    reduction="mean",
):
    """
    Calculates a circular loss with optional L1 (Lasso) and L2 (Ridge) regularization.

    The total loss is: Loss = AngularLoss + L1_Penalty + L2_Penalty.

    The angular component `1 - cos(phi_pred - phi_true)` ranges from 0 (perfect) to 2 (opposite).
    The regularization terms penalize large weight values to prevent overfitting.

    Args:
        phi_pred (torch.Tensor): The predicted angles from the model.
        phi_true (torch.Tensor): The ground truth angles.
        weights_for_reg (torch.Tensor, optional): The model weights to be regularized.
                                                  Required if l1_lambda or l2_lambda > 0.
        l1_lambda (float): Strength of the L1 (Lasso) penalty. Promotes sparsity.
        l2_lambda (float): Strength of the L2 (Ridge) penalty. Prevents large weights.
        reduction (str): Specifies the reduction for the angular loss component:
                         'none' | 'mean' | 'sum'. Default: 'mean'.

    Returns:
        torch.Tensor: The calculated total loss.
    """
    # Ensure phi_true is a tensor on the same device and dtype as the prediction
    phi_true_t = torch.as_tensor(phi_true, device=phi_pred.device, dtype=phi_pred.dtype)

    # 1. Calculate the core angular loss
    angular_loss_unreduced = 1.0 - torch.cos(phi_pred - phi_true_t)

    # Apply reduction to the angular loss component
    if reduction == "mean":
        angular_loss = angular_loss_unreduced.mean()
    elif reduction == "sum":
        angular_loss = angular_loss_unreduced.sum()
    else:  # 'none'
        angular_loss = angular_loss_unreduced

    # 2. Calculate L1 (Lasso) penalty
    l1_penalty = torch.tensor(0.0, device=phi_pred.device)
    if l1_lambda > 0:
        if weights_for_reg is None:
            raise ValueError("weights_for_reg must be provided for L1 regularization.")
        l1_penalty = l1_lambda * torch.abs(weights_for_reg).sum()

    # 3. Calculate L2 (Ridge) penalty
    l2_penalty = torch.tensor(0.0, device=phi_pred.device)
    if l2_lambda > 0:
        if weights_for_reg is None:
            raise ValueError("weights_for_reg must be provided for L2 regularization.")
        l2_penalty = l2_lambda * torch.square(weights_for_reg).sum()

    # 4. Combine all components for the final loss
    total_loss = angular_loss + l1_penalty + l2_penalty

    loss_components = {
        "angular": nmp(angular_loss),
        "l1": nmp(l1_penalty),
        "l2": nmp(l2_penalty),
        "total": nmp(total_loss),
    }

    return total_loss, loss_components


def train_gene_weights(
    model,
    data,
    true_ph,
    n_epochs=200,
    learning_rate=0.001,
    betas=(0.90, 0.999),
    show_progress=True,
    batch_size=None,
    l1_lambda=0.0,
    l2_lambda=0.0,
):
    """
    Train the model using the circular phase loss and return a history of loss components.
    """
    Nx, Nc, Ng = data.shape
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)

    if batch_size is None or batch_size >= Nc:
        batch_size = Nc

    data_permuted = data.permute(1, 0, 2)
    original_indices = torch.arange(Nc)
    dataset = TensorDataset(data_permuted, original_indices, torch.as_tensor(true_ph))
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Store the history for final output
    history = {"total_loss": [], "loss_components_per_epoch": []}

    epoch_iterator = (
        tqdm(range(n_epochs), desc="Training Progress")
        if show_progress
        else range(n_epochs)
    )

    for epoch in epoch_iterator:
        # Temporarily store losses for the current epoch
        batch_total_losses = []
        batch_loss_components_list = []

        for batch_data_permuted, indices, batch_true_ph in dataloader:
            batch_data = batch_data_permuted.permute(1, 0, 2)
            theta_hat = model(batch_data, indices=indices)

            loss, loss_components = circular_phase_loss(
                theta_hat,
                batch_true_ph,
                weights_for_reg=model.weights_g,
                reduction="sum",
                l1_lambda=l1_lambda,
                l2_lambda=l2_lambda,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record losses for this batch
            batch_total_losses.append(loss.item())
            batch_loss_components_list.append(loss_components)

        # === EPOCH-LEVEL AGGREGATION (THE CORRECTED PART) ===
        if batch_total_losses:
            # 1. Calculate average total loss for the epoch
            avg_epoch_loss = sum(batch_total_losses) / len(batch_total_losses)
            history["total_loss"].append(avg_epoch_loss)
            epoch_iterator.set_postfix(loss=avg_epoch_loss)

            # 2. Aggregate and average the loss components for the epoch
            # Use defaultdict to easily group component values from all batches
            aggregated_components = defaultdict(list)
            for components_dict in batch_loss_components_list:
                for key, value in components_dict.items():
                    aggregated_components[key].append(value)

            # Calculate the average for each component
            avg_components_for_epoch = {
                key: sum(values) / len(values)
                for key, values in aggregated_components.items()
            }
            history["loss_components_per_epoch"].append(avg_components_for_epoch)

    # === FINAL OUTPUT FORMATTING ===
    # Convert the list of dictionaries into a DataFrame
    loss_components_df = pd.DataFrame(history["loss_components_per_epoch"])

    # Set the index to be the epoch number
    if not loss_components_df.empty:
        loss_components_df.index.name = "epoch"

    return {
        "total_loss": history["total_loss"],
        "loss_components": loss_components_df,
    }


# def circular_phase_loss(
#     phi_pred,
#     phi_true,
#     weights_for_reg=None,
#     l1_lambda=0.0,
#     l2_lambda=0.0,
#     reduction="sum",
# ):
#     """
#     Calculates a circular loss with optional L1 (Lasso) and L2 (Ridge) regularization.

#     The total loss is: Loss = AngularLoss + L1_Penalty + L2_Penalty.

#     The angular component `1 - cos(phi_pred - phi_true)` ranges from 0 (perfect) to 2 (opposite).
#     The regularization terms penalize large weight values to prevent overfitting.

#     Args:
#         phi_pred (torch.Tensor): The predicted angles from the model.
#         phi_true (torch.Tensor): The ground truth angles.
#         weights_for_reg (torch.Tensor, optional): The model weights to be regularized.
#                                                   Required if l1_lambda or l2_lambda > 0.
#         l1_lambda (float): Strength of the L1 (Lasso) penalty. Promotes sparsity.
#         l2_lambda (float): Strength of the L2 (Ridge) penalty. Prevents large weights.
#         reduction (str): Specifies the reduction for the angular loss component:
#                          'none' | 'mean' | 'sum'. Default: 'mean'.

#     Returns:
#         torch.Tensor: The calculated total loss.
#     """
#     # Ensure phi_true is a tensor on the same device and dtype as the prediction
#     phi_true_t = torch.as_tensor(phi_true, device=phi_pred.device, dtype=phi_pred.dtype)

#     # 1. Calculate the core angular loss
#     angular_loss_unreduced = 1.0 - torch.cos(phi_pred - phi_true_t)

#     # Apply reduction to the angular loss component
#     if reduction == "mean":
#         angular_loss = angular_loss_unreduced.mean()
#     elif reduction == "sum":
#         angular_loss = angular_loss_unreduced.sum()
#     else:  # 'none'
#         angular_loss = angular_loss_unreduced

#     # 2. Calculate L1 (Lasso) penalty
#     l1_penalty = torch.tensor(0.0, device=phi_pred.device)
#     if l1_lambda > 0:
#         if weights_for_reg is None:
#             raise ValueError("weights_for_reg must be provided for L1 regularization.")
#         l1_penalty = l1_lambda * torch.abs(weights_for_reg).sum()

#     # 3. Calculate L2 (Ridge) penalty
#     l2_penalty = torch.tensor(0.0, device=phi_pred.device)
#     if l2_lambda > 0:
#         if weights_for_reg is None:
#             raise ValueError("weights_for_reg must be provided for L2 regularization.")
#         l2_penalty = l2_lambda * torch.square(weights_for_reg).sum()

#     # 4. Combine all components for the final loss
#     total_loss = angular_loss + l1_penalty + l2_penalty

#     return total_loss
