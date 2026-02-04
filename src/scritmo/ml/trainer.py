from xml.parsers.expat import model
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import scritmo as sr
from scritmo import (
    compute_posterior_statistics,
    compute_posterior_mean,
    circular_deviation,
    rh,
    w,
    optimal_shift,
    compute_posterior_mode,
)

from .utils import nmp, assemble_mp
from functools import partial


def train_ritmo(
    model,
    data,
    data_u=None,
    n_epochs=200,
    learning_rate=0.001,
    betas=(0.90, 0.999),
    show_progress=True,
    batch_size=None,
    true_phase=None,
    train_par_only=None,
):
    """
    Train the deterministic Tempo model with an efficient DataLoader for batching over epochs.

    Args:
        model: DeterministicTempoModel instance
        data: Data tensor [Nx, Nc, Ng]
        mp: Model parameters dictionary (Note: 'counts' is no longer modified)
        n_epochs: Number of full passes over the dataset
        learning_rate: Learning rate for Adam optimizer
        betas: Beta parameters for Adam optimizer
        show_progress: Whether to show a progress bar
        batch_size: Number of cells to use in each batch. If None, use all cells.
        true_phase: Optional true phases for evaluation (not used in training)
        train_par_only: If provided, only parameters starting with this string will be trained.
            i.e. use "m" for the means

    Returns:
        losses: List of mean loss values for each epoch during training
    """
    Nx, Nc, Ng = data.shape

    # Create optimizer
    if train_par_only is not None:
        optimizer = torch.optim.Adam(
            (p for n, p in model.named_parameters() if n.startswith(train_par_only)),
            lr=1e-3,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)

    # If batch_size is not provided or is larger than Nc, use all data in one batch
    if batch_size is None or batch_size >= Nc:
        batch_size = Nc

    # Permute data from [Nx, Nc, Ng] -> [Nc, Nx, Ng]
    data_permuted = data.permute(1, 0, 2)

    # Create a tensor of original indices
    original_indices = torch.arange(Nc)

    tensors_to_load = [data_permuted, original_indices]

    # Handle unspliced data
    if model.unspliced_mode and data_u is not None:
        data_u_permuted = data_u.permute(1, 0, 2)
        # HOPEFULLY IT WORKS
        insert_idx = 1
        tensors_to_load.insert(insert_idx, data_u_permuted)

    # 2. Create TensorDataset and DataLoader
    dataset = TensorDataset(*tensors_to_load)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # This list will store the average loss for each epoch
    epoch_losses = []

    epoch_iterator = (
        tqdm(range(n_epochs), desc="Training Progress")
        if show_progress
        else range(n_epochs)
    )

    compute_posteriors = true_phase is not None and not model.fixed_cell_mode
    mad_epochs = []
    # Outer loop for epochs
    for epoch in epoch_iterator:

        # List to store losses for the current epoch's batches
        batch_losses = []

        # Inner loop for batches
        for batch in dataloader:
            # Reset batch-specific data
            batch_data_u = None

            # --- Unpack the batch based on model mode ---
            # This logic robustly unpacks based on what's in tensors_to_load

            # Start with core data
            batch_data_permuted = batch[0]
            current_idx = 1

            if model.unspliced_mode and data_u is not None:
                batch_data_u_permuted = batch[current_idx]
                batch_data_u = batch_data_u_permuted.permute(1, 0, 2)
                current_idx += 1

            indices = batch[current_idx]
            # --- End Unpacking ---

            # Permute main batch data
            batch_data = batch_data_permuted.permute(1, 0, 2)

            # Forward pass
            loss = model(
                batch_data, y_u=batch_data_u, indices=indices
            )

            ########## rest of code

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record loss for the current batch
            batch_losses.append(loss.item() / batch_size)

        # Calculate the average loss for the current epoch and store it
        if batch_losses:
            avg_epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(avg_epoch_loss)

        

        if compute_posteriors:
            with torch.no_grad():
                posterior_xc = model.get_phase_posteriors(
                    data,  y_u=data_u, method="sum"
                )
                post_mode_c = compute_posterior_mean(
                    posterior_xc,
                )
                _, aligned_mad = optimal_shift(post_mode_c, true_phase, verbose=False)
                mad_epochs.append(aligned_mad)

        # Update the postfix of the single, outer progress bar to show current loss
        if show_progress:
            epoch_iterator.set_postfix(
                epoch=f"{epoch + 1}/{n_epochs}", loss=f"{avg_epoch_loss:.4e}"
            )
    if compute_posteriors:
        return np.array(epoch_losses), np.array(mad_epochs) * rh
    else:
        return np.array(epoch_losses)
