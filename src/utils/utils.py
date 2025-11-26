import os
import pickle
import random
from itertools import chain
from time import time

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR



def seed_everything(seed, force_deterministic):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if force_deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# OBS (Mikkjo comment): This may actually be done in place on cfg file, just not entirely sure so I
# am returning cfg object just to be sure.
def set_debug_params(cfg):
    torch.autograd.set_detect_anomaly(True)
    # Limit number of cpu threads
    torch.set_num_threads(8)
    cfg.trainer.num_epochs = 1
    cfg.ensemble.num_models = 4
    cfg.trainer.validation_interval = 1
    cfg.logger.group_name = "debug"
    return cfg


def prepare_noise_hook(noise: float):
    def gradient_noise_hook(grad_input, noise):
        noise = torch.randn_like(grad_input) * noise
        return grad_input + noise


def prepare_optimizer(optimizer, models):
    # Assume optimizer is a hydra-configured object (already instantiated)
    # Set parameters from all models
    optimizer.param_groups = []
    params = list(chain(*[model.parameters() for model in models]))
    optimizer.add_param_group({'params': params})
    return optimizer


def prepare_scheduler(optimizer, step_size, gamma):
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


# (Mikkjo comment): This validate model should maybe be included in another object.
# either a trainer or a wrapper for the models. But again, this would require large refactoring.
def validate_models(models, val_loader, device, ensemble=False, classification=False):
    for model in models:
        model.eval()
    if classification:
        correct, total = 0, 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = torch.stack([model(images) for model in models])
            outputs = torch.mean(torch.softmax(outputs, dim=0), dim=0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return 100 * correct / total
    else: 
        # regression
        for molecules, targets in val_loader:
            molecules, targets = molecules.to(device), targets.to(device)
            outputs = torch.stack([model(molecules) for model in models])
            if ensemble:
                outputs = torch.mean(outputs, dim=0)
            else:
                outputs = outputs[0]

        mse = torch.mean((outputs - targets) ** 2)
        return mse.item()


def save_results(results, log_dir):
    # Save the accuracy list
    t = str(int(time()))
    r = torch.randint(low=0, high=2**31, size=(1,)).item()
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/results_{t}_{r}.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {log_dir}/results_{t}_{r}.pkl")

import torch

def calculate_pos_weights_from_tensor(labels: torch.Tensor, max_weight: float = 100.0):
    """
    Calculate per-task pos_weight for BCEWithLogitsLoss safely from a label tensor.

    Args:
        labels (torch.Tensor): Shape [num_samples, num_tasks], multi-task labels (0,1, or NaN)
        max_weight (float): Maximum weight cap to prevent extreme values

    Returns:
        torch.Tensor: Shape [num_tasks] with pos_weight for each task
    """
    num_tasks = labels.shape[1]
    pos_counts = torch.zeros(num_tasks, dtype=torch.float)
    neg_counts = torch.zeros(num_tasks, dtype=torch.float)

    for t in range(num_tasks):
        task_labels = labels[:, t]
        valid_mask = ~torch.isnan(task_labels)
        valid_labels = task_labels[valid_mask]

        if len(valid_labels) > 0:
            pos_counts[t] = (valid_labels == 1).sum()
            neg_counts[t] = (valid_labels == 0).sum()

    pos_weights = torch.ones(num_tasks)
    mask = pos_counts > 0
    pos_weights[mask] = neg_counts[mask] / pos_counts[mask]
    pos_weights = torch.clamp(pos_weights, min=1.0, max=max_weight)

    # Optional: print summary
    pos_ratio = pos_counts.sum() / (pos_counts.sum() + neg_counts.sum())
    print(f"\nClass Distribution Analysis:")
    print(f"Average positive ratio: {pos_ratio:.4f}")
    print(f"Pos weights - Min: {pos_weights.min():.2f}, Max: {pos_weights.max():.2f}, Mean: {pos_weights.mean():.2f}")
    print(f"Pos weights - Median: {pos_weights.median():.2f}, Std: {pos_weights.std():.2f}\n")

    return pos_weights
