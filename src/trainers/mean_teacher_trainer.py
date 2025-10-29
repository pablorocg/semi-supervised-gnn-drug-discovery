"""
Mean-Teacher training loop example with label masking.
Demonstrates how to use the LabeledUnlabeledDataset for SSL training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from typing import Tuple, Dict


class MeanTeacherTrainer:
    """
    Trainer for Mean-Teacher semi-supervised learning with PyTorch Geometric models.
    
    The Mean-Teacher framework uses:
    - Student model: trained with both supervised and consistency loss
    - Teacher model: exponential moving average (EMA) of student weights
    - Consistency loss: L2 distance between student and teacher predictions on unlabeled data
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        ema_decay: float = 0.999,
        lambda_consistency: float = 1.0,
        device: str = "cuda",
    ):
        """
        Args:
            model: Student model (PyTorch Geometric GNN)
            optimizer: Optimizer for student model
            ema_decay: EMA coefficient for teacher model updates
            lambda_consistency: Weight for consistency loss
            device: Device to train on
        """
        self.student = model.to(device)
        self.teacher = deepcopy(model).to(device)
        self.optimizer = optimizer
        self.ema_decay = ema_decay
        self.lambda_consistency = lambda_consistency
        self.device = device
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def update_teacher(self):
        """Update teacher model with exponential moving average."""
        with torch.no_grad():
            for student_param, teacher_param in zip(
                self.student.parameters(), self.teacher.parameters()
            ):
                teacher_param.data = (
                    self.ema_decay * teacher_param.data
                    + (1 - self.ema_decay) * student_param.data
                )
    
    def training_step(
        self, batch, criterion_sup: nn.Module
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step with labeled and unlabeled data.
        
        Args:
            batch: Batch from dataloader (contains is_labeled mask)
            criterion_sup: Supervised loss function (e.g., CrossEntropyLoss)
        
        Returns:
            loss: Total loss (supervised + consistency)
            metrics: Dictionary with loss breakdown
        """
        batch = batch.to(self.device)
        
        # Separate labeled and unlabeled samples
        labeled_mask = batch.is_labeled
        unlabeled_mask = ~batch.is_labeled
        
        self.student.train()
        self.teacher.eval()
        
        # Forward pass through student
        student_out = self.student(batch)
        
        # Supervised loss (only on labeled samples)
        loss_sup = torch.tensor(0.0, device=self.device)
        if labeled_mask.any():
            student_pred_labeled = student_out[labeled_mask]
            target_labeled = batch.y[labeled_mask]
            loss_sup = criterion_sup(student_pred_labeled, target_labeled)
        
        # Consistency loss (only on unlabeled samples)
        loss_consistency = torch.tensor(0.0, device=self.device)
        if unlabeled_mask.any():
            with torch.no_grad():
                teacher_out = self.teacher(batch)
                teacher_pred_unlabeled = teacher_out[unlabeled_mask]
            
            student_pred_unlabeled = student_out[unlabeled_mask]
            
            # Mean squared error between student and teacher logits
            loss_consistency = torch.nn.functional.mse_loss(
                student_pred_unlabeled, teacher_pred_unlabeled
            )
        
        # Total loss
        loss_total = loss_sup + self.lambda_consistency * loss_consistency
        
        # Backward pass
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()
        
        # Update teacher model
        self.update_teacher()
        
        metrics = {
            "loss_total": loss_total.item(),
            "loss_sup": loss_sup.item(),
            "loss_consistency": loss_consistency.item(),
            "n_labeled": labeled_mask.sum().item(),
            "n_unlabeled": unlabeled_mask.sum().item(),
        }
        
        return loss_total, metrics

