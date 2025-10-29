class NoisyCPSTrainer:
    """
    Trainer for Noisy Cross Pseudo Supervision (N-CPS) with label masking.
    
    N-CPS uses:
    - Confidence thresholding on unlabeled predictions
    - Pseudo-label generation from confident predictions
    - Mixing strategy between labeled and pseudo-labeled data
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        confidence_threshold: float = 0.95,
        lambda_pseudo: float = 1.0,
        device: str = "cuda",
    ):
        """
        Args:
            model: Student model (PyTorch Geometric GNN)
            optimizer: Optimizer for model
            confidence_threshold: Confidence threshold for pseudo-labeling
            lambda_pseudo: Weight for pseudo-labeled loss
            device: Device to train on
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.confidence_threshold = confidence_threshold
        self.lambda_pseudo = lambda_pseudo
        self.device = device
    
    def training_step(
        self, batch, criterion_sup: nn.Module
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step with confidence-based pseudo-labeling.
        
        Args:
            batch: Batch from dataloader (contains is_labeled mask)
            criterion_sup: Supervised loss function
        
        Returns:
            loss: Total loss (supervised + pseudo-labeled)
            metrics: Dictionary with loss breakdown
        """
        batch = batch.to(self.device)
        
        labeled_mask = batch.is_labeled
        unlabeled_mask = ~batch.is_labeled
        
        self.model.train()
        
        # Forward pass
        logits = self.model(batch)
        
        # Supervised loss on labeled samples
        loss_sup = torch.tensor(0.0, device=self.device)
        if labeled_mask.any():
            loss_sup = criterion_sup(logits[labeled_mask], batch.y[labeled_mask])
        
        # Pseudo-labeling on unlabeled samples
        loss_pseudo = torch.tensor(0.0, device=self.device)
        n_confident = 0
        
        if unlabeled_mask.any():
            with torch.no_grad():
                probs = torch.softmax(logits[unlabeled_mask], dim=1)
                confidence, pseudo_labels = torch.max(probs, dim=1)
                
                # Filter by confidence threshold
                confident_mask = confidence > self.confidence_threshold
                n_confident = confident_mask.sum().item()
            
            if n_confident > 0:
                confident_logits = logits[unlabeled_mask][confident_mask]
                confident_pseudo_labels = pseudo_labels[confident_mask]
                loss_pseudo = criterion_sup(
                    confident_logits, confident_pseudo_labels
                )
        
        # Total loss
        loss_total = loss_sup + self.lambda_pseudo * loss_pseudo
        
        # Backward pass
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()
        
        metrics = {
            "loss_total": loss_total.item(),
            "loss_sup": loss_sup.item(),
            "loss_pseudo": loss_pseudo.item(),
            "n_labeled": labeled_mask.sum().item(),
            "n_unlabeled": unlabeled_mask.sum().item(),
            "n_confident_pseudo": n_confident,
            "confidence_threshold": self.confidence_threshold,
        }
        
        return loss_total, metrics

# Example training loop integration
def train_epoch_ssl(
    trainer,  # MeanTeacherTrainer or NoisyCPSTrainer
    train_loader,
    criterion,
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch using SSL trainer.
    
    Args:
        trainer: SSL trainer instance
        train_loader: DataLoader with labeled/unlabeled batches
        criterion: Loss function
        epoch: Current epoch number
    
    Returns:
        avg_metrics: Averaged metrics over epoch
    """
    all_metrics = []
    
    for batch_idx, batch in enumerate(train_loader):
        loss, metrics = trainer.training_step(batch, criterion)
        all_metrics.append(metrics)
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: "
                  f"loss={loss.item():.4f}, "
                  f"n_labeled={metrics['n_labeled']}, "
                  f"n_unlabeled={metrics['n_unlabeled']}")
    
    # Average metrics over epoch
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if isinstance(all_metrics[0][key], (int, float)):
            if "n_" in key or "threshold" in key:
                # For counts and thresholds, take last value or sum
                avg_metrics[key] = all_metrics[-1][key]
            else:
                # For losses, take mean
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    return avg_metrics