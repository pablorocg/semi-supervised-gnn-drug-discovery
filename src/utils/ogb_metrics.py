import torch
import numpy as np
from torchmetrics import Metric, MetricCollection
from torchmetrics.utilities import dim_zero_cat
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

class MultitaskROC_AUC(Metric):
    def __init__(self, num_tasks, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = num_tasks
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self):
        # Added .detach() before .cpu().numpy()
        preds = dim_zero_cat(self.preds).detach().cpu().numpy()
        targets = dim_zero_cat(self.targets).detach().cpu().numpy()
        
        rocauc_list = []
        for i in range(self.num_tasks):
            # Check if task has both positive and negative labels
            if np.sum(targets[:, i] == 1) > 0 and np.sum(targets[:, i] == 0) > 0:
                # Filter NaNs
                is_labeled = ~np.isnan(targets[:, i])
                rocauc_list.append(roc_auc_score(targets[is_labeled, i], preds[is_labeled, i]))

        if len(rocauc_list) == 0:
            # Return 0.0 or raise error depending on preference, consistent with torchmetrics behavior
            return torch.tensor(0.0)
            
        return torch.tensor(sum(rocauc_list) / len(rocauc_list))

class MultitaskAveragePrecision(Metric):
    def __init__(self, num_tasks, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = num_tasks
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self):
        # Added .detach() before .cpu().numpy()
        preds = dim_zero_cat(self.preds).detach().cpu().numpy()
        targets = dim_zero_cat(self.targets).detach().cpu().numpy()
        
        ap_list = []
        for i in range(self.num_tasks):
            if np.sum(targets[:, i] == 1) > 0 and np.sum(targets[:, i] == 0) > 0:
                is_labeled = ~np.isnan(targets[:, i])
                ap_list.append(average_precision_score(targets[is_labeled, i], preds[is_labeled, i]))

        if len(ap_list) == 0:
            return torch.tensor(0.0)

        return torch.tensor(sum(ap_list) / len(ap_list))

class MultitaskPR_AUC(Metric):
    """
    Computes Area Under the Precision-Recall Curve averaged across tasks.
    """
    def __init__(self, num_tasks, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = num_tasks
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self):
        # Added .detach() before .cpu().numpy()
        preds = dim_zero_cat(self.preds).detach().cpu().numpy()
        targets = dim_zero_cat(self.targets).detach().cpu().numpy()
        
        prauc_list = []
        for i in range(self.num_tasks):
            if np.sum(targets[:, i] == 1) > 0 and np.sum(targets[:, i] == 0) > 0:
                is_labeled = ~np.isnan(targets[:, i])
                # Compute Precision-Recall Curve and then AUC
                precision, recall, _ = precision_recall_curve(targets[is_labeled, i], preds[is_labeled, i])
                prauc_list.append(auc(recall, precision))

        if len(prauc_list) == 0:
            return torch.tensor(0.0)

        return torch.tensor(sum(prauc_list) / len(prauc_list))

class MultitaskAccuracy(Metric):
    def __init__(self, num_tasks, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = num_tasks
        self.threshold = threshold
        # Store correct counts and total counts per task (vectorized)
        self.add_state("correct_counts", default=torch.zeros(num_tasks), dist_reduce_fx="sum")
        self.add_state("total_counts", default=torch.zeros(num_tasks), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # Create mask for non-NaN targets
        mask = ~torch.isnan(target)
        
        # Threshold predictions
        preds_binary = (preds >= self.threshold).float()
        
        # Calculate correct predictions where mask is True
        correct = (preds_binary == target) & mask
        
        # Update states summing over batch dimension (dim 0)
        self.correct_counts += correct.sum(dim=0)
        self.total_counts += mask.sum(dim=0)

    def compute(self):
        # Avoid division by zero for tasks with no data
        valid_tasks_mask = self.total_counts > 0
        
        if valid_tasks_mask.sum() == 0:
            return torch.tensor(0.0)
            
        acc_per_task = self.correct_counts[valid_tasks_mask] / self.total_counts[valid_tasks_mask]
        return acc_per_task.mean()

class MultitaskF1(Metric):
    def __init__(self, num_tasks, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = num_tasks
        self.threshold = threshold
        # Accumulate TP, FP, FN per task
        self.add_state("tp", default=torch.zeros(num_tasks), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_tasks), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_tasks), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        mask = ~torch.isnan(target)
        preds_binary = (preds >= self.threshold).float()
        
        # Filter inputs with mask
        # Note: We multiply by mask to zero out ignored values so they don't affect sum
        target_masked = torch.nan_to_num(target) * mask.float()
        preds_masked = preds_binary * mask.float()

        # True Positives: Pred=1, Real=1, Mask=True
        self.tp += ((preds_masked == 1) & (target_masked == 1) & mask).sum(dim=0)
        
        # False Positives: Pred=1, Real=0, Mask=True
        self.fp += ((preds_masked == 1) & (target_masked == 0) & mask).sum(dim=0)
        
        # False Negatives: Pred=0, Real=1, Mask=True
        self.fn += ((preds_masked == 0) & (target_masked == 1) & mask).sum(dim=0)

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Only average over tasks that actually had samples (TP+FP+FN > 0)
        # logic: if TP+FP+FN is 0, there was no data for that task
        valid_tasks = (self.tp + self.fp + self.fn) > 0
        
        if valid_tasks.sum() == 0:
            return torch.tensor(0.0)
            
        return f1_scores[valid_tasks].mean()

# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------
if __name__ == "__main__":
    N_TASKS = 3
    
    # Create the collection
    metrics = MetricCollection({
        'roc_auc': MultitaskROC_AUC(num_tasks=N_TASKS),
        'ap': MultitaskAveragePrecision(num_tasks=N_TASKS),
        'pr_auc': MultitaskPR_AUC(num_tasks=N_TASKS),
        'acc': MultitaskAccuracy(num_tasks=N_TASKS),
        'f1': MultitaskF1(num_tasks=N_TASKS)
    })

    # Mock Data
    # Batch 1
    preds = torch.tensor([
        [0.1, 0.8, 0.3],
        [0.9, 0.2, 0.4],
        [0.4, 0.4, 0.9]
    ], requires_grad=True) # Added requires_grad to test the fix
    target = torch.tensor([
        [0.0, 1.0, float('nan')],
        [1.0, 0.0, 0.0],
        [0.0, float('nan'), 1.0]
    ])

    # Update
    metrics.update(preds, target)
    
    # Compute
    results = metrics.compute()
    print(results)