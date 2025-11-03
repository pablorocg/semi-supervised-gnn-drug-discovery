import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
)


class MultiTaskROCAUC(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds.append(preds.detach())
        self.target.append(target.detach())

    def compute(self):
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)

        rocauc_list = []
        for i in range(target.shape[1]):
            # Check if both classes present and filter NaN
            is_labeled = ~torch.isnan(target[:, i])
            y_true = target[is_labeled, i]
            y_pred = preds[is_labeled, i]

            if len(y_true) > 0 and y_true.sum() > 0 and (y_true == 0).sum() > 0:
                rocauc_list.append(binary_auroc(y_pred, y_true.long()))

        if len(rocauc_list) == 0:
            return torch.tensor(0.0)
        return torch.stack(rocauc_list).mean()


class MultiTaskAP(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds.append(preds.detach())
        self.target.append(target.detach())

    def compute(self):
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)

        ap_list = []
        for i in range(target.shape[1]):
            is_labeled = ~torch.isnan(target[:, i])
            y_true = target[is_labeled, i]
            y_pred = preds[is_labeled, i]

            if len(y_true) > 0 and y_true.sum() > 0 and (y_true == 0).sum() > 0:
                ap_list.append(binary_average_precision(y_pred, y_true.long()))

        if len(ap_list) == 0:
            return torch.tensor(0.0)
        return torch.stack(ap_list).mean()


class MultiTaskRMSE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds.append(preds.detach())
        self.target.append(target.detach())

    def compute(self):
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)

        rmse_list = []
        for i in range(target.shape[1]):
            is_labeled = ~torch.isnan(target[:, i])
            y_true = target[is_labeled, i]
            y_pred = preds[is_labeled, i]

            if len(y_true) > 0:
                rmse_list.append(torch.sqrt(((y_true - y_pred) ** 2).mean()))

        return torch.stack(rmse_list).mean() if rmse_list else torch.tensor(0.0)



if __name__ == "__main__":
    # Usage
    metrics = MetricCollection(
        {
            "rocauc": MultiTaskROCAUC(),
            "ap": MultiTaskAP(),
            "rmse": MultiTaskRMSE(),
        }
    )

    # # In training loop
    # metrics.update(preds, targets)
    # result = metrics.compute()  # {'rocauc': tensor(...), 'ap': tensor(...), 'rmse': tensor(...)}
    # metrics.reset()
