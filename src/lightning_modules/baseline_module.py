import torch
import torch.nn.functional as F
from src.lightning_modules.base_module import BaseModule


class BaselineGNNModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
    
    def forward(self, batch):
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        
        # Filtra NaN targets
        mask = ~torch.isnan(batch.y)
        if mask.sum() == 0:
            return None
        
        loss = self.loss_fn(logits[mask], batch.y[mask].float())
        
        # Chequea NaN
        if torch.isnan(loss):
            return None
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        preds = torch.sigmoid(logits)
        self.train_metrics.update(preds, batch.y)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        
        # Filtra NaN
        mask = ~torch.isnan(batch.y)
        if mask.sum() == 0:
            return None
        
        loss = self.loss_fn(logits[mask], batch.y[mask].float())
        
        if torch.isnan(loss):
            return None
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        preds = torch.sigmoid(logits)
        self.val_metrics.update(preds, batch.y)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        logits = self(batch)
        
        preds = torch.sigmoid(logits)
        self.test_metrics.update(preds, batch.y)
