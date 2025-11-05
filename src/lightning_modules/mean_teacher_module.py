import copy

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.optim import SGD


class MeanTeacher(pl.LightningModule):
    def __init__(
        self,
        model_arch: nn.Module,
        alpha: float = 0.999,
        lambda_w: float = 100.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        nesterov: bool = True,

    ):
        """
        Initializes the Mean Teacher module.

        Args:
            model_arch: A torch.nn.Module instance (e.g., a ResNet).
                        This will be deep-copied for the teacher.
            alpha: The EMA decay rate for the teacher model.
            lambda_w: The weight for the consistency loss.
            lr: Learning rate for the student model optimizer.
        """
        super().__init__()
        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters(ignore=["model_arch"])

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        # 1. Create Student and Teacher models
        self.student = model_arch
        # Create a deep copy for the teacher
        self.teacher = copy.deepcopy(model_arch)

        # 2. Freeze Teacher Model
        # The teacher model is updated via EMA, not backpropagation.
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 3. Define Loss Functions
        self.sup_loss_fn = nn.CrossEntropyLoss()
        self.con_loss_fn = nn.MSELoss()  # Common choice for consistency

        # 4. Define Metrics
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=10
        )  # Adjust num_classes
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=10
        )  # Adjust num_classes

    def forward(self, x):
        """
        For inference, we use the stable teacher model.
        """
        return self.teacher(x)

    def training_step(self, batch, batch_idx):
        # --- 0. Data Preparation ---
        # 'batch' is a dict: {'labeled': <PyG_Batch>, 'unlabeled': <PyG_Batch>}
        batch_l = batch["labeled"]
        batch_u = batch["unlabeled"]

        # Get labels *from* the labeled batch object
        y_l = batch_l.y

        # Ensure teacher model is in 'train' mode to enable stochasticity
        # (e.g., dropout) even though grads are off.
        self.teacher.train()

        # --- 1. Forward Passes (Separate) ---
        # Student passes (with gradients)
        out_student_l = self.student(batch_l)
        out_student_u = self.student(batch_u)

        # Teacher passes (no gradients)
        with torch.inference_mode():
            out_teacher_l = self.teacher(batch_l)
            out_teacher_u = self.teacher(batch_u)

        # --- 2. Loss Calculation ---

        # Part A: Supervised Loss (ONLY on labeled outputs)
        loss_sup = self.sup_loss_fn(out_student_l, y_l)

        # Part B: Consistency Loss
        # Concatenate the OUTPUTS, not the inputs
        out_student_all = torch.cat([out_student_l, out_student_u], dim=0)
        out_teacher_all = torch.cat([out_teacher_l, out_teacher_u], dim=0)

        loss_con = self.con_loss_fn(out_student_all, out_teacher_all)

        # Part C: Total Loss
        total_loss = loss_sup + self.hparams.lambda_w * loss_con

        # --- 3. Logging ---
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_sup_loss", loss_sup, on_step=True, on_epoch=True)
        self.log("train_con_loss", loss_con, on_step=True, on_epoch=True)

        # Log training accuracy on the labeled batch
        # Note: Use out_student_l, not a slice
        self.train_acc(torch.softmax(out_student_l, dim=1), y_l)
        self.log(
            "train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True
        )

        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        This hook is called after the optimizer step.
        This is the perfect place to update the teacher model via EMA.
        """
        alpha = self.hparams.alpha

        # Use torch.no_grad() to ensure no gradients are computed
        with torch.no_grad():
            # Iterate over student and teacher parameters and apply EMA update
            for param_s, param_t in zip(
                self.student.parameters(), self.teacher.parameters()
            ):
                # param_t = alpha * param_t + (1 - alpha) * param_s
                param_t.data.mul_(alpha).add_(param_s.data, alpha=1 - alpha)

    def validation_step(self, batch, batch_idx):
        # 'batch' is the complete PyG Batch object
        self.teacher.eval()

        with torch.inference_mode(): # Use inference_mode (or no_grad)
            preds = self.teacher(batch)
        
        # Get labels from the batch object
        loss = self.sup_loss_fn(preds, batch.y)

        # Log validation metrics
        self.val_acc(torch.softmax(preds, dim=1), batch.y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure the optimizer for the student's parameters.
        """
        optimizer = SGD(
            self.student.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
        return optimizer

import torch
import torch.nn as nn
import torch.nn.functional as F
# Import GATConv
from torch_geometric.nn import GATConv, global_mean_pool

class GATModel(nn.Module):
    """
    A GAT model for graph-level classification that uses edge attributes.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 edge_dim, heads=4, dropout=0.5):
        super().__init__()
        # We need the edge_dim for the GATConv
        
        # Note: GATConv concatenates the heads, so the output
        # of conv1 is hidden_channels * heads
        self.conv1 = GATConv(
            in_channels, 
            hidden_channels, 
            heads=heads, 
            dropout=dropout, 
            edge_dim=edge_dim
        )
        
        # The input to conv2 is hidden_channels * heads
        self.conv2 = GATConv(
            hidden_channels * heads, 
            hidden_channels, 
            heads=heads, 
            dropout=dropout, 
            edge_dim=edge_dim
        )
        
        # The input to the linear layer is also hidden_channels * heads
        self.lin = nn.Linear(hidden_channels * heads, out_channels)
        self.dropout_p = dropout

    def forward(self, batch):
        # Unpack the PyG Batch object, now including edge_attr
        x, edge_index, edge_attr, batch_idx = \
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        
        # Apply GAT layers
        # GATConv can accept edge_attr
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x) # ELU is a common activation with GAT
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # Apply global pooling
        x = global_mean_pool(x, batch_idx)
        
        # Apply the final classifier
        x = self.lin(x)
        return x


        
if __name__ == "__main__":
    from src.data.pcba_datamodule import OGBGDataModule
    import pytorch_lightning as L


    dm = OGBGDataModule(
        ssl_mode=True
    )


# 4. Setup Trainer
trainer = L.Trainer(
    max_epochs=10,
    # fast_dev_run=True,  # Runs 1 train batch and 1 val batch to check for errors
    accelerator="auto",
    devices=1,
    logger=TensorBoardLogger(save_dir=get_logs_dir(), name="baseline_gcn"),
    log_every_n_steps=1,
    enable_progress_bar=True,
    enable_model_summary=True,
    enable_checkpointing=False,
    # default_root_dir=get_logs_dir(),  # Specify a directory for logs and checkpoints
)

# 5. Run Training Test
try:
    trainer.fit(module, datamodule=data_module)
    log.info("--- Training Test Successful! ---")
except Exception as e:
    log.error("--- Training Test FAILED ---")
    log.error(e)
    raise e

