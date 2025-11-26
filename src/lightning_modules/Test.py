import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import numpy as np
from tqdm import tqdm
import copy
import os


def load_downsampled_data(dataset_name="ogbg-molpcba", sample_frac=0.1, seed=42):
    """
    Load ogbg-molpcba and create a smaller subset while preserving class distribution
    """
    print(f"Loading {dataset_name}...")
    import torch.serialization
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    
    # Whitelist required PyG classes for unpickling
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
    
    # Now safe to load dataset
    dataset = PygGraphPropPredDataset(name=dataset_name, root="dataset/")
    split_idx = dataset.get_idx_split()
    
    np.random.seed(seed)
    train_idx = np.random.choice(split_idx["train"], int(len(split_idx["train"]) * sample_frac), replace=False)
    valid_idx = np.random.choice(split_idx["valid"], int(len(split_idx["valid"]) * sample_frac), replace=False)
    test_idx = np.random.choice(split_idx["test"], int(len(split_idx["test"]) * sample_frac), replace=False)
    
    print(f"Original size - Train: {len(split_idx['train'])}, Valid: {len(split_idx['valid'])}, Test: {len(split_idx['test'])}")
    print(f"Downsampled sizes - Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")
    
    return dataset, {"train": train_idx, "valid": valid_idx, "test": test_idx}


def calculate_pos_weights(dataset, train_idx, max_weight=100):
    """
    Calculate pos_weight for BCEWithLogitsLoss based on class distribution
    pos_weight = num_negative_samples / num_positive_samples
    """
    # Collect all labels from training set
    all_labels = []
    for idx in train_idx:
        labels = dataset[int(idx)].y
        all_labels.append(labels)
    
    all_labels = torch.cat(all_labels, dim=0)  # Shape: [num_samples, num_tasks]
    
    # Count positives and negatives per task (Ignore NaN values)
    pos_counts = torch.zeros(all_labels.shape[1])
    neg_counts = torch.zeros(all_labels.shape[1])
    
    for task_idx in range(all_labels.shape[1]):
        task_labels = all_labels[:, task_idx]
        # Filtering NaN values (NaN means missing labels)
        valid_mask = ~torch.isnan(task_labels)
        valid_labels = task_labels[valid_mask]
        
        if len(valid_labels) > 0:
            pos_counts[task_idx] = (valid_labels == 1).sum()
            neg_counts[task_idx] = (valid_labels == 0).sum()
    
    # Calculate pos_weight (avoid division by zero)
    pos_weights = torch.zeros_like(pos_counts)
    for i in range(len(pos_counts)):
        if pos_counts[i] > 0:
            pos_weights[i] = neg_counts[i] / pos_counts[i]
        else:
            pos_weights[i] = 1.0  # Default weight if no positive samples
    
    # Cap the max weight
    pos_weights = torch.clamp(pos_weights, min=1.0, max=max_weight)
    
    print("\nClass Distribution Analysis:")
    print(f"Average positive ratio: {(pos_counts.sum() / (pos_counts.sum() + neg_counts.sum())):.4f}")
    print(f"Pos weights - Min: {pos_weights.min():.2f}, Max: {pos_weights.max():.2f}, Mean: {pos_weights.mean():.2f}")
    
    return pos_weights


def augment_graph(data, node_feature_drop_prob=0.1, edge_noise_std=0.02):
    """
    Apply graph augmentations for training robustness.
    """
    # Create a copy to avoid modifying the original
    data = copy.copy(data)
    
    # Node feature dropout
    if node_feature_drop_prob > 0 and hasattr(data, 'x') and data.x is not None:
        mask = torch.rand(data.x.size()) > node_feature_drop_prob
        data.x = data.x * mask.float()
    
    # Edge attribute noise
    if edge_noise_std > 0 and hasattr(data, 'edge_attr') and data.edge_attr is not None:
        data.edge_attr = data.edge_attr.float()
        noise = torch.randn_like(data.edge_attr) * edge_noise_std
        data.edge_attr = data.edge_attr + noise
    
    return data


class GINE(nn.Module):
    def __init__(self, num_tasks, num_node_features=9, num_edge_features=3,
                 hidden_dim=256, num_layers=5, dropout=0.3, use_residual=True):
        super().__init__()
        self.dropout = dropout
        self.use_residual = use_residual
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Edge MLP (shared for each layer)
        self.edge_mlp = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # First layer
        mlp = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Remaining layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Attention pooling
        self.attn_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )
        
        self.classifier = nn.Linear(hidden_dim * 4, num_tasks)
    
    def forward(self, data):
        x = data.x.float()
        edge_attr = data.edge_attr.float()
        edge_index = data.edge_index
        batch = data.batch
        
        # Encode edge attributes once per forward pass
        edge_attr_encoded = self.edge_mlp(edge_attr)
        
        # First conv layer (no residual connection)
        x = self.convs[0](x, edge_index, edge_attr_encoded)
        x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Remaining conv layers (with optional residual)
        for conv, bn in zip(self.convs[1:], self.batch_norms[1:]):
            x_new = conv(x, edge_index, edge_attr_encoded)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            if self.use_residual:
                x = x + x_new
            else:
                x = x_new
        
        # Pooling
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_attn = self.attn_pool(x, batch)
        
        x = torch.cat([x_mean, x_add, x_max, x_attn], dim=1)
        return self.classifier(x)


def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    
    for data in tqdm(loader, desc="Training"):
        data = augment_graph(data, node_feature_drop_prob=0.1, edge_noise_std=0.02).to(device)
        optimizer.zero_grad()
        out = model(data)
        
        # Handle NaN labels
        is_labeled = ~torch.isnan(data.y)
        
        # Compute all losses
        all_losses = loss_fn(out, torch.where(is_labeled, data.y, torch.zeros_like(data.y)))
        loss = (all_losses * is_labeled.float()).sum() / is_labeled.float().sum()
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


def eval(model, loader, evaluator, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.inference_mode():
        for data in tqdm(loader, desc="Evaluating"):
            data = data.to(device)
            out = model(data)
            y_true.append(data.y.cpu())
            y_pred.append(out.cpu())
    
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    
    return evaluator.eval(input_dict)


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    SAMPLE_FRAC = 0.5  # Use 50% of data
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 0.001
    
    # Load data
    dataset, split_idx = load_downsampled_data(sample_frac=SAMPLE_FRAC)
    
    # Calculate pos_weights for imbalanced dataset
    pos_weights = calculate_pos_weights(dataset, split_idx['train'], max_weight=100).to(device)
    
    # Create data loaders
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    num_tasks = dataset.num_tasks
    num_node_features = dataset.num_features
    num_edge_features = dataset[0].edge_attr.shape[1] if hasattr(dataset[0], 'edge_attr') else 0
    
    print(f"pos_weights shape: {pos_weights.shape}")
    print(f"num_tasks: {num_tasks}")
    
    # Initialize model
    model = GINE(
        num_tasks=num_tasks,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=300,
        num_layers=5,
        dropout=0.2,
        use_residual=True
    ).to(device)
    
    # Loss function with pos_weight for class imbalance
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    
    # Evaluator
    evaluator = Evaluator(name='ogbg-molpcba')
    
    # Training loop
    best_valid_ap = 0
    print("\nStarting training...")
    
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        
        if epoch % 5 == 0:
            train_result = eval(model, train_loader, evaluator, device)
            valid_result = eval(model, valid_loader, evaluator, device)
            
            train_ap = train_result['ap']
            valid_ap = valid_result['ap']
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f'Epoch {epoch:03d}: Loss: {loss:.4f}, '
                  f'Train AP: {train_ap:.4f}, Valid AP: {valid_ap:.4f}, '
                  f'LR: {current_lr:.6f}')
            
            if valid_ap > best_valid_ap:
                best_valid_ap = valid_ap
                torch.save(model.state_dict(), 'best_model.pt')
                print(f"  → New best model saved!")
    
    # Final evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load('best_model.pt'))
    test_result = eval(model, test_loader, evaluator, device)
    print(f'\nFinal Test AP: {test_result["ap"]:.4f}')
    print(f'Best Valid AP: {best_valid_ap:.4f}')


if __name__ == '__main__':
    main()