from src.data.ogbg_molpcba_datamodule import OgbgMolpcbaDataModule
from torch_geometric.nn import GCN
from src.utils.utils import get_logs_dir

# Read config from hydra
import pytorch_lightning as L
from src.lightning_modules.baseline_module import BaselineGNNModule
from torch_geometric.transforms import Compose, AddSelfLoops, NormalizeFeatures
from torch_geometric.nn import global_mean_pool

from src.utils.transforms import ToFloatTransform
import logging
import torch 
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder # Importante

log = logging.getLogger(__name__)


graph_transforms = Compose(
    [
        # ToFloatTransform(),
        AddSelfLoops(),
        # NormalizeFeatures(),
   
    ]
)


# class GCN2(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
#         super().__init__()
#         self.model = GCN(
#             in_channels=in_channels,
#             hidden_channels=hidden_channels,
#             num_layers=num_layers,
#             out_channels=out_channels,
#             dropout=0.2,
#             add_self_loops=False
#         )
    
#     def forward(self, x, edge_index, batch):
#         x = self.model(x, edge_index)
#         x = global_mean_pool(x, batch)
#         return x


class GNN_MolPCBA(nn.Module):
    def __init__(self, num_layers, hidden_dim, out_dim):
        super(GNN_MolPCBA, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # 1. Encoders para las características de los nodos y aristas (Átomos y Enlaces)
        self.atom_encoder = AtomEncoder(hidden_dim)
        # El dataset ogbg-molpcba utiliza 9 características de átomo
        # y 3 características de enlace por defecto.

        # 2. Capas GIN
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # MLPs para GIN
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.BatchNorm1d(2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim)
            )
            conv = GINConv(mlp, train_eps=True)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # 3. Capa de Salida para clasificación multiclase
        # El 'out_dim' es el número de tareas (128 para ogbg-molpcba)
        self.pool = global_add_pool
        self.final_linear = nn.Linear(hidden_dim, out_dim)


    def forward(self, data):
        # x: características del átomo, edge_index: conectividad, edge_attr: características del enlace
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Codificar características iniciales de los nodos
        x = self.atom_encoder(x)
        
        # Propagación del mensaje (Capas GNN)
        h_list = [x]
        for i in range(self.num_layers):
            h = self.convs[i](h_list[i], edge_index)
            h = self.batch_norms[i](h)
            h = nn.functional.relu(h)
            
            # Skip connection (como el de GIN original)
            h = h + h_list[i]
            
            h_list.append(h)

        # Agregar todas las representaciones de capa para obtener la representación final del nodo
        x = sum(h_list)

        # Global Pooling (se recomienda global_add_pool o global_mean_pool)
        h_graph = self.pool(x, batch)

        # Capa de clasificación
        out = self.final_linear(h_graph)
        
        return out


data_module = OgbgMolpcbaDataModule(
    batch_size=128,
    num_workers=4,

)

data_module.setup(stage="fit")


# 3. Setup Lightning Module
module = BaselineGNNModule(
    model=GNN_MolPCBA(num_layers=5, hidden_dim=300, out_dim=128),
    num_classes=data_module.num_classes,
    learning_rate=1e-3,
    warmup_epochs=1,
    cosine_period_ratio=0.9,
    compile_mode=None,
    optimizer="SGD",
    weight_decay=1e-5,
    train_transforms=graph_transforms,
    val_transforms=graph_transforms,
    test_transforms=graph_transforms,
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
