import torch
from torch.nn.functional import mse_loss
from torch_geometric.nn import MLP


class AELoss(torch.nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            projection_dim: int = 256,
            **kwargs
    ):
        super().__init__()
        self.encoder = MLP(
            in_channels=embedding_dim,
            hidden_channels=(embedding_dim + projection_dim) // 2,
            out_channels=projection_dim,
            num_layers=2,
            norm='batch_norm',
            dropout=0.3,
        )
        self.decoder = MLP(
            in_channels=projection_dim,
            hidden_channels=(embedding_dim + projection_dim) // 2,
            out_channels=embedding_dim,
            num_layers=2,
            norm='batch_norm',
            dropout=0.3,
        )

    def forward(self, graph_features):
        projection = self.encoder(graph_features)
        reconstruction = self.decoder(projection)
        return mse_loss(graph_features, reconstruction)
