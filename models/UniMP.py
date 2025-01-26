import torch.nn
from torch_geometric.nn import (
    global_mean_pool, global_max_pool, global_add_pool,
    MLP, TransformerConv, BatchNorm
)


class UniMP(torch.nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            num_layers: int,
            num_heads: int,
            metadata: tuple,
            **kwargs
    ):
        super().__init__()
        self.node_lins = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.node_lins[node_type] = MLP(
                in_channels=-1,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=2,
                norm=None,
            )

        self.edge_lins = torch.nn.ModuleDict()
        for edge_type in set([et[1] for et in metadata[1]]):
            self.edge_lins[edge_type] = MLP(
                in_channels=-1,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=2,
                norm=None,
            )

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(TransformerConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels // num_heads,
                edge_dim=hidden_channels,
                heads=num_heads,
                beta=True,
            ))
            self.norms.append(BatchNorm(in_channels=hidden_channels))

        self.out_lin = MLP(
            in_channels=-1,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=2,
            norm=None,
        )

    def forward(self, data, **kwargs):
        node_types, edge_types = data.metadata()
        for node_type in node_types:
            x = data[node_type].x
            data[node_type].x = self.node_lins[node_type](x)

        # aligning the edge feature length
        for edge_type in edge_types:
            edge_attr = data[edge_type].edge_attr
            data[edge_type].edge_attr = self.edge_lins[edge_type[1]](edge_attr)

        # conv operators
        data = data.to_homogeneous()
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            _x = conv(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            _x = self.norms[i](_x)
            x = x + _x.relu()

        # return the result
        emb = torch.cat([
            global_mean_pool(x, batch=data.batch),
            global_add_pool(x, batch=data.batch),
            global_max_pool(x, batch=data.batch),
        ], dim=1)
        return self.out_lin(emb)
