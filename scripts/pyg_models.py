"""Fixed-weight PyG GCN and GraphSAGE using standard GCNConv and SAGEConv layers."""
import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import remove_self_loops


PREPROCESSING = (
    'Standard PyG GCNConv (symmetric normalization with unit self-loops) '
    'and SAGEConv (mean aggregation over non-self neighbors with root weight); '
    'native weight orientation transposed to PyG Linear convention; '
    'bias and activation applied per layer'
)


class Layer(nn.Module):
    def __init__(self, specification):
        super().__init__()
        self.kind = specification['kind']
        self.activation = specification['activation']
        w_neigh = specification['w_neigh']
        w_self = specification['w_self']
        bias = specification['bias']
        in_dim, out_dim = w_neigh.shape
        has_bias = bias is not None

        if self.kind == 'GCN':
            conv = GCNConv(in_dim, out_dim, add_self_loops=True, normalize=True, bias=has_bias)
            conv.lin.weight.data.copy_(w_neigh.t())
            if has_bias:
                conv.bias.data.copy_(bias.reshape(-1))
            self.conv = conv
        elif self.kind == 'GRAPHSAGE':
            has_self = w_self is not None
            conv = SAGEConv(in_dim, out_dim, aggr='mean', root_weight=has_self, bias=has_bias)
            conv.lin_l.weight.data.copy_(w_neigh.t())
            if has_self:
                conv.lin_r.weight.data.copy_(w_self.t())
            if has_bias:
                conv.lin_l.bias.data.copy_(bias.reshape(-1))
            self.conv = conv
        else:
            raise ValueError(f"Unknown layer kind: {self.kind}")

    def forward(self, x, edge_index, edge_weight, edge_index_no_self):
        if self.kind == 'GCN':
            out = self.conv(x, edge_index, edge_weight=edge_weight)
        else:
            out = self.conv(x, edge_index_no_self)
        if self.activation == 'RELU':
            out = out.relu()
        return out


class Model(nn.Module):
    def __init__(self, tensors):
        super().__init__()
        edge_index = tensors['edge_index']
        edge_weight = tensors['edge_weight']
        edge_index_no_self, _ = remove_self_loops(edge_index)

        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)
        self.register_buffer('edge_index_no_self', edge_index_no_self)

        self.layers = nn.ModuleList(Layer(layer) for layer in tensors['layers'])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, self.edge_index, self.edge_weight, self.edge_index_no_self)
        return x


def build(tensors):
    return Model(tensors)

