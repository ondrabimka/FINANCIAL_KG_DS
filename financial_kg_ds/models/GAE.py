import torch
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import negative_sampling


class HeteroGCNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(HeteroGCNEncoder, self).__init__()
        self.conv1 = HeteroConv(
            {
                ("ticker", "holds_it", "institution"): SAGEConv((-1, -1), hidden_channels),
                ("ticker", "holds_mt", "mutual_fund"): SAGEConv((-1, -1), hidden_channels),
                ("news", "about_nt", "ticker"): SAGEConv((-1, -1), hidden_channels),
                ("mutual_fund", "rev_holds_mt", "ticker"): SAGEConv((-1, -1), hidden_channels),
                ("institution", "rev_holds_it", "ticker"): SAGEConv((-1, -1), hidden_channels),
                ("ticker", "rev_about_nt", "news"): SAGEConv((-1, -1), hidden_channels),
            },
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            {
                ("ticker", "holds_it", "institution"): SAGEConv((-1, -1), hidden_channels),
                ("ticker", "holds_mt", "mutual_fund"): SAGEConv((-1, -1), hidden_channels),
                ("news", "about_nt", "ticker"): SAGEConv((-1, -1), hidden_channels),
                ("mutual_fund", "rev_holds_mt", "ticker"): SAGEConv((-1, -1), hidden_channels),
                ("institution", "rev_holds_it", "ticker"): SAGEConv((-1, -1), hidden_channels),
                ("ticker", "rev_about_nt", "news"): SAGEConv((-1, -1), hidden_channels),
            },
            aggr="sum",
        )

    def forward(self, x_dict, edge_index_dict):
        # First convolutional layer
        conv_out = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}  # Apply ReLU
        x_dict = self.conv2(x_dict, edge_index_dict)
        return conv_out


# create decoder using SAGEConv
class HeteroGraphDecoder(torch.nn.Module):
    def __init__(self):
        super(HeteroGraphDecoder, self).__init__()
        self.conv = HeteroConv(
            {
                ("ticker", "holds_it", "institution"): SAGEConv((-1, -1), 2401),
                ("ticker", "holds_mt", "mutual_fund"): SAGEConv((-1, -1), 1798),
                ("news", "about_nt", "ticker"): SAGEConv((-1, -1), 3658),
                ("mutual_fund", "rev_holds_mt", "ticker"): SAGEConv((-1, -1), 3658),
                ("institution", "rev_holds_it", "ticker"): SAGEConv((-1, -1), 3658),
                ("ticker", "rev_about_nt", "news"): SAGEConv((-1, -1), 19340),
            },
            aggr="sum",
        )

    def forward(self, z_dict, edge_index_dict):
        z_dict = self.conv(z_dict, edge_index_dict)
        return z_dict


class SimpleHeteroGAE(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(SimpleHeteroGAE, self).__init__()
        self.encoder = HeteroGCNEncoder(hidden_channels)
        self.decoder = HeteroGraphDecoder()

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def forward(self, x_dict, edge_index_dict):
        z_dict = self.encode(x_dict, edge_index_dict)
        return self.decode(z_dict, edge_index_dict)
