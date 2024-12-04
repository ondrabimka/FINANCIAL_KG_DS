import torch
from torch_geometric.nn import GAE, VGAE, GCNConv

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    
class VariationalGCNDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, z, edge_index):
        return self.conv1(z, edge_index).relu()
    
class VariationalGCN(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_index):
        return self.decoder(z, edge_index)

    def forward(self, x, edge_index):
        mu, logstd = self.encode(x, edge_index)
        z = self.reparameterize(mu, logstd)
        return self.decode(z, edge_index), mu, logstd
    
class VariationalGraphAutoEncoder:
    def __init__(self, in_channels, out_channels):
        self.encoder = VariationalGCNEncoder(in_channels, out_channels)
        self.decoder = VariationalGCNDecoder(out_channels, in_channels)
        self.model = VariationalGCN(self.encoder, self.decoder)
        self.reconstruction_loss = torch.nn.BCEWithLogitsLoss()
        self.kl_loss = torch.nn.KLDivLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train(self, data, epochs):
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            z, mu, logstd = self.model(data.x, data.edge_index)
            print(z, mu, logstd)
            loss = self.reconstruction_loss(z, data.x) + self.kl_loss(mu, logstd)
            print(loss)
            loss.backward()
            self.optimizer.step()
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

    def test(self, data):
        self.model.eval()
        z, _, _ = self.model(data.x, data.edge_index)
        return z

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def get_embeddings(self, data):
        return self.test(data)

    def get_model(self):
        return self.model

    def get_loss(self):
        return self.reconstruction_loss

    def get_kl_loss(self):
        return self.kl_loss