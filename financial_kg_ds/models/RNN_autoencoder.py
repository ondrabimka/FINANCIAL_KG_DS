# %%
import torch
import torch.nn as nn

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, (hidden, cell) = self.lstm(x, (h0, c0))
        return out, hidden, cell

class RNNDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout=0.5):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Linear layer to map to output_dim

    def forward(self, x, hidden, cell):
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        out = self.fc(out)  # Project back to output dimension
        return out

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = RNNEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = RNNDecoder(hidden_dim, input_dim, num_layers, dropout)

    def forward(self, x):
        # Encode
        encoder_out, hidden, cell = self.encoder(x)
        
        # Use encoder's output as decoder's input for the initial time step
        out = self.decoder(encoder_out, hidden, cell)
        return out
    
    def get_embedding(self, x):
        out, _, _ = self.encoder(x)
        return out[:, -1, :]  # Last timestep's output for each batch

#%%
if __name__ == "__main__":
    # Testing the model
    input_size = 1
    hidden_size = 6
    num_layers = 2
    seq_len = 5
    batch_size = 2

    model = LSTMAutoencoder(input_size, hidden_size, num_layers)
    input = torch.randn(batch_size, seq_len, input_size)
    decoded = model(input)