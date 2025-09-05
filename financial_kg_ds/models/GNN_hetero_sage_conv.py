from torch_geometric.nn import  HeteroConv, SAGEConv, BatchNorm
from torch_geometric.nn import Linear
import torch
import torch.nn.functional as F

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers, gnn_aggr="add", dropout=0.2):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        
        # Create input projections for each node type to normalize feature dimensions
        node_types, edge_types = metadata
        self.input_projections = torch.nn.ModuleDict()
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # First layer - handle different input dimensions
        first_conv_dict = {}
        for edge_type in edge_types:
            src_type, _, dst_type = edge_type
            # Use actual input dimensions or project to hidden_channels
            first_conv_dict[edge_type] = SAGEConv((hidden_channels, hidden_channels), hidden_channels, aggr=gnn_aggr)
        
        self.convs.append(HeteroConv(first_conv_dict))
        
        # Remaining layers - all use hidden_channels
        for i in range(1, num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((hidden_channels, hidden_channels), hidden_channels, aggr=gnn_aggr)
                    for edge_type in edge_types
                }
            )
            self.convs.append(conv)
            
        # Add batch normalization for each node type
        for i in range(num_layers):
            batch_norm = torch.nn.ModuleDict({
                node_type: BatchNorm(hidden_channels) 
                for node_type in node_types
            })
            self.batch_norms.append(batch_norm)

        # Output layer with residual connection option
        self.lin_out = Linear(hidden_channels, out_channels)
        
        # Optional skip connections
        if num_layers > 1:
            self.skip_connection = Linear(hidden_channels, hidden_channels)
        else:
            self.skip_connection = None

    def forward(self, x_dict, edge_index_dict):
        # Project all node features to hidden_channels first
        projected_x = {}
        for node_type, x in x_dict.items():
            if node_type not in self.input_projections:
                # Create projection layer dynamically if not exists
                self.input_projections[node_type] = Linear(x.shape[1], self.hidden_channels).to(x.device)
            projected_x[node_type] = self.input_projections[node_type](x)
        
        x_dict = projected_x
        
        # Store initial embeddings for potential skip connections
        initial_x = None
        
        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            if i == 0:
                initial_x = {key: x.clone() for key, x in x_dict.items()}
            
            # Graph convolution
            x_dict = conv(x_dict, edge_index_dict)
            
            # Batch normalization
            for node_type in x_dict.keys():
                if node_type in batch_norm:
                    x_dict[node_type] = batch_norm[node_type](x_dict[node_type])
            
            # Activation
            x_dict = {key: F.leaky_relu(x, negative_slope=0.1) for key, x in x_dict.items()}
            
            # Skip connection for deeper networks (layer 1 onwards)
            if i > 0 and self.skip_connection is not None and i == self.num_layers - 1:
                for node_type in x_dict.keys():
                    if node_type in initial_x:
                        skip = self.skip_connection(initial_x[node_type])
                        x_dict[node_type] = x_dict[node_type] + skip
            
            # Dropout
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                     for key, x in x_dict.items()}
        
        # Output layer - only for ticker nodes
        return self.lin_out(x_dict["ticker"])