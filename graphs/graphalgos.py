import torch
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool, global_add_pool, BatchNorm, MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree

class GCN(torch.nn.Module):
    """GCN"""
    def __init__(self, num_node_features, num_classes, dim_h):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, dim_h)
        self.bn1 = BatchNorm(dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.bn2 = BatchNorm(dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.bn3 = BatchNorm(dim_h)
        self.lin = Linear(dim_h, num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = h.relu()
        h = self.conv3(h, edge_index)
        h = self.bn3(h)
        h = h.relu()

        # Graph-level readout
        hG = global_mean_pool(h, batch)

        # Classifier
        h = F.dropout(hG, p=0.1, training=self.training)
        h = self.lin(h)
        
        return hG, F.softmax(h, dim=1)
        
class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, num_node_features, dim_h):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h*3, dim_h*3)
        self.lin2 = Linear(dim_h*3, 2)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        return h, F.softmax(h, dim=1)
   
    
class GAT(torch.nn.Module):
    """GAT"""
    def __init__(self, num_node_features, num_classes, dim_h, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, dim_h, heads=num_heads)
        self.conv2 = GATConv(dim_h * num_heads, dim_h, heads=num_heads)
        self.conv3 = GATConv(dim_h * num_heads, dim_h, heads=num_heads)
        self.lin = Linear(dim_h * num_heads, num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.conv2(h, edge_index)
        h = F.elu(h)
        h = self.conv3(h, edge_index)
        h = F.elu(h)

        # Graph-level readout
        hG = global_mean_pool(h, batch)

        # Classifier
        h = F.dropout(hG, p=0.5, training=self.training)
        h = self.lin(h)

        return hG, F.softmax(h, dim=1)



class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = Linear(2048, 512)
        self.relu1 = ReLU()
        self.fc2 = Linear(512, 128)
        self.relu2 = ReLU()
        self.fc3 = Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    


class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNLayer, self).__init__(aggr='add')
        self.lin_message = Linear(in_channels, out_channels)
        self.lin_update = Linear(in_channels + out_channels, out_channels)

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Calculate message and update steps.
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # Calculate messages from neighboring nodes.
        return self.lin_message(x_j)

    def update(self, aggr_out, x):
        # Update node embeddings.
        update_input = torch.cat([aggr_out, x], dim=1)
        return F.relu(self.lin_update(update_input))

class MPNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(MPNN, self).__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        
        # Input layer
        self.layers.append(MPNNLayer(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(MPNNLayer(hidden_channels, hidden_channels))
        
        # Output layer
        self.layers.append(MPNNLayer(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

# Example usage:
# Define the input dimensions, hidden dimensions, output dimensions, and number of layers.
# in_channels = 64
# hidden_channels = 128
# out_channels = 2
# num_layers = 3

# Create an instance of the MPNN model.
# model = MPNN(in_channels, hidden_channels, out_channels, num_layers)

# # Forward pass with input data and edge indices.
# x = torch.randn(64, in_channels)  # Example node features
# edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # Example edge indices
# output = model(x, edge_index)
