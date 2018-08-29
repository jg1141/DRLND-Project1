import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=[64, 64], dropout_percentage=0.3):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list of int): Width of hidden layers
            dropout_percentage (float): Percentage of nodes to drop out
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_size = state_size
        self.output_size = action_size
        self.hidden_layers = hidden_layers
        self.dropout_percentage = dropout_percentage
        
        self.layers = nn.ModuleList([nn.Linear(self.input_size, self.hidden_layers[0])])
        self.layer_sizes = [(self.hidden_layers[0], self.hidden_layers[1]),]
        self.layers.extend(nn.Linear(size_1, size_2) for size_1, size_2 in self.layer_sizes)
        
        self.output = nn.Linear(self.hidden_layers[-1], self.output_size)
        
        self.dropout = nn.Dropout(self.dropout_percentage)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        return self.output(x)