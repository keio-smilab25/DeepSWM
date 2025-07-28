import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFlareNet(nn.Module):
    """
    A model that performs feature extraction using a 4-layer neural network
    """

    def __init__(self, input_dim, class_weights=None):
        super(DeepFlareNet, self).__init__()

        self.class_weights = class_weights

        # Network structure: number of nodes
        self.layers_dims = [input_dim, 200, 200, input_dim, 200, 200, input_dim, 200, 2]

        # Layer definitions
        self.fc1 = nn.Linear(self.layers_dims[0], self.layers_dims[1])
        self.bn1 = nn.BatchNorm1d(self.layers_dims[1])
        self.fc2 = nn.Linear(self.layers_dims[1], self.layers_dims[2])
        self.bn2 = nn.BatchNorm1d(self.layers_dims[2])
        self.fc3 = nn.Linear(self.layers_dims[2], self.layers_dims[3])
        self.bn3 = nn.BatchNorm1d(self.layers_dims[3])
        self.fc4 = nn.Linear(self.layers_dims[3], self.layers_dims[4])
        self.bn4 = nn.BatchNorm1d(self.layers_dims[4])
        self.fc5 = nn.Linear(self.layers_dims[4], self.layers_dims[5])
        self.bn5 = nn.BatchNorm1d(self.layers_dims[5])
        self.fc6 = nn.Linear(self.layers_dims[5], self.layers_dims[6])
        self.bn6 = nn.BatchNorm1d(self.layers_dims[6])
        self.fc7 = nn.Linear(self.layers_dims[6], self.layers_dims[7])
        self.fc_out = nn.Linear(self.layers_dims[7], self.layers_dims[8])

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        # Input layer
        x1 = self.dropout(F.relu(self.fc1(x)))

        # Intermediate layers with residual connections
        x2 = F.relu(self.bn1(self.fc2(x1)))
        x3 = F.relu(self.bn2(self.fc3(x2))) + x
        x4 = F.relu(self.bn3(self.fc4(x3)))
        x5 = F.relu(self.bn4(self.fc5(x4)))
        x6 = F.relu(self.bn5(self.fc6(x5))) + x
        x7 = F.relu(self.bn6(self.fc7(x6)))

        # Output layer
        self.extracted_features = x7  # Save the features
        out = F.softmax(self.fc_out(x7), dim=1)

        return out

    def get_extracted_features(self):
        """Returns the extracted features"""
        return self.extracted_features
