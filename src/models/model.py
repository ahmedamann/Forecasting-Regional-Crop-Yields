class CropYieldModel(nn.Module):
    def __init__(self, input_dim):
        super(CropYieldModel, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.1)
        
        # Hidden layers
        self.hidden1 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.1)
        
        self.hidden2 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.1)
        
        # Output layer
        self.output_layer = nn.Linear(32, 1)
        
    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Hidden layers
        x = self.hidden1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.hidden2(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.output_layer(x)
        return x 