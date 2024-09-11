import torch
import torch.nn as nn
import torch.optim as optim
import logging


logger = logging.getLogger(__name__)


class MultiLabelNN(nn.Module):

    def __init__(self, input_size, output_size, device=None):
        super(MultiLabelNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_size)

        # Initialize the loss function and optimizer
        self.criterion = nn.BCELoss()  # Binary cross-entropy for multi-label
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.label_map = None

        # Initialize device (CPU or GPU)
        self.device = device if device else torch.device('cpu')
        self.to(self.device)

    def set_label_map(self, label_map):
        self.label_map = label_map

    def get_label_map(self):
        return self.label_map

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))  # Sigmoid for multi-label classification
        return x

    def partial_fit(self, new_data, new_labels, epochs=1):
        self.train()  # Set the model to training mode

        # Ensure the new_data and new_labels are on the same device as the model
        new_data = torch.stack(new_data) if isinstance(new_data, list) else new_data
        new_data = new_data.to(self.device)  # Move data to the model's device
        new_labels = torch.tensor(new_labels, dtype=torch.float32).to(self.device)  # Move labels to the same device
        outputs = None

        for epoch in range(epochs):
            self.optimizer.zero_grad()  # Zero out gradients from previous steps
            outputs = self(new_data)
            loss = self.criterion(outputs, new_labels)  # Calculate the loss
            loss.backward()  # Backpropagation
            self.optimizer.step()  # Update model parameters
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        return outputs

    def predict(self, new_data, threshold=None):
        new_data = torch.stack(new_data) if isinstance(new_data, list) else new_data
        new_data = new_data.to(self.device)  # Move data to the model's device
        
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            new_data = torch.tensor(new_data, dtype=torch.float32)
            outputs = self(new_data)
            if threshold is None:
                return outputs

        return (outputs > threshold).int()  # Apply threshold to get binary labels

    def save(self, path):
        # torch.save(self.state_dict(), path)
        torch.save(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        # self.load_state_dict(torch.load(path))
        model = torch.load(path)
        model.eval()  # Set the model to evaluation mode
        logger.info(f"Model loaded from {path}")
        return model


class MultiLabelLSTM(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=32, num_layers=1, device=None):
        super(MultiLabelLSTM, self).__init__()

        # LSTM layer for handling sequential data
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, output_size)

        # Initialize the loss function and optimizer
        self.criterion = nn.BCELoss()  # Binary cross-entropy for multi-label classification
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.label_map = None

        # Initialize device (CPU or GPU)
        self.device = device if device else torch.device('cpu')
        self.to(self.device)

    def set_label_map(self, label_map):
        self.label_map = label_map

    def get_label_map(self):
        return self.label_map

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out contains outputs for all timesteps

        # Use the last time step output for classification
        out = lstm_out[:, -1, :]  # Take the last timestep for each batch

        out = torch.sigmoid(self.fc(out))  # Fully connected layer and Sigmoid activation for multi-label classification
        return out

    def partial_fit(self, new_data, new_labels, epochs=1):
        self.train()  # Set the model to training mode

        # Ensure new_data and new_labels are on the same device as the model
        new_data = torch.stack(new_data) if isinstance(new_data, list) else new_data
        new_data = new_data.to(self.device)  # Move data to the model's device
        new_labels = torch.tensor(new_labels, dtype=torch.float32).to(self.device)  # Move labels to the same device
        outputs = None

        for epoch in range(epochs):
            self.optimizer.zero_grad()  # Zero out gradients from previous steps
            outputs = self(new_data)
            loss = self.criterion(outputs, new_labels)  # Calculate the loss
            loss.backward()  # Backpropagation
            self.optimizer.step()  # Update model parameters
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        return outputs

    def predict(self, new_data, threshold=None):
        new_data = torch.stack(new_data) if isinstance(new_data, list) else new_data
        new_data = new_data.to(self.device)  # Move data to the model's device

        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            outputs = self(new_data)
            if threshold is None:
                return outputs

        return (outputs > threshold).int()  # Apply threshold to get binary labels

    def save(self, path):
        torch.save(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        model.eval()  # Set the model to evaluation mode
        logger.info(f"Model loaded from {path}")
        return model