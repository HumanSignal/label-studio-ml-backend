import torch
import torch.nn as nn
import torch.optim as optim
import logging

from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)

class BaseNN(nn.Module):
    def set_label_map(self, label_map):
        self.label_map = label_map

    def get_label_map(self):
        return self.label_map

    def save(self, path):
        torch.save(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        model.eval()  # Set the model to evaluation mode
        logger.info(f"Model loaded from {path}")
        return model


class MultiLabelNN(BaseNN):

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


class MultiLabelLSTM(BaseNN):

    def __init__(
            self,
            input_size,
            output_size,
            fc_size=128,
            hidden_size=16,
            num_layers=1,
            sequence_size=64,
            device=None
    ):
        super(MultiLabelLSTM, self).__init__()

        # Split the input data into sequences of sequence_size
        self.sequence_size = sequence_size

        # Reduce dimensionality of input data
        self.fc_input = nn.Linear(input_size, fc_size)

        # LSTM layer for handling sequential data
        self.lstm = nn.LSTM(fc_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Fully connected layer for classification at each time step
        self.fc = nn.Linear(2 * hidden_size, output_size)  # 2 because of bidirectional LSTM

        # Initialize the loss function and optimizer
        self.criterion = nn.BCELoss()  # Binary cross-entropy for multi-label classification
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

        # Initialize device (CPU or GPU)
        self.device = device if device else torch.device('cpu')
        self.to(self.device)

    def forward(self, x):
        # Reduce the dimensionality of the input data
        x = torch.relu(self.fc_input(x))
        
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out contains outputs for all time steps

        # Apply fully connected layer at each time step to get output for each step
        out = torch.sigmoid(self.fc(lstm_out))  # Sigmoid activation for multi-label classification

        # Output shape: (batch_size, seq_len, output_size)
        return out

    def partial_fit(self, new_data, new_labels, batch_size=32, epochs=1):
        self.train()  # Set the model to training mode
        sequence_size = self.sequence_size

        new_data = torch.stack(new_data) if isinstance(new_data, list) else new_data
        new_labels = torch.tensor(new_labels, dtype=torch.float32)

        # Split the data into small sequences by sequence_size with 1/2 overlap
        new_data = [new_data[i:i + sequence_size] for i in range(0, len(new_data), sequence_size//2)]
        new_data = pad_sequence(new_data, batch_first=True, padding_value=0)
        new_labels = [new_labels[i:i + sequence_size] for i in range(0, len(new_labels), sequence_size//2)]
        new_labels = pad_sequence(new_labels, batch_first=True, padding_value=0)

        # Create a DataLoader for batching the input data
        outputs = None
        dataset = TensorDataset(new_data, torch.tensor(new_labels, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_data, batch_labels in dataloader:
                # Move batch data and labels to the appropriate device
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Zero out gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self(batch_data)

                # Calculate loss
                loss = self.criterion(outputs, batch_labels)

                # Backpropagation
                loss.backward()

                # Update model parameters
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}')

        return outputs

    def predict(self, new_data, threshold=None):
        return self.predict_in_batches(new_data)

        new_data = torch.stack(new_data) if isinstance(new_data, list) else new_data
        new_data = new_data.to(self.device)  # Move data to the model's device

        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            new_data = torch.tensor(new_data, dtype=torch.float32)
            outputs = self(new_data)
            if threshold is None:
                return outputs

        return (outputs > threshold).int()  # Apply threshold to get binary labels

    def predict_in_batches(self, sequence):
        sequence_len = len(sequence)
        predictions = []

        # Loop over the sequence in chunks of self.sequence_size
        for i in range(0, sequence_len, self.sequence_size):
            # Get a chunk of the sequence
            chunk = torch.stack(sequence[i:i + self.sequence_size])

            # If the chunk is smaller than self.sequence_size, pad it with zeros
            if len(chunk) < self.sequence_size:
                padding = torch.zeros(self.sequence_size - len(chunk), chunk.size(1))
                chunk = torch.cat((chunk, padding), dim=0)

            # Get predictions for the chunk
            preds = self(chunk.unsqueeze(0))  # Add batch dimension
            predictions.append(preds)

        return torch.cat(predictions, dim=1)[0]  # Concatenate all predictions

