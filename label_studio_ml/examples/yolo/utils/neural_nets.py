import torch
import torch.nn as nn
import torch.optim as optim
import logging

from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List


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

    def preprocess_sequence(self, sequence: List[torch.Tensor], labels=None, overlap=2):
        sequence = torch.stack(sequence) if isinstance(sequence, list) else sequence
        sequence_size = self.sequence_size

        # Split the data into small sequences by sequence_size with overlap
        chunks = [sequence[i:i + sequence_size] for i in range(0, len(sequence), sequence_size // overlap)]
        chunks = pad_sequence(chunks, batch_first=True, padding_value=0)

        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.float32)
            labels = [labels[i:i + sequence_size] for i in range(0, len(labels), sequence_size // overlap)]
            labels = pad_sequence(labels, batch_first=True, padding_value=0) if labels is not None else None

        return chunks, labels

    def partial_fit(self, sequence, labels, batch_size=32, epochs=1):
        self.train()  # Set the model to training mode
        batches, label_batches = self.preprocess_sequence(sequence, labels)

        # Create a DataLoader for batching the input data
        outputs = None
        dataset = TensorDataset(batches, torch.tensor(label_batches, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_data, batch_labels in dataloader:
                # Move batch data and labels to the appropriate device
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self(batch_data)  # Forward pass
                loss = self.criterion(outputs, batch_labels)  # Calculate loss
                loss.backward()  # Back propagation
                self.optimizer.step()  # Update model parameters

                epoch_loss += loss.item()

            print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}')

        return outputs

    def predict(self, sequence):
        """ Split sequence into chunks with sequence_size and predict by chunks.
        Then concatenate all predictions into one sequence of labels
        """
        length = len(sequence)
        if length == 0:
            return torch.tensor([])

        batches, _ = self.preprocess_sequence(sequence, overlap=1)
        logits = self(batches)
        
        # Concatenate batches to sequence back
        shape = logits.shape
        logits = torch.reshape(logits, [shape[0] * shape[1], shape[2]])
        return logits[0: length]
