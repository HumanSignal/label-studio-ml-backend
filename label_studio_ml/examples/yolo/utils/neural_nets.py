import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging

from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Union
from joblib import Memory


logger = logging.getLogger(__name__)
memory = Memory("./cache_dir", verbose=1)  # Set up disk-based caching for model results
_models = {}


@memory.cache(ignore=["yolo_model"])
def cached_feature_extraction(yolo_model, video_path, cache_params):
    """Extract features from the last layer of the YOLO model and cache them using joblib.
    Args:
        yolo_model (YOLO): YOLO model instance
        video_path (str): Path to the video file
        cache_params (str): Parameters for caching the results, they are used in @memory.cache decorator
    """
    layer_output = [None]

    def get_last_layer_output(module, input, output):
        layer_output[0] = input

    # Register the hook on the last layer of the model
    layer = yolo_model.model.model[-1].linear
    hook_handle = layer.register_forward_hook(get_last_layer_output)

    # Run model prediction
    generator = yolo_model.predict(
        video_path, stream=True
    )  # use stream to avoid out of memory

    # Replace probs with last layer outputs
    frames = []
    for frame in generator:
        frame.orig_img = None  # remove image from cache to reduce size
        frame.probs = layer_output[0][0][0]  # => tensor, 1280 floats for yolov8n-cls
        frames.append(frame)

    # Remove the hook
    hook_handle.remove()
    return frames


class BaseNN(nn.Module):
    def __init__(self):
        super(BaseNN, self).__init__()
        self.label_map = None

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

    @staticmethod
    def load_cached_model(model_path: str):
        global _models
        if not os.path.exists(model_path):
            return None

        # Load per-project classifier
        if model_path not in _models:
            _models[model_path] = BaseNN.load(model_path)
        return _models[model_path]


class MultiLabelLSTM(BaseNN):

    def __init__(
        self,
        input_size,
        output_size,
        fc_size=128,
        hidden_size=16,
        num_layers=1,
        sequence_size=64,
        device=None,
    ):
        super(MultiLabelLSTM, self).__init__()

        # Split the input data into sequences of sequence_size
        self.sequence_size = sequence_size
        self.hidden_size = hidden_size

        # Reduce dimensionality of input data
        self.fc_input = nn.Linear(input_size, fc_size)

        # LSTM layer for handling sequential data
        self.lstm = nn.LSTM(
            fc_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )

        # Fully connected layer for classification at each time step
        self.fc = nn.Linear(
            2 * hidden_size, output_size
        )  # 2 because of bidirectional LSTM

        # Initialize the loss function and optimizer
        self.criterion = (
            nn.BCELoss()
        )  # Binary cross-entropy for multi-label classification
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

        # Initialize device (CPU or GPU)
        self.device = device if device else torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        # Reduce the dimensionality of the input data
        x = torch.relu(self.fc_input(x))

        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (hn, cn) = self.lstm(
            x
        )  # lstm_out contains outputs for all time steps

        # Apply fully connected layer at each time step to get output for each step
        out = torch.sigmoid(
            self.fc(lstm_out)
        )  # Sigmoid activation for multi-label classification

        # Output shape: (batch_size, seq_len, output_size)
        return out

    def preprocess_sequence(self, sequence: List[torch.Tensor], labels=None, overlap=2):
        sequence = torch.stack(sequence) if isinstance(sequence, list) else sequence
        sequence_size = self.sequence_size

        # Split the data into small sequences by sequence_size with overlap
        chunks = [
            sequence[i : i + sequence_size]
            for i in range(0, len(sequence), sequence_size // overlap)
        ]
        chunks = pad_sequence(chunks, batch_first=True, padding_value=0)

        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.float32)
            labels = [
                labels[i : i + sequence_size]
                for i in range(0, len(labels), sequence_size // overlap)
            ]
            labels = (
                pad_sequence(labels, batch_first=True, padding_value=0)
                if labels is not None
                else None
            )

        return chunks, labels

    def evaluate_multilabel_accuracy(self, dataloader, threshold=0.5):
        self.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for data, labels in dataloader:
                outputs = self(data)
                predictions = (outputs >= threshold).float()

                # Calculate number of correct predictions for each label
                correct += (predictions == labels).sum().item()
                total += labels.numel()  # Total number of elements (for multi-label)

        return correct / total

    def partial_fit(
        self, sequence, labels, batch_size=32, epochs=100, accuracy_threshold=0.95
    ):
        """Train the model on the given sequence data.
        Args:
            sequence (List[torch.Tensor]): List of tensors containing the input data
            labels (List[List[int]]): List of lists containing the labels for each time step
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            accuracy_threshold (float): Stop training if accuracy exceeds this threshold
        """
        batches, label_batches = self.preprocess_sequence(sequence, labels)

        # Create a DataLoader for batching the input data
        outputs = None
        dataset = TensorDataset(
            batches, torch.tensor(label_batches, dtype=torch.float32)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()  # Set the model to training mode
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

            accuracy = self.evaluate_multilabel_accuracy(dataloader)
            logger.info(
                f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}, Accuracy: {accuracy}"
            )
            if accuracy >= accuracy_threshold:
                logger.info(
                    f"Accuracy threshold reached: {accuracy} >= {accuracy_threshold}, model training stopped."
                )
                break

        return outputs

    def predict(self, sequence):
        """Split sequence into chunks with sequence_size and predict by chunks.
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
        return logits[0:length]
