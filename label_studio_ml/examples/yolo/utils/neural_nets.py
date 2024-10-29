import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging

from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import (
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score,
    MultilabelAccuracy,
)
from typing import List, Union
from joblib import Memory


logger = logging.getLogger(__name__)
memory = Memory("./cache_dir", verbose=0)  # Set up disk-based caching for model results
_models = {}


@memory.cache(ignore=["yolo_model"])
def cached_yolo_predict(yolo_model, video_path, cache_params):
    """Predict bounding boxes and labels using YOLO model and cache the results using joblib.
    Args:
        yolo_model (YOLO): YOLO model instance
        video_path (str): Path to the video file
        cache_params (str): Parameters for caching the results, they are used in @memory.cache decorator
    """
    frames = []
    generator = yolo_model.predict(video_path, stream=True)

    for frame in generator:
        frame.orig_img = None  # remove image from cache to reduce size
        frames.append(frame)

    return frames


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

    # Run model prediction, use stream to avoid out of memory
    generator = yolo_model.predict(video_path, stream=True)

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
    def __init__(self, **kwargs):
        super(BaseNN, self).__init__()
        self.label_map = None

    def set_label_map(self, label_map):
        self.label_map = label_map

    def get_label_map(self):
        return self.label_map

    def save(self, path):
        # ultralytics yolo11 patches torch.save to use dill,
        # however it leads to serialization errors,
        # so let's check for use_dill and disable it
        if 'use_dill' in torch.save.__code__.co_varnames:
            torch.save(self, path, use_dill=False)
        else:
            torch.save(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path) -> "BaseNN":
        model = torch.load(path)
        model.eval()  # Set the model to evaluation mode
        logger.info(f"Model loaded from {path}")
        return model

    @classmethod
    def load_cached_model(cls, model_path: str) -> Union["BaseNN", None]:
        global _models
        if not os.path.exists(model_path):
            return None

        # Load per-project classifier
        if model_path not in _models:
            _models[model_path] = BaseNN.load(model_path)
        return _models[model_path]

    def save_and_cache(self, path):
        self.save(path)
        _models[path] = self


class MultiLabelLSTM(BaseNN):

    def __init__(
        self,
        input_size,
        output_size,
        fc_size=128,
        hidden_size=16,
        num_layers=1,
        sequence_size=16,
        learning_rate=1e-4,
        weight_decay=1e-5,
        dropout_rate=0.2,
        device=None,
        **kwargs,
    ):
        """Initialize the MultiLabelLSTM model.
        Args:
            input_size (int): Number of features in the input data
            output_size (int): Number of labels in the output data
            fc_size (int): Size of the fully connected layer
            hidden_size (int): Size of the hidden state in the LSTM
            num_layers (int): Number of layers in the LSTM
            sequence_size (int): Size of the input sequence, used for chunking
            learning_rate (float): Learning rate for the optimizer
            weight_decay (float): Weight decay for the optimizer for L2 regularization
            dropout_rate (float): Dropout rate for the fully connected layer
            device (torch.device): Device to run the model on (CPU or GPU)
        """
        super(MultiLabelLSTM, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.fc_size = fc_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_size = sequence_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

        # Reduce dimensionality of input data
        self.fc_input = nn.Linear(input_size, fc_size)
        self.layer_norm = nn.LayerNorm(fc_size)
        self.dropout = nn.Dropout(self.dropout_rate)

        # LSTM layer for handling sequential data
        self.lstm = nn.LSTM(
            fc_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )

        # Fully connected layer for classification at each time step
        # 2 because of bidirectional LSTM
        self.fc = nn.Linear(2 * hidden_size, output_size)

        # Initialize the loss function and optimizer
        self.criterion = (
            nn.BCEWithLogitsLoss()
        )  # Binary cross-entropy for multi-label classification
        self.optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Initialize device (CPU or GPU)
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device if device else target_device
        self.to(self.device)

    def forward(self, x):
        # Reduce the dimensionality of the input data
        x = torch.relu(self.fc_input(x))
        x = self.layer_norm(x)
        x = self.dropout(x)

        # x shape: (batch_size, seq_len, input_size)
        # lstm_out contains outputs for all time steps
        lstm_out, (_, _) = self.lstm(x)

        # Apply fully connected layer at each time step to get output with final label number
        out = self.fc(lstm_out)

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
            labels = pad_sequence(labels, batch_first=True, padding_value=0)

        return chunks, labels

    def evaluate_metrics(self, dataloader, threshold=0.5):
        self.eval()
        params = {
            "num_labels": self.output_size,
            "average": "macro",
            "threshold": threshold,
            "zero_division": 1,
        }
        precision_metric = MultilabelPrecision(**params).to(self.device)
        recall_metric = MultilabelRecall(**params).to(self.device)
        f1_metric = MultilabelF1Score(**params).to(self.device)
        accuracy_metric = MultilabelAccuracy(**params).to(self.device)

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                outputs = torch.sigmoid(self(data))

                # Reshape outputs and labels from (batch_size, seq_len, num_labels)
                # to (batch_size * seq_len, num_labels)
                outputs = outputs.view(-1, self.output_size)
                labels = labels.view(-1, self.output_size)

                # No need to threshold manually; the metrics handle it
                precision_metric.update(outputs, labels)
                recall_metric.update(outputs, labels)
                f1_metric.update(outputs, labels)
                accuracy_metric.update(outputs, labels)

        return {
            "precision": precision_metric.compute().item(),
            "recall": recall_metric.compute().item(),
            "f1_score": f1_metric.compute().item(),
            "accuracy": accuracy_metric.compute().item(),
        }

    def partial_fit(
        self,
        sequence,
        labels,
        batch_size=32,
        epochs=1000,
        accuracy_threshold=1.0,
        f1_threshold=1.0,
    ):
        """Train the model on the given sequence data.
        Args:
            sequence (List[torch.Tensor]): List of tensors containing the input data
            labels (List[List[int]]): List of lists containing the labels for each time step
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            accuracy_threshold (float): Stop training if accuracy exceeds this threshold
            f1_threshold (float): Stop training if F1 score exceeds this threshold
        """
        batches, label_batches = self.preprocess_sequence(sequence, labels)

        # Create a DataLoader for batching the input data
        metrics = {}
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

            # metrics and threshold stops to avoid overfitting
            metrics = self.evaluate_metrics(dataloader)
            metrics["loss"] = epoch_loss / len(dataloader)
            metrics["epoch"] = epoch + 1

            logger.info(
                f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}, {metrics}"
            )
            if metrics["accuracy"] >= accuracy_threshold:
                logger.info(
                    f"Accuracy >= {accuracy_threshold} threshold, model training stopped."
                )
                break
            if metrics["f1_score"] >= f1_threshold:
                logger.info(
                    f"F1 score >= {f1_threshold} threshold, model training stopped."
                )
                break

        return metrics

    def predict(self, sequence):
        """Split sequence into chunks with sequence_size and predict by chunks.
        Then concatenate all predictions into one sequence of labels
        """
        length = len(sequence)
        if length == 0:
            return torch.tensor([])

        batches, _ = self.preprocess_sequence(sequence, overlap=1)
        self.eval()
        logits = torch.sigmoid(self(batches))

        # Concatenate batches to sequence back
        shape = logits.shape
        logits = torch.reshape(logits, [shape[0] * shape[1], shape[2]])
        return logits[0:length]
