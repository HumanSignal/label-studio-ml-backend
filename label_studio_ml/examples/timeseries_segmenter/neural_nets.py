import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassAccuracy,
)
from typing import List, Union, Tuple


logger = logging.getLogger(__name__)
_models = {}


class BaseNN(nn.Module):
    def __init__(self, **kwargs):
        super(BaseNN, self).__init__()
        self.label_map = None

    def set_label_map(self, label_map):
        self.label_map = label_map

    def get_label_map(self):
        return self.label_map

    def save(self, path):
        """Save model state dict and parameters safely."""
        # Get state dict and remove criterion parameters (they're not part of the model)
        state_dict = self.state_dict()
        # Remove any criterion-related parameters that shouldn't be saved
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('criterion.')}
        
        # Create save dictionary with model state and parameters
        save_dict = {
            'model_state_dict': filtered_state_dict,
            'model_params': {
                'input_size': getattr(self, 'input_size', None),
                'output_size': getattr(self, 'output_size', None),
                'fc_size': getattr(self, 'fc_size', 128),
                'hidden_size': getattr(self, 'hidden_size', 32),
                'num_layers': getattr(self, 'num_layers', 2),
                'sequence_size': getattr(self, 'sequence_size', 20),
                'learning_rate': getattr(self, 'learning_rate', 1e-3),
                'weight_decay': getattr(self, 'weight_decay', 1e-5),
                'dropout_rate': getattr(self, 'dropout_rate', 0.3),
            },
            'label_map': self.label_map,
            'model_class': self.__class__.__name__
        }
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path) -> "BaseNN":
        """Load model from state dict safely."""
        # Load the save dictionary
        save_dict = torch.load(path, map_location='cpu', weights_only=True)
        
        # Get model class and parameters
        model_class_name = save_dict.get('model_class', 'TimeSeriesLSTM')
        model_params = save_dict['model_params']
        
        # Create the appropriate model instance
        if model_class_name in ['TimeSeriesLSTM', 'MultiLabelLSTM']:
            # Import here to avoid circular imports
            model = TimeSeriesLSTM(**model_params)
        else:
            raise ValueError(f"Unknown model class: {model_class_name}")
        
        # Load the state dict with error handling for unexpected keys
        try:
            model.load_state_dict(save_dict['model_state_dict'], strict=False)
        except Exception as e:
            # Filter out problematic keys if needed
            state_dict = save_dict['model_state_dict']
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('criterion.')}
            model.load_state_dict(filtered_state_dict, strict=False)
            logger.warning(f"Filtered out problematic keys during model loading: {e}")
        
        # Set label map
        if 'label_map' in save_dict:
            model.set_label_map(save_dict['label_map'])
        
        model.eval()  # Set to evaluation mode
        logger.info(f"Model loaded from {path}")
        return model

    @classmethod
    def load_cached_model(cls, model_path: str) -> Union["BaseNN", None]:
        global _models
        if not os.path.exists(model_path):
            return None

        # Load per-project classifier
        if model_path not in _models:
            try:
                _models[model_path] = cls.load(model_path)
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                return None
        return _models[model_path]

    def save_and_cache(self, path):
        self.save(path)
        _models[path] = self


class TimeSeriesLSTM(BaseNN):
    """LSTM-based time series segmenter optimized for sequential sensor data."""

    def __init__(
        self,
        input_size,
        output_size,
        fc_size=128,
        hidden_size=32,
        num_layers=2,
        sequence_size=20,
        learning_rate=1e-3,
        weight_decay=1e-5,
        dropout_rate=0.3,
        device=None,
        **kwargs,
    ):
        """Initialize the TimeSeriesLSTM model.
        Args:
            input_size (int): Number of features in the input data (sensor channels)
            output_size (int): Number of classes in the output data (including background)
            fc_size (int): Size of the fully connected layer
            hidden_size (int): Size of the hidden state in the LSTM
            num_layers (int): Number of layers in the LSTM
            sequence_size (int): Size of the input sequence window
            learning_rate (float): Learning rate for the optimizer
            weight_decay (float): Weight decay for the optimizer for L2 regularization
            dropout_rate (float): Dropout rate for regularization
            device (torch.device): Device to run the model on (CPU or GPU)
        """
        super(TimeSeriesLSTM, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.fc_size = fc_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_size = sequence_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

        # Input preprocessing layers
        self.fc_input = nn.Linear(input_size, fc_size)
        self.input_norm = nn.LayerNorm(fc_size)
        self.input_dropout = nn.Dropout(self.dropout_rate)

        # LSTM layer for temporal modeling
        self.lstm = nn.LSTM(
            fc_size, hidden_size, num_layers, 
            batch_first=True, bidirectional=True, dropout=dropout_rate
        )

        # Output layers
        self.output_dropout = nn.Dropout(self.dropout_rate)
        # 2 * hidden_size because of bidirectional LSTM
        self.fc_output = nn.Linear(2 * hidden_size, output_size)

        # Loss function for multi-class classification (will be set with class weights in partial_fit)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding
        self.optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Initialize device (CPU or GPU)
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device if device else target_device
        self.to(self.device)
        
        logger.info(f"TimeSeriesLSTM initialized on device: {self.device}")

    def forward(self, x):
        """Forward pass through the network.
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        # Input preprocessing
        x = torch.relu(self.fc_input(x))
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # LSTM processing
        lstm_out, (_, _) = self.lstm(x)
        
        # Output processing
        lstm_out = self.output_dropout(lstm_out)
        out = self.fc_output(lstm_out)

        return out

    def preprocess_sequence(
        self, 
        sequence: Union[List[torch.Tensor], torch.Tensor, np.ndarray], 
        labels=None, 
        overlap_ratio=0.5,
        padding_mode='reflect'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create overlapping windows from sequence data with proper padding.
        Args:
            sequence: Input sequence data
            labels: Optional labels for training
            overlap_ratio: Overlap ratio between windows (0.5 = 50% overlap)
            padding_mode: Padding strategy ('reflect', 'edge', 'constant')
        Returns:
            chunks: Windowed sequences
            label_chunks: Windowed labels (if provided)
        """
        # Convert to tensor if needed
        if isinstance(sequence, (list, np.ndarray)):
            sequence = torch.tensor(sequence, dtype=torch.float32)
        
        seq_len = len(sequence)
        sequence_size = self.sequence_size
        step_size = max(1, int(sequence_size * (1 - overlap_ratio)))
        
        # Handle short sequences
        if seq_len < sequence_size:
            # Pad short sequences using reflection
            if padding_mode == 'reflect' and seq_len > 1:
                pad_size = sequence_size - seq_len
                # Reflect padding
                padded = torch.cat([
                    sequence.flip(0)[:pad_size//2 + pad_size%2],
                    sequence,
                    sequence.flip(0)[:pad_size//2]
                ])
                chunks = padded.unsqueeze(0)
            elif padding_mode == 'edge':
                # Edge padding
                first_val = sequence[0:1].repeat(sequence_size//2, 1)
                last_val = sequence[-1:].repeat(sequence_size//2, 1)
                padded = torch.cat([first_val, sequence, last_val])[:sequence_size]
                chunks = padded.unsqueeze(0)
            else:
                # Zero padding (fallback)
                padded = torch.zeros(sequence_size, sequence.shape[-1])
                start_idx = (sequence_size - seq_len) // 2
                padded[start_idx:start_idx + seq_len] = sequence
                chunks = padded.unsqueeze(0)
        else:
            # Create overlapping windows
            indices = list(range(0, seq_len - sequence_size + 1, step_size))
            # Ensure we cover the end of the sequence
            if indices[-1] + sequence_size < seq_len:
                indices.append(seq_len - sequence_size)
            
            chunks = torch.stack([sequence[i:i + sequence_size] for i in indices])

        # Handle labels if provided
        label_chunks = None
        if labels is not None:
            if isinstance(labels, (list, np.ndarray)):
                labels = torch.tensor(labels, dtype=torch.long)
                
            if seq_len < sequence_size:
                # For short sequences, use center label or pad
                if seq_len == 1:
                    padded_labels = torch.full((sequence_size,), labels[0].item(), dtype=torch.long)
                else:
                    padded_labels = torch.full((sequence_size,), -1, dtype=torch.long)  # Padding token
                    start_idx = (sequence_size - seq_len) // 2
                    padded_labels[start_idx:start_idx + seq_len] = labels
                label_chunks = padded_labels.unsqueeze(0)
            else:
                indices = list(range(0, seq_len - sequence_size + 1, step_size))
                if indices[-1] + sequence_size < seq_len:
                    indices.append(seq_len - sequence_size)
                    
                label_chunks = torch.stack([labels[i:i + sequence_size] for i in indices])

        return chunks, label_chunks

    def predict_with_overlap_averaging(self, sequence: Union[List, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Predict with overlapping windows and average predictions for overlapping regions.
        Args:
            sequence: Input sequence data
        Returns:
            Predictions for each timestep
        """
        if len(sequence) == 0:
            return torch.tensor([])

        # Convert to tensor
        if isinstance(sequence, (list, np.ndarray)):
            sequence = torch.tensor(sequence, dtype=torch.float32)

        sequence = sequence.to(self.device)
        self.eval()
        
        with torch.no_grad():
            # Get windowed data with overlap
            chunks, _ = self.preprocess_sequence(sequence, overlap_ratio=0.5)
            chunks = chunks.to(self.device)
            
            # Predict on all chunks
            chunk_predictions = self(chunks)  # Shape: (n_chunks, seq_size, n_classes)
            chunk_predictions = torch.softmax(chunk_predictions, dim=-1)
            
            # Average overlapping predictions
            seq_len = len(sequence)
            sequence_size = self.sequence_size
            step_size = max(1, int(sequence_size * 0.5))  # 50% overlap
            
            # Initialize prediction accumulator
            predictions = torch.zeros(seq_len, self.output_size, device=self.device)
            counts = torch.zeros(seq_len, device=self.device)
            
            # Handle short sequences
            if seq_len < sequence_size:
                pred = chunk_predictions[0]  # Single chunk
                start_idx = (sequence_size - seq_len) // 2
                predictions = pred[start_idx:start_idx + seq_len]
            else:
                # Accumulate predictions from overlapping windows
                indices = list(range(0, seq_len - sequence_size + 1, step_size))
                if indices[-1] + sequence_size < seq_len:
                    indices.append(seq_len - sequence_size)
                
                for i, start_idx in enumerate(indices):
                    end_idx = start_idx + sequence_size
                    predictions[start_idx:end_idx] += chunk_predictions[i]
                    counts[start_idx:end_idx] += 1
                
                # Average the predictions
                predictions = predictions / counts.unsqueeze(-1)
        
        return predictions

    def evaluate_metrics(self, dataloader, threshold=0.5):
        """Evaluate model performance using multiclass metrics optimized for imbalanced data."""
        self.eval()
        
        # Macro-averaged metrics (better for imbalanced data)
        precision_macro = MulticlassPrecision(
            num_classes=self.output_size, average='macro', ignore_index=-1
        ).to(self.device)
        recall_macro = MulticlassRecall(
            num_classes=self.output_size, average='macro', ignore_index=-1
        ).to(self.device)
        f1_macro = MulticlassF1Score(
            num_classes=self.output_size, average='macro', ignore_index=-1
        ).to(self.device)
        
        # Per-class F1 scores for detailed analysis
        f1_per_class = MulticlassF1Score(
            num_classes=self.output_size, average=None, ignore_index=-1
        ).to(self.device)
        
        # Balanced accuracy (macro-averaged recall)
        balanced_accuracy = MulticlassRecall(
            num_classes=self.output_size, average='macro', ignore_index=-1
        ).to(self.device)

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                outputs = self(data)

                # Reshape for metrics calculation
                outputs_flat = outputs.view(-1, self.output_size)
                labels_flat = labels.view(-1)
                
                # Get predictions
                predictions = torch.argmax(outputs_flat, dim=1)

                # Update metrics (ignoring padding tokens)
                valid_mask = labels_flat != -1
                if valid_mask.any():
                    valid_preds = predictions[valid_mask]
                    valid_labels = labels_flat[valid_mask]
                    
                    precision_macro.update(valid_preds, valid_labels)
                    recall_macro.update(valid_preds, valid_labels)
                    f1_macro.update(valid_preds, valid_labels)
                    f1_per_class.update(valid_preds, valid_labels)
                    balanced_accuracy.update(valid_preds, valid_labels)

        # Compute per-class F1 scores
        per_class_f1 = f1_per_class.compute()
        
        # Handle NaN values (can occur if a class has no samples)
        per_class_f1 = torch.nan_to_num(per_class_f1, nan=0.0)
        
        # Find minimum F1 score across classes (excluding classes with 0 samples)
        non_zero_f1 = per_class_f1[per_class_f1 > 0]
        min_class_f1 = non_zero_f1.min().item() if len(non_zero_f1) > 0 else 0.0

        return {
            "precision": precision_macro.compute().item(),
            "recall": recall_macro.compute().item(),
            "f1_score": f1_macro.compute().item(),
            "balanced_accuracy": balanced_accuracy.compute().item(),
            "per_class_f1": per_class_f1.cpu().numpy().tolist(),
            "min_class_f1": min_class_f1,
        }

    def partial_fit(
        self,
        sequence,
        labels,
        batch_size=16,
        epochs=100,
        balanced_accuracy_threshold=0.85,
        min_class_f1_threshold=0.70,
        use_class_weights=True,
    ):
        """Train the model on the given sequence data with imbalanced data handling.
        Args:
            sequence: Input sequence data
            labels: Target labels for each timestep
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            balanced_accuracy_threshold: Stop training if balanced accuracy exceeds this threshold
            min_class_f1_threshold: Stop training if minimum per-class F1 exceeds this threshold
            use_class_weights: Whether to use class weights to handle imbalanced data
        """
        if len(sequence) == 0:
            logger.warning("Empty sequence provided for training")
            return {}

        batches, label_batches = self.preprocess_sequence(sequence, labels, overlap_ratio=0.5)
        
        if batches is None or len(batches) == 0:
            logger.warning("No valid batches created from sequence")
            return {}

        # Calculate class weights for imbalanced data handling
        if use_class_weights:
            flat_labels = label_batches.view(-1)
            valid_labels = flat_labels[flat_labels != -1]  # Remove padding tokens
            
            # Calculate class frequencies
            unique_classes, class_counts = torch.unique(valid_labels, return_counts=True)
            total_samples = len(valid_labels)
            
            # Calculate inverse frequency weights
            class_weights = torch.zeros(self.output_size, dtype=torch.float32, device=self.device)
            for i, class_idx in enumerate(unique_classes):
                class_weights[class_idx] = total_samples / (len(unique_classes) * class_counts[i])
            
            # Update loss function with class weights
            self.criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)
            
            # Log class distribution and weights
            class_info = {}
            for i, class_idx in enumerate(unique_classes):
                class_info[f"class_{class_idx.item()}"] = {
                    "count": class_counts[i].item(),
                    "percentage": (class_counts[i].item() / total_samples * 100),
                    "weight": class_weights[class_idx].item()
                }
            logger.info(f"Class distribution and weights: {class_info}")

        # Create DataLoader
        dataset = TensorDataset(batches, label_batches)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        logger.info(f"Training on {len(batches)} windows, {len(dataset)} total samples")

        metrics = {}
        best_min_f1 = 0.0
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self(batch_data)
                
                # Reshape for loss calculation
                outputs_flat = outputs.view(-1, self.output_size)
                labels_flat = batch_labels.view(-1)
                
                loss = self.criterion(outputs_flat, labels_flat)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()

            # Evaluate metrics
            metrics = self.evaluate_metrics(dataloader)
            metrics["loss"] = epoch_loss / len(dataloader)
            metrics["epoch"] = epoch + 1
            
            # Track best minimum F1 for early stopping
            if metrics["min_class_f1"] > best_min_f1:
                best_min_f1 = metrics["min_class_f1"]

            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch + 1}/{epochs}: Loss={metrics['loss']:.4f}, "
                           f"Balanced Acc={metrics['balanced_accuracy']:.3f}, "
                           f"Macro F1={metrics['f1_score']:.3f}, "
                           f"Min Class F1={metrics['min_class_f1']:.3f}")
                           
                # Log per-class F1 scores for detailed monitoring
                if len(metrics["per_class_f1"]) <= 5:  # Only log if reasonable number of classes
                    per_class_str = ", ".join([f"F1_{i}={f1:.3f}" for i, f1 in enumerate(metrics["per_class_f1"])])
                    logger.info(f"Per-class F1 scores: {per_class_str}")

            # Improved early stopping criteria for imbalanced data
            stop_training = False
            
            # Stop if balanced accuracy is high AND minimum class F1 is acceptable
            if (metrics["balanced_accuracy"] >= balanced_accuracy_threshold and 
                metrics["min_class_f1"] >= min_class_f1_threshold):
                logger.info(f"Both balanced accuracy ({metrics['balanced_accuracy']:.3f} >= {balanced_accuracy_threshold}) "
                           f"and minimum class F1 ({metrics['min_class_f1']:.3f} >= {min_class_f1_threshold}) "
                           f"thresholds reached, stopping training.")
                stop_training = True
            
            # Alternative: Stop if we haven't improved minimum class F1 for many epochs (patience mechanism)
            elif epoch > 50 and metrics["min_class_f1"] < 0.1:  # If very poor performance on minority classes
                logger.warning(f"Minimum class F1 ({metrics['min_class_f1']:.3f}) is very low after {epoch} epochs. "
                              f"Consider adjusting class weights or thresholds.")
            
            if stop_training:
                break

        return metrics

    def predict(self, sequence):
        """Predict labels for input sequence using overlap averaging."""
        return self.predict_with_overlap_averaging(sequence)

    @classmethod
    def load_model(cls, path) -> "TimeSeriesLSTM":
        """Load TimeSeriesLSTM model from state dict safely."""
        try:
            # Load the save dictionary
            save_dict = torch.load(path, map_location='cpu', weights_only=True)
            
            # Get model parameters
            model_params = save_dict['model_params']
            
            # Create model instance
            model = cls(**model_params)
            
            # Load the state dict with error handling for unexpected keys
            try:
                model.load_state_dict(save_dict['model_state_dict'], strict=False)
            except Exception as e:
                # Filter out problematic keys if needed
                state_dict = save_dict['model_state_dict']
                filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('criterion.')}
                model.load_state_dict(filtered_state_dict, strict=False)
                logger.warning(f"Filtered out problematic keys during TimeSeriesLSTM loading: {e}")
            
            # Set label map
            if 'label_map' in save_dict:
                model.set_label_map(save_dict['label_map'])
            
            model.eval()  # Set to evaluation mode
            logger.info(f"TimeSeriesLSTM model loaded from {path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load TimeSeriesLSTM model from {path}: {e}")
            raise
