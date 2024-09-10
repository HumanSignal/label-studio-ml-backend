import torch
import torch.nn as nn
import torch.optim as optim
import logging


logger = logging.getLogger(__name__)


class MultiLabelNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiLabelNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_size)

        # Initialize the loss function and optimizer
        self.criterion = nn.BCELoss()  # Binary cross-entropy for multi-label
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))  # Sigmoid for multi-label classification
        return x

    def partial_fit(self, new_data, new_labels, epochs=1):
        self.train()  # Set the model to training mode
        new_data = torch.stack(new_data)
        new_labels = torch.tensor(new_labels, dtype=torch.float32)
        outputs = None

        for epoch in range(epochs):
            self.optimizer.zero_grad()  # Zero out gradients from previous steps
            outputs = self(new_data)
            loss = self.criterion(outputs, new_labels)  # Calculate the loss
            loss.backward()  # Backpropagation
            self.optimizer.step()  # Update model parameters
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        return outputs

    def predict(model, new_data, threshold=None):
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            new_data = torch.tensor(new_data, dtype=torch.float32)
            outputs = model(new_data)
            if threshold is None:
                return outputs

        return (outputs > threshold).int()  # Apply threshold to get binary labels

    def save(self, path):
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()  # Set the model to evaluation mode
        logger.info(f"Model loaded from {path}")