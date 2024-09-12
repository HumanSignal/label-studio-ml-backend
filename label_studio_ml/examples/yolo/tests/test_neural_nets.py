import pytest
import torch

from ..utils.neural_nets import MultiLabelNN, MultiLabelLSTM
from label_studio_ml.utils import compare_nested_structures


def test_multi_label_nn():
    # Create the model
    input_size = 10  # Replace this with the number of features (e.g., len(frame_results[i].probs))
    output_size = 2  # Number of labels (classes)
    model = MultiLabelNN(input_size, output_size)

    # Replace with actual frame_results[i].probs and labels_array
    new_data = [torch.tensor([0.9, 0.2, 0.1, 0.3, 0.7, 0.8, 0.5, 0.2, 0.4, 0.6])]  # Example frame_results[i].probs
    new_labels = [[1, 0]]  # Corresponding labels (from labels_array)

    # Incrementally train the model on this single new sample
    outputs = model.partial_fit(new_data, new_labels, epochs=10)
    expected_outputs = [[0.5, 0.5]]
    compare_nested_structures(outputs.tolist(), expected_outputs, abs=0.3)

    predictions = model.predict(new_data, threshold=0.5)
    compare_nested_structures(predictions.tolist(), [[1, 0]], abs=0.1)

    model.save("test_model.pt")
    new_model = MultiLabelNN.load("test_model.pt")
    print(new_model)


def test_multi_label_lstm():
    # Model configuration
    input_size = 50  # Number of features per time step
    output_size = 2  # Number of output classes (multi-label classification)
    seq_len = 72  # Sequence length (number of time steps)
    hidden_size = 8  # LSTM hidden state size

    # Initialize device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create an instance of the model
    model = MultiLabelLSTM(input_size=input_size, output_size=output_size, hidden_size=hidden_size, device=device)

    # Example sequential data: 100 samples, each with 5 time steps and 10 features per time step
    data = torch.randn(seq_len, input_size)  # Shape: (batch_size, seq_len, input_size)
    labels = torch.randint(0, 2, (seq_len, output_size)).tolist() # Shape: (batch_size, seq_len, output_size)

    # Perform partial training with batch size of 16
    model.partial_fit(data, labels, batch_size=16, epochs=500)

    # Example prediction
    predictions = model.predict(data)
    print(predictions)

    # Save the model
    model.save("lstm_model.pth")

    # Load the model
    loaded_model = MultiLabelLSTM.load("lstm_model.pth")

    # Predict with the loaded model
    loaded_predictions = loaded_model.predict(data)
    loaded_labels = (loaded_predictions > 0.5).int()

    assert torch.equal(torch.tensor(labels), loaded_labels.int()), "Predicted labels do not match the training labels."
