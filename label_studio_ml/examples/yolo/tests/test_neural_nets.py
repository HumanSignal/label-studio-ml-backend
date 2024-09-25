import pytest
import torch

from ..utils.neural_nets import MultiLabelLSTM


def test_multi_label_lstm():
    # Model configuration
    input_size = 50  # Number of features per time step
    output_size = 2  # Number of output classes (multi-label classification)
    seq_len = 72  # Sequence length (number of time steps)
    hidden_size = 16  # LSTM hidden state size

    # Initialize device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of the model
    model = MultiLabelLSTM(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        dropout_rate=0.05,
        device=device,
    )

    # Example sequential data: 100 samples, each with 5 time steps and 10 features per time step
    data = torch.randn(seq_len, input_size)  # Shape: (batch_size, seq_len, input_size)
    labels = torch.randint(
        0, 2, (seq_len, output_size)
    ).tolist()  # Shape: (batch_size, seq_len, output_size)

    # Perform partial training with batch size of 16
    model.partial_fit(
        data,
        labels,
        batch_size=16,
        epochs=1000,
        accuracy_threshold=0.999,
        f1_threshold=0.999,
    )

    # Example prediction
    predictions = model.predict(data)
    print(predictions)

    # Save the model
    model.save_and_cache("lstm_model.pth")

    # Load the model
    loaded_model = MultiLabelLSTM.load("lstm_model.pth")

    # Predict with the loaded model
    loaded_predictions = loaded_model.predict(data)
    loaded_labels = (loaded_predictions > 0.5).int()

    assert torch.equal(
        torch.tensor(labels), loaded_labels.int()
    ), "Predicted labels do not match the training labels."
