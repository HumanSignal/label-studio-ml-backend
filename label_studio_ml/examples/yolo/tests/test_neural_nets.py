import pytest

from ..utils.neural_nets import MultiLabelNN
from label_studio_ml.utils import compare_nested_structures
import torch


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


