# Label Studio ML Backend Examples

This directory contains a collection of example machine learning backends that can be integrated with Label Studio. These examples demonstrate how to set up and use various machine learning models for different types of data labeling tasks.

Each subdirectory in this `examples` folder represents a separate ML backend. These examples cover a range of use cases and machine learning frameworks, showing how to integrate them with Label Studio for enhancing the data labeling process.

## Getting Started

To get started with any of the examples, you will need to have Docker installed on your machine as most examples utilize Docker for ease of setup and deployment.

### Prerequisites

- Docker
- Docker Compose
- Label Studio (installation instructions can be found on the [official Label Studio documentation](https://labelstud.io/guide/index.html))

### Common Setup Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/HumanSignal/label-studio-ml-backend.git
   cd label-studio-ml-backend/label_studio_ml/examples
   ```

2. **Navigate to Example Directory:**
   Each example is contained in its own directory. Navigate into the directory of the example you want to try:
   ```bash
   cd <example_name>
   ```

3. **Build and Run the Docker Containers:**
   Most examples include a `docker-compose.yml` file which can be used to build and run the necessary containers:
   ```bash
   docker-compose up
   ```

4. **Connect the ML Backend to Label Studio:**
   Once your ML backend is running, you can connect it to Label Studio by adding it as a new ML backend in the Label Studio UI under the Project Settings => Model.

## Examples

#### bert_classifier
This folder contains an example of a text classification model using BERT (Bidirectional Encoder Representations from Transformers). It is designed to classify text into predefined categories.

#### grounding_dino
Includes an implementation of the Grounding DINO model, which combines DINO (a vision transformer) with grounding techniques for improved object detection and image segmentation.

#### llm_interactive
This example demonstrates the integration of large language models (LLMs) for interactive text generation and processing within Label Studio, enhancing tasks like text completion or modification based on user input.

#### segment_anything_model
Contains the Segment Anything Model (SAM) backend, which uses Meta's SAM for image segmentation tasks. This model can segment various objects in an image with high precision.

#### sklearn_text_classifier
An example of a text classification setup using Scikit-Learn. This is suitable for simple classification tasks with textual data.

#### spacy
This folder provides an example of integrating the spaCy library for named entity recognition (NER) tasks, showcasing how to use spaCy models within Label Studio.

#### tensorflow
Contains examples of using TensorFlow models within Label Studio, demonstrating how to set up and use TensorFlow for various machine learning tasks.

#### tesseract
An example of integrating the Tesseract OCR engine for optical character recognition tasks. This setup helps in converting images of text into machine-encoded text.

#### flair
This backend uses the Flair framework for advanced natural language processing, particularly effective for sequence labeling tasks like NER.

#### easyocr
Provides an example of integrating EasyOCR for OCR tasks, which is useful for extracting text from images.

#### huggingface_llm
Demonstrates how to integrate Hugging Face's transformers for tasks that can benefit from large language models, such as text generation or translation.

#### huggingface_ner
An example setup for named entity recognition using models from Hugging Face's transformers library.

#### interactive_substring_matching
This backend is designed for substring matching tasks, where the model interacts with the text to identify and match substrings based on patterns or queries.

#### langchain_search_agent
Integrates Langchain for advanced search and retrieval tasks, combining language understanding with search capabilities to improve information retrieval within Label Studio.

#### mmdetection-3
An example using the MMDetection library for object detection tasks, which can detect various objects within images using state-of-the-art detection models.

#### nemo_asr
Contains a setup for using NVIDIA's NeMo toolkit for automatic speech recognition (ASR), converting speech audio into text.

#### rerankers
This folder includes examples of using reranking models that can adjust the order of given items based on certain criteria, useful in recommendation systems or sorting tasks.

#### dummy_model
A simple placeholder model used for demonstration purposes, showing the basic setup of an ML backend without any specific machine learning functionality.


## Contributing

Contributions to this repository are welcome. You can contribute by improving the existing examples or adding new examples that demonstrate other use cases or machine learning frameworks.
