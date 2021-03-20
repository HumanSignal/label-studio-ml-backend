## Quickstart

Here is a quick example tutorial on how to run the ML backend with a simple text classifier:

0. Clone repo
   ```bash
   git clone https://github.com/heartexlabs/label-studio-ml-backend  
   ```
   
1. Setup environment
   ```bash
   cd label-studio-ml-backend
   pip install -e .
   cd label_studio_ml/examples
   pip install -r requirements.txt
   ```
   
2. Create new ML backend
   ```bash
   label-studio-ml init my_ml_backend --script label_studio_ml/examples/simple_text_classifier.py
   ```
   
3. Start ML backend server
   ```bash
   label-studio-ml start my_ml_backend
   ```
   
4. Run Label Studio connecting it to the running ML backend from the project settings page

## Create your own ML backend

Check examples in `label-studio_ml/examples` directory.
