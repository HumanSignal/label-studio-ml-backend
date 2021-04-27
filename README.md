## Quickstart

Here is a quick example tutorial on how to run the ML backend with a simple text classifier:

0. Clone repo
   ```bash
   git clone https://github.com/heartexlabs/label-studio-ml-backend  
   ```
   
1. Setup environment
    
    It is highly recommended to use `venv`, `virtualenv` or `conda` python environments. You can use the same environment as Label Studio does. [Read more](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) about creating virtual environments via `venv`.
   ```bash
   cd label-studio-ml-backend
   
   # Install label-studio-ml and its dependencies
   pip install -U -e .
   
   # Install example dependencies
   pip install -r label_studio_ml/examples/requirements.txt
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

Check examples in `label_studio_ml/examples` directory.
