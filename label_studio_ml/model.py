import copy
import os
import logging
import sys
import json
import importlib
import importlib.util
import inspect

try:
    import torch.multiprocessing as mp
    try:
        # avoid "cannot reinit CUDA in forked process" error in loading cuda?
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
except ImportError:
    import multiprocessing as mp

from semver import Version

from typing import Tuple, Callable, Union, List, Dict, Optional
from abc import ABC
from colorama import Fore

from label_studio_sdk.label_interface import LabelInterface
from label_studio_sdk._extensions.label_studio_tools.core.label_config import parse_config
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from .response import ModelResponse
from .utils import is_preload_needed
from .cache import create_cache

logger = logging.getLogger(__name__)

CACHE = create_cache(
    os.getenv('CACHE_TYPE', 'sqlite'),
    path=os.getenv('MODEL_DIR', '.'))


# Decorator to register predict function
_predict_fn: Callable = None
_update_fn: Callable = None


def predict_fn(f):
    global _predict_fn
    _predict_fn = f
    logger.info(f'{Fore.GREEN}Predict function "{_predict_fn.__name__}" registered{Fore.RESET}')
    return f


def update_fn(f):
    global _update_fn
    _update_fn = f
    logger.info(f'{Fore.GREEN}Update function "{_update_fn.__name__}" registered{Fore.RESET}')
    return f


class LabelStudioMLBase(ABC):
    """
    This is the base class for all LabelStudio Machine Learning models.
    It provides the structure and functions necessary for the machine learning models.
    """
    INITIAL_MODEL_VERSION = "0.0.1"
    
    TRAIN_EVENTS = (
        'ANNOTATION_CREATED',
        'ANNOTATION_UPDATED',
        'ANNOTATION_DELETED',
        'START_TRAINING'
    )

    def __init__(self, project_id: Optional[str] = None, label_config=None):
        """
        Initialize LabelStudioMLBase with a project ID.

        Args:
            project_id (str, optional): The project ID. Defaults to None.
        """
        self.project_id = project_id or ''
        if label_config is not None:
            self.use_label_config(label_config)
        else:
            logger.warning('Label config is not provided')

        # set initial model version
        if not self.model_version:
            self.set("model_version", self.INITIAL_MODEL_VERSION)
        
        self.setup()
        
    def setup(self):
        """Abstract method for setting up the machine learning model.
        This method should be overridden by subclasses of
        LabelStudioMLBase to conduct any necessary setup steps, for
        example to set model_version
        """
        
        # self.set("model_version", "0.0.2")
        
        
    def use_label_config(self, label_config: str):
        """
        Apply label configuration and set the model version and parsed label config.

        Args:
            label_config (str): The label configuration.
        """
        self.label_interface = LabelInterface(config=label_config)
        
        # if not current_label_config:
            # first time model is initialized
            # self.set('model_version', 'INITIAL')                            

        current_label_config = self.get('label_config')    
        # label config has been changed, need to save
        if current_label_config != label_config:
            self.set('label_config', label_config)
            self.set('parsed_label_config', json.dumps(parse_config(label_config)))        
            

    def set_extra_params(self, extra_params):
        """Set extra parameters. Extra params could be used to pass
        any additional static metadata from Label Studio side to ML
        Backend.
        
        Args:
            extra_params: Extra parameters to set.

        """
        self.set('extra_params', extra_params)

    @property
    def extra_params(self):
        """
        Get the extra parameters.

        Returns:
            json: If parameters exist, returns parameters in JSON format. Else, returns None.
        """
        # TODO this needs to have exception
        params = self.get('extra_params')
        if params:
            return json.loads(params)
        else:
            return {}
            
    def get(self, key: str):
        return CACHE[self.project_id, key]

    def set(self, key: str, value: str):
        CACHE[self.project_id, key] = value

    def has(self, key: str):
        return (self.project_id, key) in CACHE

    @property
    def label_config(self):
        return self.get('label_config')

    @property
    def parsed_label_config(self):        
        return json.loads(self.get('parsed_label_config'))

    @property
    def model_version(self):
        mv = self.get('model_version')
        if mv:
            try:
                sv = Version.parse(mv)
                return sv
            except:
                return mv
        else:
            return None

    def bump_model_version(self):
        """
        """
        mv = self.model_version

        # TODO: check if this is correct - seems like it doesn't work, check RND-7 and make sure it's test covered
        mv.bump_minor()
        logger.debug(f'Bumping model version from {self.model_version} to {mv}')
        self.set('model_version', str(mv))
        
        return mv
        
    # @abstractmethod
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> Union[List[Dict], ModelResponse]:
        """
        Predict and return a list of dicts with predictions for each task.

        Args:
            tasks (list[dict]): A list of tasks.
            context (dict, optional): A dictionary with additional context. Defaults to None.
            kwargs: Additional parameters passed on to the predict function.

        Returns:
            list[dict]: A list of dictionaries containing predictions.                
        """

        # if there is a registered predict function, use it
        if _predict_fn:
            return _predict_fn(tasks, context, helper=self, **kwargs)

    def process_event(self, event, data, job_id, additional_params):
        """
        Process a given event. If event is of TRAIN type, start fitting the model.

        Args:
          event: Current event to process.
          data: The data relevant to the event.
          job_id: ID of the job related to the event.
          additional_params: Additional parameters to be processed.
        """
        if event in self.TRAIN_EVENTS:
            logger.debug(f'Job {job_id}: Received event={event}: calling {self.__class__.__name__}.fit()')
            train_output = self.fit(event=event, data=data, job_id=job_id, **additional_params)
            logger.debug(f'Job {job_id}: Train finished.')
            return train_output

    def fit(self, event, data, **additional_params):
        """
        Fit/update the model based on the specified event and data.

        Args:
          event: The event for which the model is fitted.
          data: The data on which the model is fitted.
          additional_params: Additional parameters (params after ** are optional named parameters)
        """
        # if there is a registered update function, use it
        if _update_fn:
            return _update_fn(event, data, helper=self, **additional_params)

    def get_local_path(self, url, project_dir=None, ls_host=None, ls_access_token=None, task_id=None, *args, **kwargs):
        """
        Return the local path for a given URL.

        Args:
          url: The URL to find the local path for.
          project_dir: The project directory.
          ls_host: The Label Studio host,
            if not provided, it will be taken from LABEL_STUDIO_URL env variable
          ls_access_token: The access token for the Label Studio backend,
            if not provided, it will be taken from LABEL_STUDIO_API_KEY env variable
          task_id: Label Studio Task ID is required param for Cloud Storage URI resolving

        Returns:
          The local path for the given URL.
        """
        return get_local_path(
            url,
            project_dir=project_dir,
            hostname=ls_host,
            access_token=ls_access_token,
            task_id=task_id,
            *args,
            **kwargs
        )

    def preload_task_data(self, task: Dict, value=None, read_file=True):
        """ Preload task_data values using get_local_path() if values are URI/URL/local path.

        Args:
            task: Task root.
            value: task['data'] if it's None.
            read_file: If True, read file content. Otherwise, return file path only.

        Returns:
            Any: Preloaded task data value.
        """
        # recursively preload dict
        if isinstance(value, dict):
            for key, item in value.items():
                value[key] = self.preload_task_data(task=task, value=item, read_file=read_file)
            return value

        # recursively preload list
        elif isinstance(value, list):
            return [
                self.preload_task_data(task=task, value=item, read_file=read_file)
                for item in value
            ]

        # preload task data if value is URI/URL/local path
        elif isinstance(value, str) and is_preload_needed(value):
            filepath = self.get_local_path(url=value, task_id=task.get('id'))
            if not read_file:
                return filepath
            with open(filepath, 'r') as f:
                return f.read()

        # keep value as is
        return value

    ## TODO this should go into SDK
    def get_first_tag_occurence(
        self,
        control_type: Union[str, Tuple],
        object_type: Union[str, Tuple],
        name_filter: Optional[Callable] = None,
        to_name_filter: Optional[Callable] = None
    ) -> Tuple[str, str, str]:
        
        """
        Reads config and fetches the first control tag along with first object tag that matches the type.

        Args:
          control_type (str or tuple): The control type for checking tag matches.
          object_type (str or tuple): The object type for checking tag matches.
          name_filter (function, optional): If given, only tags with this name will be considered.
                                           Default is None.
          to_name_filter (function, optional): If given, only tags with this name will be considered.
                                              Default is None.

        Returns:
          tuple: (from_name, to_name, value), representing control tag, object tag and input value.        
        """
        return self.label_interface.get_first_tag_occurence(
            control_type=control_type,
            object_type=object_type,
            name_filter=name_filter,
            to_name_filter=to_name_filter
        )

    def build_label_map(self, tag_name: str, names: List[str]) -> Dict[str, str]:
        """Build a mapping between model label names and the label names defined in the Label Studio configuration.

        This function creates a dictionary that maps predicted label names from a machine learning model
        (e.g., "car", "truck") to the corresponding label names used in Label Studio (e.g., "Car").

        The mapping is primarily based on the `predicted_values` attribute within each <Label> tag in the
        Label Studio configuration. If a match is not found via `predicted_values`, the function attempts
        to match the labels directly by name, either by exact match or by a case-insensitive comparison.

        Args:
            tag_name (str): The name of the Label Studio tag that contains the labels (<Label>).
            names (List[str]): A list of label names used in the model, which need to be mapped
                               to the corresponding labels in Label Studio.

        Returns:
            Dict[str, str]: A dictionary where the keys are the label names from the model,
                            and the values are the corresponding label names from the Label Studio configuration.
                            The resulting dictionary is intended to facilitate the integration of model predictions
                            with Label Studio, ensuring that the labels used in predictions match the labels
                            configured in the tool.

        Example:
            Given a Label Studio configuration like:

            <View>
              <Image name="image" value="$image"/>
              <RectangleLabels name="label" toName="image" model_score_threshold="0.25">
                <Label value="Airplane" background="green"/>
                <Label value="Car" background="blue" predicted_values="car, truck"/>
              </RectangleLabels>
            </View>

            And a list of model names like:
            ["car", "truck", "airplane"]

            The function would return:
            {
                'car': 'Car',
                'truck': 'Car',
                'airplane': 'Airplane'
            }

        Notes:
            - If a label in `predicted_values` is not found in the model's labels, a warning is logged.
            - If `predicted_values` are not present or are empty, the function will attempt to map the label
              directly using the label's name, considering both case-sensitive and case-insensitive matches.
        """
        label_map = {}
        labels_attrs = self.label_interface.get_control(tag_name).labels_attrs

        model_labels = list(names)
        model_labels_lower = [label.lower() for label in model_labels]
        logger.debug(f"Labels supported by model for {tag_name}: {names}")

        if labels_attrs:
            for ls_label, label_tag in labels_attrs.items():
                # try to find `predicted_values` in Label tags
                predicted_values = label_tag.attr.get("predicted_values", "").split(",")
                matched = False
                for value in predicted_values:
                    value = value.strip()  # remove spaces at the beginning and at the end
                    if value and value in names:  # check if value is in model labels
                        if value not in model_labels:
                            logger.warning(f'Predicted value "{value}" is not in model labels')
                        label_map[value] = ls_label
                        matched = True

                # no `predicted_values`, use common Label's `value` attribute
                if not matched:
                    # model has the same label
                    if ls_label in model_labels:
                        label_map[ls_label] = ls_label
                    # model has the same lower name
                    elif ls_label.lower() in model_labels_lower:
                        label_map[ls_label.lower()] = ls_label

        logger.debug(f"Model Labels <=> Label Studio Labels:\n{label_map}")
        return label_map


def get_all_classes_inherited_LabelStudioMLBase(script_file):
    """
    Returns all classes in a provided script file that are inherited from LabelStudioMLBase.

    Args:
        script_file (str): The file path of a Python script.

    Returns:
        list[str]: A list of names of classes that inherit from LabelStudioMLBase.
    """
    names = set()
    abs_path = os.path.abspath(script_file)
    module_name = os.path.splitext(os.path.basename(script_file))[0]
    sys.path.append(os.path.dirname(abs_path))
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        print(Fore.RED + 'Can\'t import module "' + module_name + f'", reason: {e}.\n'
              'If you are looking for examples, you can find a dummy model.py here:\n' +
              Fore.LIGHTYELLOW_EX + 'https://labelstud.io/tutorials/dummy_model.html')
        module = None
        exit(-1)

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name == LabelStudioMLBase.__name__:
            continue
        if issubclass(obj, LabelStudioMLBase):
            names.add(name)
        for base in obj.__bases__:
            if LabelStudioMLBase.__name__ == base.__name__:
                names.add(name)
    sys.path.pop()
    return list(names)
