import os
import logging
import sys
import json
import importlib
import importlib.util
import inspect
from semver import Version

from typing import Tuple, Callable, Union, List, Dict, Optional
from abc import ABC, abstractmethod
from colorama import Fore

from label_studio_sdk.label_interface import LabelInterface
from label_studio_tools.core.label_config import parse_config
from label_studio_tools.core.utils.io import get_local_path
from .response import ModelResponse
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
        'PROJECT_UPDATED'
    )

    def __init__(self, project_id: Optional[str] = None, label_config=None):
        """
        Initialize LabelStudioMLBase with a project ID.

        Args:
            project_id (str, optional): The project ID. Defaults to None.
        """
        self.project_id = project_id or ''
        self.use_label_config(label_config)

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
        
        mv.bump_minor()
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

    def get_local_path(self, url, project_dir=None):
        """
        Return the local path for a given URL.

        Args:
          url: The URL to find the local path for.
          project_dir: The project directory.

        Returns:
          The local path for the given URL.
        """
        return get_local_path(url, project_dir=project_dir, hostname=self.hostname, access_token=self.access_token)

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
            to_name_filter=to_name_filter)        


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
