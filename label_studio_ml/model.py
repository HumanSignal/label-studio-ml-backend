import os
import logging
import sys
import json
import importlib
import importlib.util
import inspect

from typing import Tuple, Callable, Union, List, Dict, Optional
from abc import ABC, abstractmethod
from colorama import Fore

from label_studio_tools.core.label_config import parse_config
from label_studio_tools.core.utils.io import get_local_path
from .cache import create_cache

logger = logging.getLogger(__name__)

CACHE = create_cache(
    os.getenv('CACHE_TYPE', 'sqlite'),
    path=os.getenv('MODEL_DIR', '.'))


# Decorator to register predict function
_predict_fn: Callable
_update_fn: Callable


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
    
    TRAIN_EVENTS = (
        'ANNOTATION_CREATED',
        'ANNOTATION_UPDATED',
        'ANNOTATION_DELETED',
        'PROJECT_UPDATED'
    )

    def __init__(self, project_id: Optional[str] = None):
        """
        :param cache:
        """
        self.project_id = project_id or ''

    def use_label_config(self, label_config: str):
        current_label_config = self.get('label_config')
        if not current_label_config:
            # first time model is initialized
            self.set('model_version', 'INITIAL')
        if current_label_config != label_config:
            # label config has been changed
            self.set('label_config', label_config)
            self.set('parsed_label_config', json.dumps(parse_config(label_config)))

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
        return self.get('model_version')

    # @abstractmethod
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """
        Predict method should return a list of dicts with predictions for each task.
        :param tasks: a list of tasks
        :param context: a dict with additional context for model
        :param kwargs:
        :return: the list of dicts with predictions
        """

        # if there is a registered predict function, use it
        if _predict_fn:
            return _predict_fn(tasks, context, helper=self, **kwargs)

    def process_event(self, event, data, job_id, additional_params):
        if event in self.TRAIN_EVENTS:
            logger.debug(f'Job {job_id}: Received event={event}: calling {self.__class__.__name__}.fit()')
            train_output = self.fit(event=event, data=data, job_id=job_id, **additional_params)
            logger.debug(f'Job {job_id}: Train finished.')
            return train_output

    def fit(self, event, data, **additional_params):
        # if there is a registered update function, use it
        if _update_fn:
            return _update_fn(event, data, helper=self, **additional_params)

    def get_local_path(self, url, project_dir=None):
        return get_local_path(url, project_dir=project_dir, hostname=self.hostname, access_token=self.access_token)

    def get_first_tag_occurence(
        self,
        control_type: Union[str, Tuple],
        object_type: Union[str, Tuple],
        name_filter: Optional[Callable] = None,
        to_name_filter: Optional[Callable] = None
    ) -> Tuple[str, str, str]:
        """
        Reads config and returns the first control tag and the first object tag that match the given types
        For example: get_first_tag_occurence('Choices', 'Text') will return the first Choices tag that has an Text tag as input
        :param control_type:
        :param object_type:
        :param name_filter: if given, only tags with this name will be considered, e.g. name_filter=lambda name: name.startswith('my_')
        :param to_name_filter: if given, only tags with this name will be considered, e.g. to_name_filter=lambda name: toName.startswith('my_')
        :return: tuple of (from_name, to_name, value)
        """
        parsed_label_config = self.parsed_label_config
        for from_name, info in parsed_label_config.items():
            control_type_mathes = isinstance(control_type, str) and info['type'] == control_type or \
               isinstance(control_type, tuple) and info['type'] in control_type
            control_name_matches = name_filter is None or name_filter(from_name)

            if control_type_mathes and control_name_matches:

                for input_name, input in zip(info['to_name'], info['inputs']):
                    object_type_matches = isinstance(object_type, str) and input['type'] == object_type or \
                       isinstance(object_type, tuple) and input['type'] in object_type
                    object_name_matches = to_name_filter is None or to_name_filter(input_name)

                    if object_type_matches and object_name_matches:
                        return from_name, info['to_name'][0], input['value']
        raise ValueError(f'No control tag of type {control_type} and object tag of type {object_type} found in label config')


def get_all_classes_inherited_LabelStudioMLBase(script_file):
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
