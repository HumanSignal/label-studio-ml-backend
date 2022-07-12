import json
import logging
import requests

from PIL import Image

from label_studio_tools.core.utils.params import get_env
from label_studio_tools.core.utils.io import get_local_path

DATA_UNDEFINED_NAME = '$undefined$'

logger = logging.getLogger(__name__)


def get_single_tag_keys(parsed_label_config, control_type, object_type):
    """
    Gets parsed label config, and returns data keys related to the single control tag and the single object tag schema
    (e.g. one "Choices" with one "Text")
    :param parsed_label_config: parsed label config returned by "label_studio.misc.parse_config" function
    :param control_type: control tag str as it written in label config (e.g. 'Choices')
    :param object_type: object tag str as it written in label config (e.g. 'Text')
    :return: 3 string keys and 1 array of string labels: (from_name, to_name, value, labels)
    """
    assert len(parsed_label_config) == 1
    from_name, info = list(parsed_label_config.items())[0]
    assert info['type'] == control_type, 'Label config has control tag "<' + info['type'] + '>" but "<' + control_type + '>" is expected for this model.'  # noqa

    assert len(info['to_name']) == 1
    assert len(info['inputs']) == 1
    assert info['inputs'][0]['type'] == object_type
    to_name = info['to_name'][0]
    value = info['inputs'][0]['value']
    return from_name, to_name, value, info['labels']


def is_skipped(completion):
    if len(completion['annotations']) != 1:
        return False
    completion = completion['annotations'][0]
    return completion.get('skipped', False) or completion.get('was_cancelled', False)


def get_choice(completion):
    return completion['annotations'][0]['result'][0]['value']['choices'][0]


def get_image_local_path(url, image_cache_dir=None, project_dir=None, image_dir=None):
    return get_local_path(url, image_cache_dir, project_dir, get_env('HOSTNAME'), image_dir)


def get_image_size(filepath):
    return Image.open(filepath).size


def get_annotated_dataset(project_id, hostname=None, api_key=None):
    """Just for demo purposes: retrieve annotated data from Label Studio API"""
    if hostname is None:
        hostname = get_env('HOSTNAME')
    if api_key is None:
        api_key = get_env('API_KEY')
    download_url = f'{hostname.rstrip("/")}/api/projects/{project_id}/export'
    response = requests.get(download_url, headers={'Authorization': f'Token {api_key}'})
    if response.status_code != 200:
        raise Exception(f"Can't load task data using {download_url}, "
                        f"response status_code = {response.status_code}")
    return json.loads(response.content)
