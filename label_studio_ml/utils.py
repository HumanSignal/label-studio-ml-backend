import logging

from PIL import Image, ImageOps
from collections import OrderedDict

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


def get_first_tag_keys(parsed_label_config, control_type, object_type):
    """
    Reads config and returns the first control tag and the first object tag that match the given types
    :param parsed_label_config:
    :param control_type:
    :param object_type:
    :return:
    """
    for from_name, info in parsed_label_config.items():
        if info['type'] == control_type:
            for input in info['inputs']:
                if input['type'] == object_type:
                    return from_name, info
    return None, None


def is_skipped(completion):
    if len(completion['annotations']) != 1:
        return False
    completion = completion['annotations'][0]
    return completion.get('skipped', False) or completion.get('was_cancelled', False)


def get_choice(completion):
    return completion['annotations'][0]['result'][0]['value']['choices'][0]


def get_image_local_path(url, image_cache_dir=None, project_dir=None, image_dir=None,
                         label_studio_host=None, label_studio_access_token=None):
    image_local_path = get_local_path(
        url=url,
        cache_dir=image_cache_dir,
        project_dir=project_dir,
        hostname=label_studio_host or get_env('HOSTNAME'),
        image_dir=image_dir,
        access_token=label_studio_access_token
    )
    logger.debug(f'Image stored in the local path: {image_local_path}')
    return image_local_path


def get_image_size(filepath):
    img = Image.open(filepath)
    img = ImageOps.exif_transpose(img)
    return img.size


class InMemoryLRUDictCache:
    def __init__(self, capacity=1):
        self.cache = OrderedDict()
        self.capacity = capacity

    def __contains__(self, item):
        return item in self.cache

    def get(self, key):
        if key in self.cache:
            # Move the accessed item to the end
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            # Move the updated item to the end
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # Pop the first item if cache reached its capacity
            self.cache.popitem(last=False)

        self.cache[key] = value

    def __str__(self):
        return str(self.cache)


if __name__ == "__main__":
    c = InMemoryLRUDictCache(2)
    c.put(1, 1)
    c.put(2,2)
    print(c.cache)