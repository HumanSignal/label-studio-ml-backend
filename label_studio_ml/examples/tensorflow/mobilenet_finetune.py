import os
import logging
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path, get_single_tag_keys, get_choice, is_skipped

logger = logging.getLogger(__name__)
feature_extractor_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'


class TFMobileNet(LabelStudioMLBase):

    def __init__(self, trainable=False, batch_size=32, epochs=3, **kwargs):
        super(TFMobileNet, self).__init__(**kwargs)

        self.image_width, self.image_height = 224, 224
        self.trainable = trainable
        self.batch_size = batch_size
        self.epochs = epochs

        self.feature_extractor_layer = hub.KerasLayer(
            feature_extractor_model,
            input_shape=(self.image_width, self.image_height, 3),
            trainable=trainable)

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'Choices', 'Image')
        self.labels = tf.convert_to_tensor(sorted(self.labels_in_config))
        num_classes = len(self.labels_in_config)
        self.model = tf.keras.Sequential([
            self.feature_extractor_layer,
            tf.keras.layers.Dense(num_classes)
        ])
        self.model.summary()
        if self.train_output:
            model_file = self.train_output['model_file']
            logger.info('Restore model from ' + model_file)
            # Restore previously saved weights
            self.model.load_weights(self.train_output['model_file'])

    def predict(self, tasks, **kwargs):
        image_path = get_image_local_path(tasks[0]['data'][self.value])

        image = Image.open(image_path).resize((self.image_width, self.image_height))
        image = np.array(image) / 255.0
        result = self.model.predict(image[np.newaxis, ...])
        predicted_label_idx = np.argmax(result[0], axis=-1)
        predicted_label_score = result[0][predicted_label_idx]
        predicted_label = self.labels[predicted_label_idx]
        return [{
            'result': [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'choices',
                'value': {'choices': [str(predicted_label.numpy(), 'utf-8')]}
            }],
            'score': float(predicted_label_score)
        }]

    def fit(self, completions, workdir=None, **kwargs):

        annotations = []
        for completion in completions:
            if is_skipped(completion):
                continue
            image_path = get_image_local_path(completion['data'][self.value])
            image_label = get_choice(completion)
            annotations.append((image_path, image_label))

        # Create dataset
        ds = tf.data.Dataset.from_tensor_slices(annotations)

        def prepare_item(item):
            label = tf.argmax(item[1] == self.labels)
            img = tf.io.read_file(item[0])
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [self.image_height, self.image_width])
            return img, label

        ds = ds.map(prepare_item, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache().shuffle(buffer_size=1000).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['acc'])
        self.model.fit(ds, epochs=self.epochs)
        model_file = os.path.join(workdir, 'checkpoint')
        self.model.save_weights(model_file)
        return {'model_file': model_file}
