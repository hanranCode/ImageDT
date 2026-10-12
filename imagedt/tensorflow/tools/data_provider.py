import cv2
import numpy as np

import imagedt
from imagedt._backends import tensorflow_v1

from ...image.process import padd_pixel
from .DataInterface import DataSet as DataSet
from .DataInterface import DataSets as DataSets

tf = tensorflow_v1()


def dynamically_loaded_data(image_paths, labels, height=224, width=224):
    if len(image_paths) != len(labels):
        raise ValueError("image paths and labels must have equal length")
    images = np.empty((len(image_paths), height, width, 3), dtype=np.float32)
    for index, path in enumerate(image_paths):
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        images[index] = padd_pixel(image, width, height)
    return images, list(map(int, labels))


def resize_image_keep_aspect(image, max_edge=224):
    if max_edge < 1:
        raise ValueError("max_edge must be positive")
    shape = tf.shape(image)[:2]
    ratio = tf.cast(max_edge, tf.float32) / tf.cast(tf.reduce_max(shape), tf.float32)
    target = tf.maximum(
        1, tf.cast(tf.round(tf.cast(shape, tf.float32) * ratio), tf.int32)
    )
    return tf.image.resize_images(image, target)


def tf_noise_padd(images, max_edge=224, start_pixel=0, end_pixel=255):
    if not 0 <= start_pixel < end_pixel <= 256:
        raise ValueError("invalid pixel range")
    images = resize_image_keep_aspect(images, max_edge)
    shape = tf.shape(images)
    top = (max_edge - shape[0]) // 2
    left = (max_edge - shape[1]) // 2
    paddings = [
        [top, max_edge - shape[0] - top],
        [left, max_edge - shape[1] - left],
        [0, 0],
    ]
    padded = tf.pad(images, paddings)
    mask = tf.pad(tf.ones_like(images), paddings)
    noise = tf.random_uniform([max_edge, max_edge, shape[2]], start_pixel, end_pixel)
    return padded + noise * (1 - mask)


def tf_JitterCut(images, jitter=0.05):
    if not 0 <= jitter < 1:
        raise ValueError("jitter must be in [0, 1)")
    shape = tf.shape(images)
    ratio = tf.random_uniform([], 0, jitter) if jitter else tf.constant(0.0)
    target = tf.maximum(
        1, tf.cast(tf.cast(shape[:2], tf.float32) * (1 - ratio), tf.int32)
    )
    return tf.random_crop(images, [target[0], target[1], shape[2]])


class Data_Provider:
    """docstring for Data_Provider"""

    def __init__(self, data_dir, num_classes):
        super().__init__()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self._init_dataset_infos()
        self._init_reader()

    @property
    def _get_tfrecords(self, file_format=(".tfrecord",)):
        return imagedt.dir.loop(self.data_dir, file_format)

    def _search_laebls_file(self):
        sear_files = imagedt.dir.loop(self.data_dir, [".txt"])
        sear_label_file = [
            item for item in sear_files if ("label" in item) and item.endswith(".txt")
        ]
        if sear_label_file:
            lines = imagedt.file.readlines(sear_label_file[0])
            self.num_classes = len(lines)

    def _init_dataset_infos(self):
        self._search_laebls_file()

    @property
    def _log_dataset_infos(self):
        print("##" * 20)
        print(
            f"Dataset infos:\ndata_dir: {self.data_dir} \nnum_classes: {self.num_classes}"
        )
        print("##" * 20)

    def _init_reader(self):
        # read all tfrecord files
        self.reader = tf.TFRecordReader()
        record_files = self._get_tfrecords
        self.filequeue = tf.train.string_input_producer(record_files)

    def read_from_tfrecord(self, batch_size=32, image_shape=(224, 224, 3)):
        self.image_shape = image_shape
        _, fetch_tensors = self.reader.read(self.filequeue)
        load_features = tf.parse_single_example(
            fetch_tensors,
            features={
                "image/height": tf.FixedLenFeature([], dtype=tf.int64),
                "image/width": tf.FixedLenFeature([], dtype=tf.int64),
                "image/class/label": tf.FixedLenFeature([], dtype=tf.int64),
                "image/encoded": tf.FixedLenFeature([], dtype=tf.string),
            },
        )
        label = tf.cast(load_features["image/class/label"], tf.int64)
        image = tf.image.decode_image(
            load_features["image/encoded"],
            channels=image_shape[2],
            expand_animations=False,
        )
        image.set_shape([None, None, image_shape[2]])
        image = tf.image.resize_image_with_pad(image, image_shape[0], image_shape[1])
        image.set_shape(image_shape)
        # make a batch
        with tf.control_dependencies(
            [
                tf.assert_greater_equal(label, tf.constant(0, dtype=tf.int64)),
                tf.assert_less(label, tf.cast(self.num_classes, tf.int64)),
            ]
        ):
            label = tf.one_hot(label, self.num_classes)
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            capacity=500,
            min_after_dequeue=100,
            num_threads=4,
        )
        return image_batch, label_batch
