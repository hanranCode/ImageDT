"""Preview up to two images from a TFRecord without altering global graph state."""

from io import BytesIO

from PIL import Image

from imagedt._backends import tensorflow_v1


def visual_records(tfrecords_path):
    tf = tensorflow_v1()
    for index, raw in enumerate(tf.io.tf_record_iterator(str(tfrecords_path))):
        if index == 2:
            break
        example = tf.train.Example.FromString(raw)
        data = example.features.feature["image/encoded"].bytes_list.value[0]
        with Image.open(BytesIO(data)) as image:
            image.convert("RGB").show()
