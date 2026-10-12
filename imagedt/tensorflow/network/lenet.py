"""LeNet with the historical endpoint names and a maintained Keras backend."""

from imagedt._backends import require


def lenet(
    images,
    num_classes=10,
    is_training=True,
    drop_keep_prob=0.5,
    prediction_fn=None,
    scope="LeNet",
):
    tf = require("tensorflow", "tensorflow")
    if not 0 < drop_keep_prob <= 1:
        raise ValueError("drop_keep_prob must be in (0, 1]")
    layers = tf.keras.layers
    endpoints = {}
    with tf.name_scope(scope):
        net = endpoints["conv1"] = layers.Conv2D(
            32, 5, padding="same", activation="relu", name="conv1"
        )(images)
        net = endpoints["pool1"] = layers.MaxPool2D(2, name="pool1")(net)
        net = endpoints["conv2"] = layers.Conv2D(
            64, 5, padding="same", activation="relu", name="conv2"
        )(net)
        net = endpoints["pool2"] = layers.MaxPool2D(2, name="pool2")(net)
        net = endpoints["Flatten"] = layers.Flatten(name="Flatten")(net)
        net = endpoints["fc3"] = layers.Dense(1024, activation="relu", name="fc3")(net)
        if not num_classes:
            return net, endpoints
        net = endpoints["dropout3"] = layers.Dropout(
            1 - drop_keep_prob, name="dropout3"
        )(net, training=is_training)
        logits = endpoints["Logits"] = layers.Dense(num_classes, name="fc4")(net)
        endpoints["Predictions"] = (
            tf.nn.softmax(logits, name="Predictions")
            if prediction_fn is None
            else prediction_fn(logits, scope="Predictions")
        )
        return logits, endpoints


lenet.default_image_size = 28
