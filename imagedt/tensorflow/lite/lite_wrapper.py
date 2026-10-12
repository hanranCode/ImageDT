"""Single-input float image classifier using the public Lite interpreter."""

import numpy as np

from imagedt._backends import require


class Lite_Wrapper:
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()
        self._check_model_type()

    def load_model(self):
        backend = require("ai_edge_litert.interpreter", "lite")
        self.interpreter = backend.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _check_model_type(self):
        if len(self.input_details) != 1 or self.input_details[0]["dtype"] != np.float32:
            raise ValueError("Only a single float32 input is supported")
        shape = self.input_details[0]["shape"]
        if len(shape) != 4 or shape[0] != 1:
            raise ValueError("Expected a single-image NHWC input")
        self.height, self.width = map(int, shape[1:3])

    def predict(self, input_data):
        data = np.asarray(input_data)
        if data.dtype != np.float32 or tuple(data.shape) != tuple(
            self.input_details[0]["shape"]
        ):
            raise ValueError("input dtype or shape does not match the model")
        self.interpreter.set_tensor(self.input_details[0]["index"], data)
        self.interpreter.invoke()
        predictions = np.asarray(
            self.interpreter.get_tensor(self.output_details[0]["index"])
        ).reshape(-1)
        pre_class = int(np.argmax(predictions))
        return pre_class, predictions[pre_class]
