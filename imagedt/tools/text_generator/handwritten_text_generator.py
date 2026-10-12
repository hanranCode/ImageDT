"""Compatibility entry for an unsupported pretrained handwriting pipeline."""


class HandwrittenTextGenerator:
    @classmethod
    def generate(cls, text):
        raise NotImplementedError(
            "The legacy handwriting pipeline depends on an unbundled TensorFlow 1 model. "
            "Use ComputerTextGenerator or supply a maintained handwriting backend."
        )
