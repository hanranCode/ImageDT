"""Retained Caffe-to-CoreML entry with an explicit unsupported-backend error."""


def convert(
    model_path,
    prototxt_path,
    label_path,
    red=-123,
    green=-117,
    blue=-104,
    scale=1.0,
    bgr=False,
    output_model_name="sku_cls_model_noise.mlmodel",
):
    raise NotImplementedError(
        "Modern coremltools removed the Caffe converter. "
        "Convert the model to a supported PyTorch or TensorFlow format first."
    )
