# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import os
import coremltools

# try:
#     import coremltools
# except ImportError:
#     os.system("pip install coremltools")
# finally:
#     print("No module named coremltools.")


def convert(model_path, prototxt_path, label_path, red=-123, green=-117, blue=-104, 
                scale=1.0, bgr=False, output_model_name='sku_cls_model_noise.mlmodel'):
    """
    Args: red, grenn, blue should be nagative: like -123, -117, -104
        output_model_name: default, sku_cls_model_noise.mlmodel
        label_path: Classes are sorted by incrementing。 
            liby_0, 0
            liby_1, 1
            liby_2, 2
            liby_3, 3
            liby_4, 4
    Returns: auto-saved coreml model in source path.
    """

    coreml_model = coremltools.converters.caffe.convert((
        model_path, prototxt_path),
        image_input_names='data',
        is_bgr=bgr,
        image_scale=scale,
        red_bias= red*scale,
        green_bias=green*scale,
        blue_bias=blue*scale,
        class_labels=label_path,
    )
    coreml_model.save(os.path.join(os.path.dirname(model_path), output_model_name))
    print("finished converte {0} to {1}".format(os.path.basename(model_path), output_model_name))
    return
