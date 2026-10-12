"""Compatibility exports, loaded on first use."""

from imagedt._lazy import resolve as _resolve

_EXPORTS = {
    "alpha_to_color": (".image", "alpha_to_color"),
    "IMG_EXTENSIONS": (".image", "IMG_EXTENSIONS"),
    "clear_dir": (".image", "clear_dir"),
    "noise_padd": (".image", "noise_padd"),
    "remove_broken_image": (".image", "remove_broken_image"),
    "resize_with_scale": (".image", "resize_with_scale"),
    "padd_pixel": (".image", "padd_pixel"),
    "reduce_imagenet_mean": (".image", "reduce_imagenet_mean"),
    "swap_chanel_to_RGB": (".image", "swap_chanel_to_RGB"),
    "vgg_preprocesing": (".image", "vgg_preprocesing"),
    "inception_preprocesing": (".image", "inception_preprocesing"),
    "our_processing": (".image", "our_processing"),
    "md5": (".image", "md5"),
    "get_box": (".image", "get_box"),
    "put_cvtext": (".image", "put_cvtext"),
    "draw_cvrect": (".image", "draw_cvrect"),
    "cv_show": (".image", "cv_show"),
    "resize_dir_images_with_scale": (".image", "resize_dir_images_with_scale"),
    "points_to_contour": (".image", "points_to_contour"),
    "points_to_contours": (".image", "points_to_contours"),
    "put_text": (".image", "put_text"),
    "draw_contours": (".image", "draw_contours"),
    "polly_points2contours": (".image", "polly_points2contours"),
    "rename_imfiles_with_md5": (".image", "rename_imfiles_with_md5"),
    "DIR_TYPR": (".file", "DIR_TYPR"),
    "FilesOp": (".file", "FilesOp"),
    "Anno_OP": (".file", "Anno_OP"),
    "write_txt": (".file", "write_txt"),
    "write_csv": (".file", "write_csv"),
    "read_csv": (".file", "read_csv"),
    "readlines": (".file", "readlines"),
    "write_json": (".file", "write_json"),
    "read_json": (".file", "read_json"),
    "decode_strs": (".file", "decode_strs"),
    "writeExcel": (".file", "writeExcel"),
    "LoopFile": (".dir", "LoopFile"),
    "loop": (".dir", "loop"),
    "get_dir_images": (".dir", "get_dir_images"),
    "time_cost": (".decorator", "time_cost"),
    "getip": (".tools", "getip"),
    "detect_eval": (".tools", "detect_eval"),
    "f1_score": (".tools", "f1_score"),
    "plot_tools": (".tools", "plot_tools"),
    "text_generator": (".tools", "text_generator"),
    "pdf": (".tools", "pdf"),
    "metrics": (".tools", "metrics"),
    "image": (".image", None),
    "file": (".file", None),
    "dir": (".dir", None),
    "decorator": (".decorator", None),
    "tools": (".tools", None),
    "caffe": (".caffe", None),
    "tensorflow": (".tensorflow", None),
    "torch": (".torch", None),
    "converters": (".converters", None),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(name, _EXPORTS, globals(), __name__)


def __dir__():
    return sorted(set(globals()) | set(__all__))


__version__ = "0.6.0rc1"
