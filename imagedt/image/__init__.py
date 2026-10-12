"""Compatibility exports, loaded on first use."""

from imagedt._lazy import resolve as _resolve

_EXPORTS = {
    "alpha_to_color": (".image_pil", "alpha_to_color"),
    "IMG_EXTENSIONS": (".process", "IMG_EXTENSIONS"),
    "clear_dir": (".process", "clear_dir"),
    "noise_padd": (".process", "noise_padd"),
    "remove_broken_image": (".process", "remove_broken_image"),
    "resize_with_scale": (".process", "resize_with_scale"),
    "padd_pixel": (".process", "padd_pixel"),
    "reduce_imagenet_mean": (".process", "reduce_imagenet_mean"),
    "swap_chanel_to_RGB": (".process", "swap_chanel_to_RGB"),
    "vgg_preprocesing": (".process", "vgg_preprocesing"),
    "inception_preprocesing": (".process", "inception_preprocesing"),
    "our_processing": (".process", "our_processing"),
    "md5": (".process", "md5"),
    "get_box": (".process", "get_box"),
    "put_cvtext": (".process", "put_cvtext"),
    "draw_cvrect": (".process", "draw_cvrect"),
    "cv_show": (".process", "cv_show"),
    "resize_dir_images_with_scale": (".process", "resize_dir_images_with_scale"),
    "points_to_contour": (".process", "points_to_contour"),
    "points_to_contours": (".process", "points_to_contours"),
    "put_text": (".process", "put_text"),
    "draw_contours": (".process", "draw_contours"),
    "polly_points2contours": (".process", "polly_points2contours"),
    "rename_imfiles_with_md5": (".process", "rename_imfiles_with_md5"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(name, _EXPORTS, globals(), __name__)


def __dir__():
    return sorted(set(globals()) | set(__all__))
