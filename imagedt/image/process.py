import hashlib
import os

import cv2
import numpy as np

# from PIL import Image
from ..dir.dir_loop import loop

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def clear_dir(images_dir, debug=True, multi_process=4):
    """Remove unreadable images and wait for all workers to finish."""
    from concurrent.futures import ThreadPoolExecutor

    if not isinstance(multi_process, int) or multi_process < 1:
        raise ValueError("multi_process must be a positive integer")
    paths = loop(images_dir, IMG_EXTENSIONS)

    def check(path):
        if cv2.imread(path) is None:
            os.remove(path)
            if debug:
                print(f"Remove broken image: {path}")

    with ThreadPoolExecutor(max_workers=multi_process) as executor:
        list(executor.map(check, paths))


def _pad_image(img, width, height, low, high):
    if img.ndim != 3 or img.shape[2] != 3 or min(img.shape[:2]) < 1:
        raise ValueError("expected a nonempty HWC image with three channels")
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or min(width, height) < 1
    ):
        raise ValueError("target dimensions must be positive integers")
    if not 0 <= low < high <= 256:
        raise ValueError("pixel range must satisfy 0 <= low < high <= 256")
    h, w = img.shape[:2]
    ratio = min(width / w, height / h)
    rw = min(width, max(1, round(w * ratio)))
    rh = min(height, max(1, round(h * ratio)))
    # Retain the old symmetric padding behavior, including parity adjustment.
    rw = min(width, rw + (width - rw) % 2)
    rh = min(height, rh + (height - rh) % 2)
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    canvas = np.random.randint(low, high, (height, width, 3)).astype(np.uint8)
    y, x = (height - rh) // 2, (width - rw) // 2
    canvas[y : y + rh, x : x + rw] = resized
    return canvas


def noise_padd(img, edge_size=224, start_pixel_value=0, end_pixel_value=256):
    return _pad_image(img, edge_size, edge_size, start_pixel_value, end_pixel_value)


def remove_broken_image(data_dir):
    image_files = loop(data_dir, IMG_EXTENSIONS)

    for image_file in image_files:
        try:
            img_mat = cv2.imread(image_file)
            if img_mat is None:
                os.remove(image_file)
                print(f"remove broken image {image_file}")
        except cv2.error:
            os.remove(image_file)
            print(f"remove broken file {image_file}")


def resize_with_scale(cvmat, max_length):
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    h, w = cvmat.shape[:2]
    max_edge = max(h, w)
    if max_edge > max_length:
        ratio = float(max_edge) / max_length
        width, height = w / ratio, h / ratio
        cvmat = cv2.resize(cvmat, (max(1, int(width)), max(1, int(height))))
    return cvmat


def padd_pixel(img, edge_size_w=200, edge_size_h=96):
    return _pad_image(img, edge_size_w, edge_size_h, 230, 240)


def reduce_imagenet_mean(cv_mat):
    [B, G, R] = cv2.split(cv_mat)
    return cv2.merge([B - _B_MEAN, G - _G_MEAN, R - _R_MEAN])


def swap_chanel_to_RGB(cvmat):
    return cv2.cvtColor(cvmat, cv2.COLOR_BGR2RGB)


def vgg_preprocesing(cvmat):
    cvmat = np.array(cvmat, dtype=np.float32)
    cvmat -= [_R_MEAN, _G_MEAN, _B_MEAN]
    return cvmat


def inception_preprocesing(cvmat):
    cvmat = (cvmat / 255.0 - 0.5) * 2
    cvmat = np.array(cvmat, dtype=np.float32)
    return cvmat


def our_processing(cvmat):
    cvmat = reduce_imagenet_mean(cvmat)
    cvmat = np.array(cvmat, dtype=np.float32)
    return np.expand_dims(cvmat, axis=0)


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_box(bndbox, loca_type="mid"):
    if loca_type == "top":
        # xmid = bndbox[0]+(bndbox[2]-bndbox[0])/2.
        xmid = bndbox[0]
        ymid = bndbox[1]
    elif loca_type == "bot":
        xmid = bndbox[0]
        ymid = max(0, bndbox[3] - 5)
    elif loca_type == "botup":
        xmid = bndbox[0]
        ymid = max(0, bndbox[3] - (bndbox[3] - bndbox[1]) / 4)
    else:
        # xmid = bndbox[0]+(bndbox[2]-bndbox[0])/2.
        xmid = bndbox[0]
        ymid = bndbox[1] + (bndbox[3] - bndbox[1]) / 2.0
    return (int(xmid), int(ymid))


def put_cvtext(
    img, text, posi, color=(0, 0, 255), fontScale=1, thickness=1, font_type=None
):
    if font_type is None:
        font = cv2.FONT_HERSHEY_DUPLEX
    else:
        font = font_type
    cv2.putText(
        img=img,
        text=text,
        org=posi,
        fontFace=font,
        fontScale=fontScale,
        color=color,
        thickness=thickness,
        lineType=2,
    )


def draw_cvrect(
    img,
    bndboxes,
    box_color=(0, 0, 255),
    border=1,
    font_color=(0, 255, 0),
    draw_strs=None,
    fontScale=3,
    compare_strs=None,
    draw_location="mid",
    font_type=None,
    thickness=2,
):
    for index, bndbox in enumerate(bndboxes):
        bndbox = list(map(int, bndbox))
        left_up = (bndbox[0], bndbox[1])
        right_bottom = (bndbox[2], bndbox[3])
        cv2.rectangle(img, left_up, right_bottom, box_color, border)
        mid_box = get_box(bndbox, loca_type=draw_location)
        if draw_strs is not None:
            show_str = draw_strs[index]
        else:
            show_str = str(index)
        put_cvtext(
            img,
            show_str,
            mid_box,
            color=font_color,
            fontScale=fontScale,
            thickness=thickness,
            font_type=font_type,
        )
    return img


def cv_show(imgs, win_name="show_image"):
    if not isinstance(imgs, list):
        imgs = [imgs]
    for index, img in enumerate(imgs):
        cv2.imshow(win_name + str(index), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_dir_images_with_scale(image_dir, resize_max_edge=224):
    for path in loop(image_dir, IMG_EXTENSIONS):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        resized = resize_with_scale(img, resize_max_edge)
        if resized is not img and not cv2.imwrite(path, resized):
            raise OSError(f"Cannot write image: {path}")


def points_to_contour(points):
    contours = [[list(p)] for p in points]
    return np.asarray(contours, dtype=np.int32)


def points_to_contours(points):
    return np.asarray([points_to_contour(points)])


def put_text(img, text, pos, scale=1, color=(255, 255, 255), thickness=1):
    pos = np.int32(pos)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img=img,
        text=text,
        org=tuple(pos),
        fontFace=font,
        fontScale=scale,
        color=color,
        thickness=thickness,
    )


def draw_contours(img, contours, idx=-1, color=1, border_width=1):
    cv2.drawContours(img, contours, idx, color, border_width)
    return img


def polly_points2contours(points):
    return np.asarray(points, dtype=np.int32).reshape(-1, 4, 1, 2)


def rename_imfiles_with_md5(imdir, rename_xml=False):
    """Rename images by content hash; preflight and roll back paired XML changes."""
    from pathlib import Path

    from ..file.file_operate import FilesOp

    for image in loop(imdir):
        image_path = Path(image)
        digest = md5(image_path)
        paths = [str(image_path)]
        if rename_xml:
            parts = list(image_path.parts)
            matching = [
                index for index, part in enumerate(parts) if part == "JPEGImages"
            ]
            if matching:
                parts[matching[-1]] = "Annotations"
            annotation = Path(*parts).with_suffix(".xml")
            paths.append(str(annotation))
        FilesOp.rename_data_pairs(paths, digest)
