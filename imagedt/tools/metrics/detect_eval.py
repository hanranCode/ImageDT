import os
import xml.etree.ElementTree as ET

import numpy as np

from imagedt.dir.dir_loop import loop


def parse_rec(filename):
    """Parse a PASCAL VOC xml file"""
    tree = ET.parse(filename)
    objects = {}
    for obj in tree.findall("object"):
        obj_struct = {}
        name = obj.find("name").text

        if name not in objects:
            objects[name] = []
        # obj_struct['pose'] = obj.find('pose').text
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct["name"] = name
        obj_struct["difficult"] = int(obj.findtext("difficult", "0"))

        obj_struct["confidence"] = (
            obj.find("confidence").text if obj.find("confidence") is not None else 0.99
        )
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            float(bbox.find("xmin").text),
            float(bbox.find("ymin").text),
            float(bbox.find("xmax").text),
            float(bbox.find("ymax").text),
        ]
        objects[obj_struct["name"]].append(obj_struct)
    return objects


def parse_detect(detannos):
    xmls_path = loop(detannos, [".xml"])
    print(f"########## Reading {len(xmls_path)} annotation files of DT.##########")
    infos = []
    for xml_path in xmls_path:
        info = parse_rec(xml_path)
        info = [item for cls_key in info for item in info[cls_key]]
        basename = os.path.basename(xml_path)
        infos.append(
            [
                os.path.splitext(basename)[0],
                [[item["confidence"]] + [item["name"]] + item["bbox"] for item in info],
            ]
        )

    return [[item[0], item[1]] for item in infos]


def load_detect_lines(detannos, conf):
    txts_path = loop(detannos, [".txt"])
    print(f"########## Reading {len(txts_path)} annotation files of DT.##########")
    infos = []
    for txt_path in txts_path:
        with open(txt_path) as f:
            lines = f.readlines()
        true_class = os.path.basename(txt_path).split(".")[0].split("_")[-1]
        # 图片名[无后缀] 置信度 x1 y1 x2 y2
        # target: image_name, conf, det_class, [xmin, ymin, xmax, ymax]
        for line in lines:
            line = line.strip().split()
            if float(line[1]) < conf:
                continue
            infos.append(line[0:2] + [true_class] + line[2:])
    return infos


def get_xmls_basename(xml_path):
    return [
        os.path.splitext(os.path.basename(item))[0] for item in loop(xml_path, [".xml"])
    ]


def voc_eval(
    detpath, annopath, ovthresh=0.5, det_file_type="xml", conf=0.5, set_gt_labels=None
):
    """Macro F1 with class-aware, confidence-ordered VOC box matching.

    VOC coordinates are inclusive; an IoU strictly greater than ovthresh matches.
    Difficult ground truths are ignored. Empty evaluated label sets score zero.
    """
    if det_file_type not in ("xml", "txt"):
        raise ValueError("det_file_type must be xml or txt")
    if not 0 <= ovthresh <= 1 or not 0 <= conf <= 1:
        raise ValueError("thresholds must be between zero and one")
    truth = {}
    labels = set()
    for path in loop(annopath, [".xml"]):
        key = os.path.splitext(os.path.basename(path))[0]
        if key in truth:
            raise ValueError(f"Duplicate ground-truth image id: {key}")
        truth[key] = [obj for group in parse_rec(path).values() for obj in group]
        labels.update(obj["name"] for obj in truth[key])
    if det_file_type == "xml":
        detections = [
            [key] + obj
            for key, objects in parse_detect(detpath)
            for obj in objects
            if float(obj[0]) >= conf
        ]
    else:
        detections = load_detect_lines(detpath, conf)
    labels.update(str(row[2]) for row in detections)
    labels = set(map(str, set_gt_labels)) if set_gt_labels is not None else labels
    labels.discard("0")
    tp = dict.fromkeys(labels, 0)
    fp = dict.fromkeys(labels, 0)
    totals = {
        label: sum(
            obj["name"] == label and not obj["difficult"]
            for objects in truth.values()
            for obj in objects
        )
        for label in labels
    }
    matched = set()
    for row in sorted(detections, key=lambda row: -float(row[1])):
        image, _, label, *coords = row
        label = str(label)
        if label not in labels:
            continue
        box = np.asarray(coords, dtype=float)
        if box.shape != (4,) or np.any(box[2:] < box[:2]):
            raise ValueError("invalid detection box")
        candidates = [
            (i, obj)
            for i, obj in enumerate(truth.get(image, []))
            if obj["name"] == label
        ]
        if not candidates:
            fp[label] += 1
            continue
        boxes = np.asarray([obj["bbox"] for _, obj in candidates], dtype=float)
        intersection = np.prod(
            np.maximum(
                np.minimum(boxes[:, 2:], box[2:])
                - np.maximum(boxes[:, :2], box[:2])
                + 1,
                0,
            ),
            axis=1,
        )
        union = (
            np.prod(box[2:] - box[:2] + 1)
            + np.prod(boxes[:, 2:] - boxes[:, :2] + 1, axis=1)
            - intersection
        )
        overlap = np.divide(
            intersection, union, out=np.zeros_like(intersection), where=union > 0
        )
        best = int(np.argmax(overlap))
        index, obj = candidates[best]
        if overlap[best] > ovthresh:
            if obj["difficult"]:
                continue
            if (image, index) not in matched:
                matched.add((image, index))
                tp[label] += 1
                continue
        fp[label] += 1
    scores = [
        2 * tp[label] / max(totals[label] + tp[label] + fp[label], 1)
        for label in sorted(labels)
    ]
    mean_f1 = float(np.mean(scores)) if scores else 0.0
    print(f"Mean F1: {mean_f1:.4f}")
    return mean_f1
