import os
import uuid

import cv2

from ..dir.dir_loop import loop

DIR_TYPR = ["JPEGImages", "Annotations"]


class FilesOp:
    def __init__(self):
        super().__init__()

    def get_abs_path(self, file_dir, files_list):
        return [os.path.join(file_dir, f_item) for f_item in files_list]

    def check_dir_exist(self, path):
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            print(f"mkidr {path}")

    def check_file_exist(self, fpath):
        if not os.path.exists(fpath):
            raise OSError(f"{fpath} not exist")

    def fillter_dirname(self, file_paths):
        lines = list()
        for file_path in file_paths:
            imtes = file_path.strip().split(",")
            bsname = os.path.basename(imtes[0])
            lines.append([bsname, imtes[1]])
        return lines

    def fillter_file_type(self, file_paths):
        return [os.path.splitext(file_path)[0] for file_path in file_paths]

    def rename_with_uuid(self, fname, uid):
        dir_name = os.path.dirname(fname)
        bs_name = os.path.basename(fname)
        try:
            self.assert_in_op(".", bs_name)
        except ValueError as e:
            print(e)
            raise ValueError(f"file {bs_name} basename error ") from e
        return os.path.join(dir_name, uid + "." + bs_name.split(".")[-1])

    def rename_data_pairs(self, file_pair, uid):
        changes = [(path, self.rename_with_uuid(path, uid)) for path in file_pair]
        for path, destination in changes:
            self.check_file_exist(path)
            if path != destination and os.path.exists(destination):
                raise FileExistsError(destination)
        completed = []
        try:
            for path, destination in changes:
                os.rename(path, destination)
                completed.append((path, destination))
        except OSError:
            for path, destination in reversed(completed):
                os.rename(destination, path)
            raise

    def assert_not_in_op(self, member, container):
        if member in container:
            raise ValueError(f"Value {container} contains {member}")

    def assert_in_op(self, member, container):
        members = member if isinstance(member, list) else [member]
        if any(item not in container for item in members):
            raise ValueError(f"Value {container} didn't contain {member}")

    def assert_instance_op(self, inst, tar_inst=None):
        expected = list if tar_inst is None else type(tar_inst)
        if type(inst) is not expected:
            raise ValueError(f"input Values should be {expected.__name__}")

    def del_broken_image(self, root_dir):
        image_files = os.listdir(os.path.join(root_dir, DIR_TYPR[0]))
        for index, img in enumerate(image_files):
            img_name = os.path.join(root_dir, "JPEGImages", img)
            imat = cv2.imread(img_name)
            if imat is None:
                os.remove(img_name)
                print(f"remove broken image: {img}")
            print(f"check_data_pairs finished: {index + 1} / {len(image_files)}")

    def check_data_pairs(self, root_dir, anno_type=".xml"):
        # root dir should contain 'JPEGImages' and 'Annotations' dirs
        self.assert_in_op(DIR_TYPR, os.listdir(root_dir))

        images_files = self.fillter_file_type(
            os.listdir(os.path.join(root_dir, DIR_TYPR[0]))
        )
        allxml_files = self.fillter_file_type(
            os.listdir(os.path.join(root_dir, DIR_TYPR[1]))
        )

        image_names = set(images_files)
        num_images = len(images_files)
        num_xmlfil = len(allxml_files)
        if (
            num_images == num_xmlfil
            and len(set(images_files + allxml_files)) == num_images
        ):
            print(
                f"check_data_pairs finished: effective data pairs {len(images_files)} . "
            )
            return "have same files"

        for index, file_name in enumerate(allxml_files):
            if file_name not in image_names:
                print(f"annotation file {file_name} not in JPG")
                os.remove(os.path.join(root_dir, DIR_TYPR[1], file_name + anno_type))
            print(f"check_data_pairs finished: {index + 1} / {num_xmlfil}")

    def get_jpg_xml_pairs(self, root_dir, anno_type=".xml"):
        """return: image and xml pairs abs path [image_path, xml_path]......"""
        file_pairs = list()
        allxml_dir = os.path.join(root_dir, DIR_TYPR[1])
        images_dir = os.path.join(root_dir, DIR_TYPR[0])
        xml_files = self.get_abs_path(
            allxml_dir,
            sorted(name for name in os.listdir(allxml_dir) if name.endswith(anno_type)),
        )
        for xml_path in xml_files:
            imgae_bsname = os.path.splitext(os.path.basename(xml_path))[0] + ".jpg"
            image_path = os.path.join(images_dir, imgae_bsname)
            file_pairs.append([image_path, xml_path])
        return file_pairs

    def rename_file_with_uuid(self, root_dir):
        """
        root_dir contains: JPEGImages and Annotations
        """
        self.check_data_pairs(root_dir)
        file_pairs = self.get_jpg_xml_pairs(root_dir)
        for index, file_pair in enumerate(file_pairs):
            uid = str(uuid.uuid1())
            self.rename_data_pairs(file_pair, uid)
            print(f"finished: {index + 1} / {len(file_pairs)}.")

    def rename_single_file_uuid(self, root_dir):
        f_names = self.get_abs_path(root_dir, os.listdir(root_dir))
        for index, f_name in enumerate(f_names):
            f_uuid = str(uuid.uuid1())
            new_fpath = self.rename_with_uuid(f_name, f_uuid)
            os.rename(f_name, new_fpath)
            print(f"finished {index + 1}/{len(f_names)}")

    def loop_dir(self, root_dir):
        dep_files = list()
        dirs = self.get_abs_path(root_dir, os.listdir(root_dir))
        for item in dirs:
            if not os.path.isdir(item):
                continue
            dep_files += self.get_abs_path(item, os.listdir(item))
        return dep_files

    def rename_class_dir(self, root_dir, before_cls_str="other"):
        directories = {os.path.dirname(path) for path in loop(root_dir, [".jpg"])}
        for path in sorted(directories, key=lambda p: p.count(os.sep), reverse=True):
            basename = os.path.basename(path)
            if not basename.startswith(before_cls_str):
                destination = os.path.join(
                    os.path.dirname(path), before_cls_str + basename
                )
                if os.path.exists(destination):
                    raise FileExistsError(destination)
                os.rename(path, destination)


FilesOp = FilesOp()
