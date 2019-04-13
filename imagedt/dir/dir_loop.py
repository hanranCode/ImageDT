# coding: utf-8
import os
import re


class LoopFile:
    def __init__(self, root_dir, file_extend=[], short_exclude=[], long_exclude=[]):
        self.root_dir = root_dir
        self.short_exclude = short_exclude
        self.long_exclude = long_exclude
        self.file_extend = file_extend
    
    def __del__(self):
        pass
    
    def start(self, func):
        self.func = func
        return self.loop_file(self.root_dir)
    
    def loop_file(self, root_dir):
        t_sum = []
        sub_gen = os.listdir(root_dir)
        for sub in sub_gen:
            is_exclude = False
            for extends in self.short_exclude:  ##在不检查文件、目录范围中
                if extends in sub:              ##包含特定内容
                    is_exclude = True
                    break
                if re.search(extends, sub):     ##匹配指定正则
                    is_exclude = True
                    break                    
            if is_exclude:
                continue
            abs_path = os.path.join(root_dir, sub)
            is_exclude = False
            for exclude in self.long_exclude:
                if exclude == abs_path[-len(exclude):]:
                    is_exclude = True
                    break
            if is_exclude:
                continue
            if os.path.isdir(abs_path):
                t_sum.extend(self.loop_file(abs_path))
            elif os.path.isfile(abs_path):
                if len(self.file_extend) > 0:
                    if not "." + abs_path.rsplit(".", 1)[1] in self.file_extend:  ##不在后缀名 检查范围中
                        continue
                t_sum.append(self.func(abs_path))
        return t_sum


def loop(root, extensions=['.jpg', '.png']):
    lf = LoopFile(root, extensions)
    return lf.start(lambda f: f)


def get_dir_images(root='./', extensions=['.jpg', '.png']):
    lf = LoopFile(root, extensions)
    return len(lf.start(lambda f: f))