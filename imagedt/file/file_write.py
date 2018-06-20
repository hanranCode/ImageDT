# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import json


def write_txt(data, path):
    with open(path, "w") as text_file:
        if isinstance(data, (str, unicode)):
            text_file.write(str(data))
        elif isinstance(data, list):
            for line in data:
                if isinstance(line, (str, unicode)):
                    text_file.write(str(line) + '\n')
        else:
            text_file.write(json.dumps(data))
