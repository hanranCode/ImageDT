# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import csv
import json


def write_txt(data, path, separator=','):
    with open(path, "w") as text_file:
        if isinstance(data, (str, unicode)):
            text_file.write(str(data))
        elif isinstance(data, list):
            for line in data:
                if isinstance(line, (str, unicode)):
                    text_file.write(str(line) + '\n')
                elif isinstance(line, list):
                    line = map(str, line)
                    split_str = separator+''
                    line = split_str.join(line)
                    text_file.write(str(line) + '\n')
        else:
            text_file.write(json.dumps(data))


def write_csv(data, path, is_excel=False):
  with file(path, 'wb') as csvfile:
    if is_excel:
      csvfile.write(u"\ufeff")
    writer = csv.writer(csvfile)
    for row in data:
      writer.writerow(row)


def read_csv(path, delimiter=','):
    with file(path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        data = [line for line in reader]
    return data


def readlines(file_path):
    with open(file_path, 'r') as f:
        lines = [item.strip() for item in f.readlines()]
    return lines


def write_json(json_objects, file_path):
    with open(file_path, 'w') as f:
        json.dump(json_objects, f)


def read_json(file_path):
    with open(file_path) as f:
        items = json.load(f)
    return items