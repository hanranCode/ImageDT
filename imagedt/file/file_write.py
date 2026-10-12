import csv
import json


def write_txt(data, path, separator=","):
    with open(path, "w", encoding="utf-8") as text_file:
        if isinstance(data, str):
            text_file.write(str(data))
        elif isinstance(data, list):
            for line in data:
                if isinstance(line, str):
                    text_file.write(str(line) + "\n")
                elif isinstance(line, list):
                    line = list(map(str, line))
                    split_str = separator + ""
                    line = split_str.join(line)
                    text_file.write(str(line) + "\n")
        else:
            text_file.write(json.dumps(data))


def write_csv(data, path, is_excel=False):
    with open(path, "w", encoding="utf-8", newline="") as csvfile:
        if is_excel:
            csvfile.write("\ufeff")
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)


def read_csv(path, delimiter=","):
    with open(path, encoding="utf-8-sig", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        data = [line for line in reader]
    return data


def readlines(file_path):
    with open(file_path, encoding="utf-8") as f:
        lines = [item.strip() for item in f.readlines()]
    return lines


def write_json(json_objects, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_objects, f)


def read_json(file_path):
    with open(file_path, encoding="utf-8") as f:
        items = json.load(f)
    return items


def decode_strs(strs):
    return strs.decode("utf-8") if isinstance(strs, bytes) else strs


def writeExcel(write_infos_mul, excel_path, sheetNames=None, set_color=False):
    import xlwt

    workbook = xlwt.Workbook()
    # style = xlwt.easyxf('align: wrap on')
    xlwt.easyxf("pattern: pattern solid, fore_colour red;")
    for ind, write_infos in enumerate(write_infos_mul):
        if sheetNames is None:
            sheetname = str(ind)
        else:
            sheetname = sheetNames[ind]
        sheetname = decode_strs(sheetname)
        writeSheet = workbook.add_sheet(sheetname)
        for row, item_infos in enumerate(write_infos):
            for idx, info in enumerate(item_infos):
                if isinstance(info, str):
                    info = decode_strs(info)
                writeSheet.write(row, idx, info)
    workbook.save(excel_path)
