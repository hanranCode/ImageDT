"""Compatibility exports, loaded on first use."""

from imagedt._lazy import resolve as _resolve

_EXPORTS = {
    "TFmodel_Wrapper": (".tfmodel_wrapper", "TFmodel_Wrapper"),
    "int64_list_feature": (".dataset_util", "int64_list_feature"),
    "bytes_list_feature": (".dataset_util", "bytes_list_feature"),
    "float_list_feature": (".dataset_util", "float_list_feature"),
    "int64_feature": (".dataset_util", "int64_feature"),
    "bytes_feature": (".dataset_util", "bytes_feature"),
    "float_feature": (".dataset_util", "float_feature"),
    "read_examples_list": (".dataset_util", "read_examples_list"),
    "recursive_parse_xml_to_dict": (".dataset_util", "recursive_parse_xml_to_dict"),
    "make_initializable_iterator": (".dataset_util", "make_initializable_iterator"),
    "read_dataset": (".dataset_util", "read_dataset"),
    "DataSet": (".DataInterface", "DataSet"),
    "DataSets": (".DataInterface", "DataSets"),
    "dynamically_loaded_data": (".data_provider", "dynamically_loaded_data"),
    "resize_image_keep_aspect": (".data_provider", "resize_image_keep_aspect"),
    "tf_noise_padd": (".data_provider", "tf_noise_padd"),
    "tf_JitterCut": (".data_provider", "tf_JitterCut"),
    "Data_Provider": (".data_provider", "Data_Provider"),
    "ImageReader": (".Records", "ImageReader"),
    "Record_Writer": (".Records", "Record_Writer"),
    "RecordWriter": (".Records", "RecordWriter"),
    "write_pbmodel_summery": (".Records", "write_pbmodel_summery"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(name, _EXPORTS, globals(), __name__)


def __dir__():
    return sorted(set(globals()) | set(__all__))
