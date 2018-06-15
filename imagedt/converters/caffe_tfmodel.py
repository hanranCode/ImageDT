# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
from .taffe import KaffeError, print_stderr
from .taffe.tensorflow import TensorFlowTransformer

import shutil
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph


def fatal_error(msg):
    print_stderr(msg)
    exit(-1)


def assert_exist_files(*files):
    for check_file in files:
        # print("check_file: {0}".format(check_file))
        if not os.path.isfile(check_file):
            print("{0} not file".format(check_file))
            exit(-1)


def convert(prototxt_path, caffemodel_path, phase='test', output_model_name='freeze_graph_model.pb'):
    """
    'def_path', 'Model definition (.prototxt) path'
    'caffemodel', 'Model data (.caffemodel) path'
    'phase', default='test','The phase to convert: test (default) or train'
    'output_model_name', saved tf model name. default= 'standalonehybrid.pb'
    """
    assert_exist_files(prototxt_path, caffemodel_path)

    data_output_path = os.path.join(os.path.dirname(prototxt_path), 'output.mat')
    code_output_path = os.path.join(os.path.dirname(prototxt_path), 'output.py')
    standalone_output_path = os.path.join(os.path.dirname(prototxt_path), output_model_name)

    try:
        with tf.Session() as sess:
            transformer = TensorFlowTransformer(prototxt_path, caffemodel_path, phase=phase)
            print_stderr('Converting data...')
            if data_output_path is not None:
                data = transformer.transform_data()
                print_stderr('Saving data...')
                with open(data_output_path, 'wb') as data_out:
                    np.save(data_out, data)
            if code_output_path is not None:
                print_stderr('Saving source...')
                with open(code_output_path, 'wb') as src_out:
                    src_out.write(transformer.transform_source())

            if standalone_output_path:
                filename, _ = os.path.splitext(os.path.basename(standalone_output_path))
                temp_folder = os.path.join(os.path.dirname(standalone_output_path), 'tmp')

                if not os.path.exists(temp_folder):
                    os.makedirs(temp_folder)

                if data_output_path is None:
                    data = transformer.transform_data()
                    print_stderr('Saving data...')
                    data_output_path = os.path.join(temp_folder, filename) + '.npy'
                    with open(data_output_path, 'wb') as data_out:
                        np.save(data_out, data)

                if code_output_path is None:
                    print_stderr('Saving source...')
                    code_output_path = os.path.join(temp_folder, filename) + '.py'
                    with open(code_output_path, 'wb') as src_out:
                        src_out.write(transformer.transform_source())

                checkpoint_path = os.path.join(temp_folder, filename + '.ckpt')
                graph_name = os.path.basename(standalone_output_path)
                graph_folder = os.path.dirname(standalone_output_path)
                input_node = transformer.graph.nodes[0].name
                output_node = transformer.graph.nodes[-1].name
                tensor_shape = transformer.graph.get_node(input_node).output_shape
                tensor_shape_list = [tensor_shape.batch_size, tensor_shape.height, tensor_shape.width, tensor_shape.channels]

                sys.path.append(os.path.dirname(code_output_path))
                module = os.path.splitext(os.path.basename(code_output_path))[0]
                class_name = transformer.graph.name
                KaffeNet = getattr(__import__(module), class_name)

                data_placeholder = tf.placeholder(tf.float32, tensor_shape_list, name=input_node)
                net = KaffeNet({input_node: data_placeholder})

                # load weights stored in numpy format
                net.load(data_output_path, sess)

                print_stderr('Saving checkpoint...')

                saver = tf.train.Saver()
                saver.save(sess, checkpoint_path)

                print_stderr('Saving graph definition as protobuf...')
                tf.train.write_graph(sess.graph.as_graph_def(), graph_folder, graph_name, as_text=False)

                input_graph_path = standalone_output_path
                input_saver_def_path = ""
                input_binary = True
                input_checkpoint_path = checkpoint_path
                output_node_names = output_node
                restore_op_name = 'save/restore_all'
                filename_tensor_name = 'save/Const:0'
                output_graph_path = standalone_output_path
                clear_devices = True

                print_stderr('Saving standalone model...')
                freeze_graph(input_graph_path, input_saver_def_path,
                             input_binary, input_checkpoint_path,
                             output_node_names, restore_op_name,
                             filename_tensor_name, output_graph_path,
                             clear_devices, '')

                shutil.rmtree(temp_folder)

        print_stderr('Done.')
    except KaffeError as err:
        fatal_error('Error encountered: {}'.format(err))