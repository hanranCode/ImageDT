# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

from datasets import flowers
import tensorflow as tf

from tensorflow.contrib import slim

def show_some_images(tf_records_dir, data_type='train'):
    with tf.Graph().as_default(): 
        dataset = flowers.get_split(data_type, flowers_data_dir)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=32, common_queue_min=1)
        image, label = data_provider.get(['image', 'label'])
        
        with tf.Session() as sess:    
            with slim.queues.QueueRunners(sess):
                for i in range(4):
                    np_image, np_label = sess.run([image, label])
                    height, width, _ = np_image.shape
                    class_name = name = dataset.labels_to_names[np_label]

                    plt.figure()
                    plt.imshow(np_image)
                    plt.title('%s, %d x %d' % (name, height, width))
                    plt.axis('off')
                    plt.show()