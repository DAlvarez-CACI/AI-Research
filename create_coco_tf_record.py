# Altered from  https://github.com/MetaPeak/tensorflow_object_detection_create_coco_tfrecord to write continously and change class lables

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pycocotools.coco import COCO
from PIL import Image
from random import shuffle
import os, sys
import numpy as np
import tensorflow as tf
import logging

from object_detection.utils import dataset_util

def dict_to_coco_example(img_data):
    """Convert python dictionary formath data of one image to tf.Example proto.
    Args:
        img_data: infomation of one image, inclue bounding box, labels of bounding box,\
            height, width, encoded pixel data.
    Returns:
        example: The converted tf.Example
    """
    bboxes = img_data['bboxes']
    xmin, xmax, ymin, ymax = [], [], [], []
    for bbox in bboxes:
        xmin.append(bbox[0])
        xmax.append(bbox[0] + bbox[2])
        ymin.append(bbox[1])
        ymax.append(bbox[1] + bbox[3])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(img_data['height']),
        'image/width': dataset_util.int64_feature(img_data['width']),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/label': dataset_util.int64_list_feature(img_data['labels']),
        'image/encoded': dataset_util.bytes_feature(img_data['pixel_data']),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf-8')),
    }))
    return example

def main(_):
    annotations_filepath = "/home/dean/COCODATASET/Raw/annotations/instances_train2017.json"
    imgs_dir = "/home/dean/COCODATASET/Raw/train2017"


    coco = COCO(annotations_filepath)
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()

    shuffle(img_ids)
    with tf.python_io.TFRecordWriter("/home/dean/COCODATASET/PLtrain.tfrecord") as tfrecord_writer:
        nb_imgs = len(img_ids)
        for index, img_id in enumerate(img_ids):
            if index % 100 == 0:
                print("Reading images: %d / %d "%(index, nb_imgs))
            img_info = {}
            bboxes = []
            labels = []

            img_detail = coco.loadImgs(img_id)[0]
            pic_height = img_detail['height']
            pic_width = img_detail['width']
            ann_ids = coco.getAnnIds(imgIds=img_id,catIds=cat_ids)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                bboxes_data = ann['bbox']
                bboxes_data = [bboxes_data[0]/float(pic_width), bboxes_data[1]/float(pic_height),bboxes_data[2]/float(pic_width), bboxes_data[3]/float(pic_height)]
                             # the format of coco bounding boxs is [Xmin, Ymin, width, height]
            bboxes.append(bboxes_data)
            labels.append(ann['category_id'])


            img_path = os.path.join(imgs_dir, img_detail['file_name'])
            img_bytes = tf.gfile.FastGFile(img_path,'rb').read()
            img_info['pixel_data'] = img_bytes
            img_info['height'] = pic_height
            img_info['width'] = pic_width
            img_info['bboxes'] = bboxes
            if len(labels) != 0:
                for label in labels:
                    if label == 45:
                        label = label + 1
                    label = label % 45 #class 1 = somethign
                    print(label)

            img_info['labels'] = labels


            example = dict_to_coco_example(img_info)
            tfrecord_writer.write(example.SerializeToString())

        tfrecord_writer.close()

    sys.exit()

if __name__ == "__main__":
    tf.app.run()
