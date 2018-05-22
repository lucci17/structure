from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import cv2

import tensorflow as tf

import deeplab_model
from utils import preprocessing
from utils import dataset_util

from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='./grass/VOCdevkit/VOC2012/JPEGImages',
                    help='The directory containing the image data.')

parser.add_argument('--output_dir', type=str, default='./grass/inference_output',
                    help='Path to the directory to generate the inference results')

parser.add_argument('--infer_data_list', type=str, default='./grass/sample_images_list.txt',
                    help='Path to the file listing the inferring images.')

parser.add_argument('--model_dir', type=str, default='./model',
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")

parser.add_argument('--base_architecture', type=str, default='resnet_v2_101',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

_NUM_CLASSES = 8

def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  pred_hooks = None
  if FLAGS.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    pred_hooks = [debug_hook]

  model = tf.estimator.Estimator(
      model_fn=deeplab_model.deeplabv3_model_fn,
      model_dir=FLAGS.model_dir,
      params={
          'output_stride': FLAGS.output_stride,
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
          'base_architecture': FLAGS.base_architecture,
          'pre_trained_model': None,
          'batch_norm_decay': None,
          'num_classes': _NUM_CLASSES,
      })
  path = "D:/deeplab/video/capture"
  cap = cv2.VideoCapture('C:/Users/bongos/Desktop/test1.avi')
  i = 0
  while(cap.isOpened()):
      ret, frame = cap.read()
      if ret == False:
          break
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
##      cv2.imshow('frame', frame)
      predictions = model.predict(
        input_fn=lambda: preprocessing.eval_input_fn(frame),
        hooks=pred_hooks)
      pre_dict = (list((predictions)))
      
      output_dir = FLAGS.output_dir
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)

      output_filename = "grass" + str(i) + '_mask.png'
      path_to_output = os.path.join(output_dir, output_filename)

      print("generating:", path_to_output)
      mask = pre_dict['decoded_labels']

##    print(i)
      s = path + "/grass" + str(i) +".jpg"
      cv2.imwrite(s,frame)
      i = i+1
  cap.release()
  cv2.destroyAllWindows()
##  examples = dataset_util.read_examples_list(FLAGS.infer_data_list)
##  image_files = [os.path.join(FLAGS.data_dir, filename) for filename in examples]

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)  



