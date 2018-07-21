"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import time
import lfw
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

def main(args):
  
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
            
            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

            print('%s pairs'%len(paths))

            # Load the model
            facenet.load_model(args.model_dir)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            if '.pb' in args.model_dir:
                image_size = args.image_size
                embedding_size = args.embedding_size
            else:
                image_size = images_placeholder.get_shape()[1]
                embedding_size = embeddings.get_shape()[1]
        
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on LFW images')
            batch_size = args.lfw_batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            print("%s batches"%nrof_batches)
            emb_array = np.zeros((nrof_images, embedding_size))
            emb_array_flip = np.zeros((nrof_images, embedding_size))

            t_total = 0
            for i in range(nrof_batches):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
#feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                feed_dict = { images_placeholder:images }
                images_flip = facenet.load_data(paths_batch, False, True, image_size)
                feed_dict_flip = {images_placeholder:images_flip}

                t_start = time.time()
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                if args.mirrorface:
                    emb_array_flip[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict_flip)

                t = 1000*(time.time()-t_start) / batch_size
#print(t)
                if i>= 0:
                    t_total += t
            print("---inference speed = %s milliseconds---"%(t_total/(nrof_batches-1)))
            
            if args.mirrorface:
                emb_sum = np.add(emb_array, emb_array_flip)
                emb_norm = np.linalg.norm(emb_sum, ord=2, axis=1)

                for i in list(xrange(len(emb_norm))):
                    emb_array[i] = np.divide(emb_sum[i], emb_norm[i])
            
            
            tpr, fpr, accuracy, val, val_std, far, f1= lfw.evaluate(emb_array, 
                actual_issame, nrof_folds=args.lfw_nrof_folds, distance_metric=args.distance_metric)
            '''
            for i in xrange(len(tpr)):
                print('%1.5f %1.5f'%(1-tpr[i], fpr[i]))
            '''
            print('Accuracy: %1.4f+-%1.4f' % (np.mean(accuracy), np.std(accuracy)))

            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            print('F1: %2.5f' % f1)

            auc = metrics.auc(fpr, tpr)
            print('Area Under Curve (AUC): %1.3f' % auc)
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            print('Equal Error Rate (EER): %1.3f' % eer)
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model_dir', type=str, 
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='jpg', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--distance_metric', type=int,
        help='0 to use L2 distance, 1 to use cosine similarity', default=0)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.9)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=128)
    parser.add_argument('--embedding_size', type=int,
        help='embedding size to represent the feature dimension', default=256)
    parser.add_argument('--mirrorface', 
        help='Flag set to use mirrorface trick', action='store_true')




    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
