from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import math
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import lfw

import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import models

def main(args):
  
    network = importlib.import_module(args.model_def)
    
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    # load training dataset
    train_set = facenet.get_dataset(args.data_dir)

    if args.filter_min_nrof_images_per_class > 0:
        train_set = clean_dataset(train_set, args.filter_min_nrof_images_per_class)
    nrof_classes = len(train_set)
    
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)
    # load lfw dataset for validation 
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
     
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        
        # Get a list of image paths and their labels
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)
        assert len(image_list)>0, 'The dataset should not be empty'
        
        # Create a queue that produces indices into the image_list and label_list 
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        with tf.device('/cpu:0'):

            index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                                 shuffle=True, seed=None, capacity=32)
            
            index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')
            
            learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
            
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
            
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')

            labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')
            
            input_queue = data_flow_ops.FIFOQueue(capacity=1000000,
                                        dtypes=[tf.string, tf.int64],
                                        shapes=[(1,), (1,)],
                                        shared_name=None, name=None)
            enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')
            
            nrof_preprocess_threads = args.nrof_preprocess_threads
            images_and_labels = []
            
            for _ in range(nrof_preprocess_threads):
                images_and_labels = distorted_inputs(images_and_labels,input_queue, args)
            image_batch, label_batch = tf.train.batch_join(
                images_and_labels, batch_size=args.batch_size, 
                enqueue_many=True,
                capacity=4 * nrof_preprocess_threads * args.batch_size,
                allow_smaller_final_batch=True)
            
            print(image_batch) 
            # perfetching queue not working properly yet

            batch_queue = slim.prefetch_queue.prefetch_queue(
                            [image_batch, label_batch], dynamic_pad=True, capacity=4)
            image_batch, label_batch = batch_queue.dequeue()
            
            image_batch = tf.identity(image_batch, 'image_batch')
            image_batch = tf.identity(image_batch, 'input')
            label_batch = tf.identity(label_batch, 'label_batch')
            
        print('Total number of classes: %d' % nrof_classes)
        print('Total number of examples: %d' % len(image_list))
        
        print('Building training graph')
        
        #///////////////////////////////////////////////////////////////////////////////////
        # Build the inference graph
        prelogits, _ = network.inference(image_batch, args.keep_probability, 
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size, 
            weight_decay=args.weight_decay, reuse=None)
    
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        print(embeddings)

        nrof_classes = len(train_set)
  
        weights = tf.get_variable('softmax_weights', shape=(args.embedding_size,nrof_classes), dtype=tf.float32,
            initializer=tf.variance_scaling_initializer(),
            regularizer=slim.l2_regularizer(args.weight_decay), trainable=True)

        weights = tf.nn.l2_normalize(weights, 0, name='norm_weights')
        
        if args.keep_probability < 1.0:
            scaled_prelogits = slim.dropout(scaled_prelogits, args.keep_probability, is_training=phase_train_placeholder,scope='Dropout')
        
        logits = facenet.combined_loss(embeddings, label_batch, nrof_classes, weights, scale_factor=args.l2_constrained_scale_factor, m1=args.m1, m2=args.m2)

        # Add norm loss
        if args.norm_loss_factor>0.0:
            norm_loss = args.norm_loss_factor*tf.reduce_mean(tf.pow(tf.norm(prelogits, axis=1)-args.l2_constrained_scale_factor, 2))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, norm_loss)
        
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        
        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, args.optimizer, 
            learning_rate, args.moving_average_decay, tf.trainable_variables(), args.num_gpus, args.log_histograms)
        #///////////////////////////////////////////////////////////////////////////////////
     
        # Create a saver
        if args.finetune:
            print("finetune model")
            all_vars = tf.trainable_variables()
            vars_to_restore = [v for v in all_vars if not v.name.startswith('Logits')]
        else:
            vars_to_restore = tf.trainable_variables()

        saver = tf.train.Saver(vars_to_restore, max_to_keep=40)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        
        # create corresponding model and log directories
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
        if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(log_dir)
        model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
        if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
            os.makedirs(model_dir)
        print('Model directory: %s' % model_dir)
        print('Log directory: %s' % log_dir)

        # Write arguments to a text file
        facenet.write_arguments_to_file(args, os.path.join(model_dir, 'arguments.txt'))
        

        #=============================================================================================================
        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement = True, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        save_graph_def(sess, model_dir)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver.restore(sess, pretrained_model)

            # Training and validation loop
            print('Running training')
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step= sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                    learning_rate_placeholder, phase_train_placeholder, global_step, 
                    total_loss, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file, learning_rate, log_dir)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                # Evaluate on LFW
                if args.lfw_dir:
                    evaluate(sess, lfw_paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds, log_dir, step, summary_writer)
                    
    return model_dir

def distorted_inputs(images_and_labels,input_queue, args):

    filenames, label = input_queue.dequeue()
    images = []
    for filename in tf.unstack(filenames):
        
        file_contents = tf.read_file(filename)
        image = tf.image.decode_image(file_contents)
        
        smallest_side = (int)(args.image_size * 1.10)
        image = aspect_preserving_resize(image, smallest_side)
        if args.random_rotate:
            image = tf.py_func(facenet.random_rotate_image, [image], tf.uint8)
        if args.random_crop:
            image = tf.random_crop(image, [args.image_size, args.image_size, 3])
        else:
            image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
        if args.random_flip:
            image = tf.image.random_flip_left_right(image)
        image.set_shape((args.image_size, args.image_size, 3))
        image = tf.image.per_image_standardization(image)
                        
        images.append(image)        

    images_and_labels.append([images, label]) 
    return images_and_labels



def inputs(input_queue, args):
    filenames, label = input_queue.dequeue()
    images = []
    for filename in tf.unstack(filenames):
        
        file_contents = tf.read_file(filename)
        image = tf.image.decode_image(file_contents)
        image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
        image.set_shape((args.image_size, args.image_size, 3))
        image = tf.image.per_image_standardization(image)
                        
        images.append(image)        

    images_and_labels.append([images, label])
    return images_and_labels


def _add_loss_summaries(total_loss, scope=None):
    """Add summaries for losses.
  
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses', scope)
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l)) 
    return loss_averages_op

def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold
  
def clean_dataset(dataset, min_nrof_images_per_class):
    removelist = []
    for i in range(len(dataset)):
        if len(dataset[i].image_paths)<min_nrof_images_per_class:
            removelist.append(i)
    ix = sorted(list(set(removelist)), reverse=True)
    for i in ix:
        del(dataset[i])
    return dataset

            
def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder, 
      learning_rate_placeholder, phase_train_placeholder, global_step, 
      loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file, learning_rate, log_dir):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch) 

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch),1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True}
        if (batch_number % 100 == 0):
            err, _, step, reg_loss, summary_str,updated_lr = sess.run([loss, train_op, global_step, regularization_losses, summary_op, learning_rate], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
            with open(os.path.join(log_dir,'training_log.txt'),'at') as f:
                f.write('Epoch: [%d][%d/%d]\tLoss %2.3f\tRegLoss %2.3f\tlr %.4f\n' %
                        (epoch, batch_number+1, args.epoch_size, err, np.sum(reg_loss),updated_lr))
        else:
            err, _, step, reg_loss,updated_lr = sess.run([loss, train_op, global_step, regularization_losses, learning_rate], feed_dict=feed_dict)
        duration = time.time() - start_time
        if (batch_number % 10 == 0):
            print('Epoch: [%d][%d/%d]\tTime %.3f\tImages/sec %d\tLoss %2.3f\tRegLoss %2.3f\tlr %.4f' %
                    (epoch, batch_number+1, args.epoch_size, duration, args.batch_size/duration,  err, np.sum(reg_loss),updated_lr))
        
        batch_number += 1
        train_time += duration 

    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step

def evaluate(sess, paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer):
    start_time = time.time()

    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    
    image_size = images_placeholder.get_shape()[1]
    embedding_size = embeddings.get_shape()[1]

    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    batch_size = batch_size
    nrof_images = len(paths)
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
    emb_array = np.zeros((nrof_images, embedding_size))
    for i in range(nrof_batches):
        start_index = i*batch_size
        end_index = min((i+1)*batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet.load_data(paths_batch, False, False, image_size)
        feed_dict = { images_placeholder:images, phase_train_placeholder:False }
        emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

    #assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    _, _, accuracy, val, val_std, far,f1 = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    
    print('Accuracy: %1.4f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    print('F1: %2.5f' % f1)
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='lfw/f1', simple_value=f1)
    #summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))

def save_graph_def(sess,model_dir):
    tf.train.write_graph(sess.graph.as_graph_def(),model_dir,'graph.pbtxt')


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)

def _smallest_size_at_least(height, width, smallest_side):
    """Computes new shape with the smallest side equal to `smallest_side`.

    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.

    Args:
        height: an int32 scalar tensor indicating the current height.
        width: an int32 scalar tensor indicating the current width.
        smallest_side: A python integer or scalar `Tensor` indicating the size 
        of
            the smallest side after resize.

    Returns:
        new_height: an int32 scalar tensor indicating the new height.
        new_width: and int32 scalar tensor indicating the new width.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                                    lambda: smallest_side / width,
                                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    return new_height, new_width


def aspect_preserving_resize(image, smallest_side):
    """Resize images preserving the original aspect ratio.

    Args:
        image: A 3-D image `Tensor`.
        smallest_side: A python integer or scalar `Tensor` indicating the size
        of the smallest side after resize.

    Returns:
        resized_image: A 3-D tensor containing the resized image.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width], 
            align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image


  

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.90)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='~/datasets/facescrub/fs_aligned:~/datasets/casia/casia-webface-aligned')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate', 
        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--decov_loss_factor', type=float,
        help='DeCov loss factor.', default=0.0)

    parser.add_argument('--l2_constrained_scale_factor', type=float,
        help='scale factor for "L2-constrained Softmax Loss for Discriminative Face verification" http://arxiv.org/abs/1703.09507.', 
        default=12)
    parser.add_argument('--m1', type=float,
        help='m1 factor for angular based algorithms', default=0.20)
    parser.add_argument('--m2', type=float,
        help='m2 factor for angular based algorithms', default=0.20)
    parser.add_argument('--norm_loss_factor', type=float,
        help='Feature normalizaiton loss factor.', default=0.0)

    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms', 
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
        help='Keep only the classes with this number of examples or more', default=0)
    parser.add_argument('--no_store_revision_info', 
        help='Disables storing of git revision info in revision_info.txt.', action='store_true')
    parser.add_argument('--num_gpus', type=int,
        help='Number of GPUs used for data parallism training', default=1)
    parser.add_argument('--finetune', 
        help='Finetune indicato, when set,the logits layer will not be restored ', action='store_true')

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=200)
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
