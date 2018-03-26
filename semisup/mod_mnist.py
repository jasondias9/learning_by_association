#! /usr/bin/env python


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import semisup
import sys
import numpy as np

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from sklearn.model_selection import train_test_split

FLAGS = flags.FLAGS

flags.DEFINE_integer('sup_per_class', 10,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_seed', -1,
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_per_batch', 10,
                     'Number of labeled samples per class per batch.')

flags.DEFINE_integer('unsup_batch_size', 10,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 1,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 5000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 1.0, 'Weight for visit loss.')

flags.DEFINE_integer('max_steps', 100, 'Number of training steps.')

flags.DEFINE_string('logdir', '/tmp/mod_mnist', 'Training log path.')

NUM_LABELS = 10
IMAGE_SHAPE = [64, 64, 1]

import logging


def main(_):

  print('[INFO]    reading training images...')
  train_images = np.loadtxt("train_x.csv", delimiter=",")
  train_images = np.array([[1 if x > 250 else 0 for x in y] for y in train_images])
  train_images = train_images.reshape((train_images.shape[0], 64, 64, 1))
  # print('[INFO]    reading unsupervised...')
  # unsup = np.loadtxt("test_x_redu.csv", delimiter=",")
  # unsup = np.array([[1 if x > 250 else 0 for x in y] for y in unsup])
  # unsup = unsup.reshape((unsup.shape[0], 64, 64, 1))
  print('[INFO]    reading labels...')
  train_labels = np.loadtxt("train_y.csv", delimiter=",", dtype = 'uint8')

  print('[INFO]    test/train split...')
  train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size = 0.2)

  # Sample labeled training subset.
  seed = FLAGS.sup_seed if FLAGS.sup_seed != -1 else None
  sup_by_label = semisup.sample_by_label(train_images, train_labels,
                                         FLAGS.sup_per_class, NUM_LABELS, seed)

  graph = tf.Graph()
  with graph.as_default():
    model = semisup.SemisupModel(semisup.architectures.mnist_model, NUM_LABELS,
                                 IMAGE_SHAPE)

    # Set up inputs.
    t_unsup_images, _ = semisup.create_input(train_images, train_labels,
                                             FLAGS.unsup_batch_size)
    t_sup_images, t_sup_labels = semisup.create_per_class_inputs(
        sup_by_label, FLAGS.sup_per_batch)

    # Compute embeddings and logits.
    t_sup_emb = model.image_to_embedding(t_sup_images)
    t_unsup_emb = model.image_to_embedding(t_unsup_images)
    t_sup_logit = model.embedding_to_logit(t_sup_emb)

    # Add losses.
    model.add_semisup_loss(
        t_sup_emb, t_unsup_emb, t_sup_labels, visit_weight=FLAGS.visit_weight)
    model.add_logit_loss(t_sup_logit, t_sup_labels)

    t_learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,
        model.step,
        FLAGS.decay_steps,
        FLAGS.decay_factor,
        staircase=True)
    train_op = model.create_train_op(t_learning_rate)
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph)

    saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in xrange(FLAGS.max_steps):
      _, summaries = sess.run([train_op, summary_op])
      if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
        print('Step: %d' % step)
        test_pred = model.classify(test_images).argmax(-1)
        conf_mtx = semisup.confusion_matrix(test_labels, test_pred, NUM_LABELS)
        test_err = (test_labels != test_pred).mean() * 100
        print(conf_mtx)
        print('Test error: %.2f %%' % test_err)
        print()

        test_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag='Test Err', simple_value=test_err)])

        summary_writer.add_summary(summaries, step)
        summary_writer.add_summary(test_summary, step)

        saver.save(sess, FLAGS.logdir, model.step)

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  app.run()
