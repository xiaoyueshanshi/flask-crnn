#coding:utf-8
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.rnn import BasicLSTMCell
from data_generator import get_charsets, captcha_batch_gen, scence_batch_gen, get_img_label


def read_tfrecode(recode_data_path,batch_size,epoch):
    tf_read = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(recode_data_path,epoch)
    _,example = tf_read.read(filename_queue)
    feature = tf.parse_single_example(example,features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'image_width': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.string)
    })
    image = tf.decode_raw(feature['image_raw'],tf.int8)
    label = tf.decode_raw(feature['image_width'],tf.int64)
    image_width = feature['image_width']
    capacity = epoch + 3 * batch_size
    images, label,image_width = tf.train.batch([image, label,image_width], batch_size, capacity, num_thread=4)

    return

#img,lab = tf.train.shuffle_batch([image,label], batch_size=3,capacity=32,min_after_dequeue=10)
if __name__ == '__main__':
    batch_size = 64
    tfcode_path = r"E:\A_1_logos\tf-recode\test_recode.tfrecodes"
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        images, label, image_width = read_tfrecode(batch_size=64,epoch=5)
        try:
            while not coord.should_stop():
                images, label, image_width = sess.run([images, label, image_width])
                print(label)
                print(image_width)
        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()
        coord.request_stop()
        coord.join(threads)




class CRNN(object):
    def __init__(self,
                 image_shape,
                 min_len,
                 max_len,
                 lstm_hidden,
                 pool_size,
                 learning_decay_rate,
                 learning_rate,
                 learning_decay_steps,
                 mode,
                 dict,
                 is_training,
                 train_label_path,
                 train_images_path,
                 charset_path):
        self.min_len = min_len
        self.max_len = max_len
        self.lstm_hidden = lstm_hidden
        self.pool_size = pool_size
        self.learning_decay_rate = learning_decay_rate
        self.learning_rate = learning_rate
        self.learning_decay_steps = learning_decay_steps
        self.mode = mode
        self.dict = dict
        self.is_training = is_training
        self.train_label_path = train_label_path
        self.train_images_path = train_images_path
        self.charset_path = charset_path
        self.charsets = get_charsets(self.dict, self.mode, self.charset_path)
        self.image_shape = image_shape
        self.images = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.image_shape[0], self.image_shape[1], self.image_shape[2]])
        self.image_widths = tf.placeholder(dtype=tf.int32, shape=[None])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len])
        self.seq_len_inputs = tf.divide(self.image_widths, self.pool_size, name='seq_len_input_op') - 1
        self.train_op, self.loss_ctc = self.create_train_op(self.logprob)
        self.dense_predicts = self.decode_predict(self.logprob)
        self.logprob = self.forward(self.is_training)

    def vgg_net(self, inputs, is_training, scope='vgg'):
        batch_norm_params = {
            'is_training': is_training
        }
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
                with slim.arg_scope([slim.max_pool2d], padding='SAME'):
                    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                        net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
                        net = slim.max_pool2d(net, [2, 2], scope='pool1')
                        net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
                        net = slim.max_pool2d(net, [2, 2], scope='pool2')
                        net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
                        net = slim.max_pool2d(net, [2, 2], stride=[2, 1], scope='pool3')
                        net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
                        net = slim.max_pool2d(net, [2, 2], stride=[2, 1], scope='pool4')
                        net = slim.repeat(net, 1, slim.conv2d, 512, [3,3], scope='conv5')
                        net = tf.layers.conv2d(net, 512, [2, 2], padding='valid', activation='relu')
                        return net
 
    def forward(self, is_training):
        dropout_keep_prob = 0.7 if is_training else 1.0
        cnn_net = self.vgg_net(self.images, is_training)
 
        with tf.variable_scope('Reshaping_cnn'):
            shape = cnn_net.get_shape().as_list()  # [batch, height, width, features]
            transposed = tf.transpose(cnn_net, perm=[0, 2, 1, 3],
                                      name='transposed')  # [batch, width, height, features]
            conv_reshaped = tf.reshape(transposed, [-1, shape[2], shape[1] * shape[3]],
                                       name='reshaped')  # [batch, width, height x features]
 
        list_n_hidden = [self.lstm_hidden, self.lstm_hidden]
 
        with tf.name_scope('deep_bidirectional_lstm'):
            # Forward direction cells
            fw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]
            # Backward direction cells
            bw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]
 
            lstm_net, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell_list,
                                                                            bw_cell_list,
                                                                            conv_reshaped,
                                                                            dtype=tf.float32
                                                                            )
            # Dropout layer
            lstm_net = tf.nn.dropout(lstm_net, keep_prob=dropout_keep_prob)
 
        with tf.variable_scope('fully_connected'):
            shape = lstm_net.get_shape().as_list()  # [batch, width, 2*n_hidden]
            fc_out = slim.layers.linear(lstm_net, len(self.charsets) + 1)  # [batch x width, n_class]
 
            lstm_out = tf.reshape(fc_out, [-1, shape[1], len(self.charsets) + 1],
                                  name='lstm_out')  # [batch, width, n_classes]
 
            # Swap batch and time axis
            logprob = tf.transpose(lstm_out, [1, 0, 2], name='transpose_time_major')  # [width(time), batch, n_classes]
 
        return logprob
 
    def create_loss(self, logprob):
        sparse_code_target = self.dense_to_sparse(self.labels, blank_symbol=len(self.charsets) + 1)
        with tf.control_dependencies(
                [tf.less_equal(sparse_code_target.dense_shape[1],
                               tf.reduce_max(tf.cast(self.seq_len_inputs, tf.int64)))]):
            loss_ctc = tf.nn.ctc_loss(labels=sparse_code_target,
                                      inputs=logprob,
                                      sequence_length=tf.cast(self.seq_len_inputs, tf.int32),
                                      preprocess_collapse_repeated=False,
                                      ctc_merge_repeated=True,
                                      ignore_longer_outputs_than_inputs=True,
                                      # returns zero gradient in case it happens -> ema loss = NaN
                                      time_major=True)
            loss_ctc = tf.reduce_mean(loss_ctc)
        return loss_ctc
 
    def create_train_op(self, logprob):
        loss_ctc = self.create_loss(logprob)
        tf.losses.add_loss(loss_ctc)
 
        self.global_step = tf.train.get_or_create_global_step()
 
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                   self.learning_decay_steps, self.learning_decay_rate,
                                                   staircase=True)
 
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
 
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
 
        train_op = slim.learning.create_train_op(total_loss=tf.losses.get_total_loss(), optimizer=optimizer,
                                                 update_ops=update_ops)
        return train_op, loss_ctc
 
    def decode_predict(self, logprob):
        with tf.name_scope('decode_conversion'):
            sparse_code_pred, log_probability = tf.nn.ctc_greedy_decoder(logprob,
                                                                         sequence_length=tf.cast(
                                                                             self.seq_len_inputs,
                                                                             tf.int32
                                                                         ))
            sparse_code_pred = sparse_code_pred[0]
            dense_predicts = tf.sparse_to_dense(sparse_code_pred.indices,
                                                sparse_code_pred.dense_shape,
                                                sparse_code_pred.values, default_value=-1)
 
        return dense_predicts
 
    def dense_to_sparse(self, dense_tensor, blank_symbol):
        """
        将标签转化为稀疏表示
        :param dense_tensor: 原始的密集标签
        :param blank_symbol: padding的符号
        :return:
        """
        indices = tf.where(tf.not_equal(dense_tensor, blank_symbol))
        values = tf.gather_nd(dense_tensor, indices)
        sparse_target = tf.SparseTensor(indices, values, [-1, self.image_shape[1]])
        return sparse_target

    def train(self,
              epoch=100,
              batch_size=32,
              train_images_path=None,
              train_label_path=None,
              restore=False,
              fonts=None,
              logs_path=None,
              models_path=None,
              ):
        # 创建相关目录
        if not os.path.exists(models_path):
            os.mkdir(models_path)
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)
 
        # summary
        tf.summary.scalar('loss_ctc', self.loss_ctc)
        tf.summary.scalar('train_ops',self.train_op)
        # tf.
        merged = tf.summary.merge_all()
 
        # sess and writer
        sess = tf.Session()
        writer = tf.summary.FileWriter(logs_path, sess.graph)
        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())
 
        # restore model
        last_epoch = 0
        if restore:
            ckpt = tf.train.latest_checkpoint(models_path)
            if ckpt:
                last_epoch = int(ckpt.split('-')[1]) + 1
                saver.restore(sess, ckpt)
 
        # 计算batch的数量
        if self.mode == 1:
            batch_nums = 1000

        else:
            train_img_list, train_label_list = get_img_label(train_label_path, train_images_path)
            batch_nums = int(np.ceil(len(train_img_list) / batch_size))
 
        if self.mode == 1:
            for i in range(last_epoch, epoch):
                for j in range(batch_nums):
                    batch_images, batch_image_widths, batch_labels = captcha_batch_gen(
                        batch_size,
                        self.charsets,
                        self.min_len,
                        self.max_len,
                        self.image_shape,
                        len(self.charsets) + 1,
                        fonts
                    )
                    _, loss, predict_label = sess.run(
                        [self.train_op, self.loss_ctc, self.dense_predicts],
                        feed_dict={self.images: batch_images,
                                   self.image_widths: batch_image_widths,
                                   self.labels: batch_labels}
                    )
                    if j % 1 == 0:
                        print('epoch:%d/%d, batch:%d/%d, loss:%.4f, truth:%s, predict:%s' % (
                            i, epoch,
                            j, batch_nums,
                            loss,
                            ''.join([self.charsets[k] for k in batch_labels[0] if k != (len(self.charsets) + 1)]),
                            ''.join([self.charsets[v] for v in predict_label[0] if v != -1])
                        ))



                #tf_recode



                saver.save(sess, save_path=models_path, global_step=i)
                summary = sess.run(merged,
                                   feed_dict={
                                       self.images: batch_images,
                                       self.image_widths: batch_image_widths,
                                       self.labels: batch_labels
                                   })
                writer.add_summary(summary, global_step=i)
        else:
            for i in range(last_epoch, epoch):
                random_index = random.sample(range(len(train_img_list)), len(train_img_list))
                batch_index = np.array_split(np.array(random_index), batch_nums)
                for j in range(batch_nums):
                    this_batch_index = list(batch_index[j])
                    this_train_img_list = [train_img_list[index] for index in this_batch_index]
                    this_train_label_list = [train_label_list[index] for index in this_batch_index]
                    batch_images, batch_image_widths, batch_labels = scence_batch_gen(
                        this_train_img_list,
                        this_train_label_list,
                        self.charsets,
                        self.image_shape,
                        self.max_len,
                        len(self.charsets) + 1
                    )
                    _, loss, predict_label = sess.run(
                        [self.train_op, self.loss_ctc, self.dense_predicts],
                        feed_dict={self.images: batch_images,
                                   self.image_widths: batch_image_widths,
                                   self.labels: batch_labels}
                    )
                    if j % 1 == 0:
                        print('epoch:%d/%d, batch:%d/%d, loss:%.4f, truth:%s, predict:%s' % (
                            i, epoch,
                            j, batch_nums,
                            loss,
                            ''.join([self.charsets[i] for i in batch_labels[0] if i != (len(self.charsets) + 1)]),
                            ''.join([self.charsets[v] for v in predict_label[0] if v != -1])
                        ))
 
                saver.save(sess, save_path=models_path, global_step=i)
                summary = sess.run(merged,
                                   feed_dict={
                                       self.images: batch_images,
                                       self.image_widths: batch_image_widths,
                                       self.labels: batch_labels
                                   })
                writer.add_summary(summary, global_step=i)