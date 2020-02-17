# -*- coding: utf-8 -*-
from environment.lookup_tables import ActionLabelsRed
import tensorflow as tf
from functools import reduce
import os
DATA_FORMAT = "channels_first"


class CChessModel(object):
    def __init__(self, config):
        self.config = config
        self.n_labels = len(ActionLabelsRed)  # 所有的走法，共计2086
        # 车、马、炮、象、士、将、兵，共7种棋子，每个玩家7个特征平面，一共14个特征平面。棋盘大小为10行9列
        self.inputs = tf.placeholder(tf.float32, (None, 14, 10, 9), name="chessboard")
        self.policy_labels = tf.placeholder(tf.float32, (None, self.n_labels), name="policy_labels")
        self.value_labels = tf.placeholder(tf.float32, (None,), name="value_labels")
        self.weights = tf.placeholder(tf.float32, (None,), name="weights")
        self.training = tf.placeholder(tf.bool, (), "training")  # 现在是在训练还是测试。用于BN
        self.policy, self.value = None, None  # 输出
        self.xentropy_loss, self.value_loss, self.L2_loss, self.total_loss = None, None, None, None
        self.journalist, self.summury_op = None, None
        self.opt = None
        self.global_step = tf.get_variable("global_step", dtype=tf.int32, initializer=0, trainable=False)
        self.lr = tf.get_variable("lr", dtype=tf.float32, initializer=self.config.Train.init_lr, trainable=False)
        self.build_network()
        self.build_train_loop()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=100)

    def build_train_loop(self):
        self.build_loss_and_optimizer()
        self.add_summary()

    def add_summary(self):
        self.journalist = tf.summary.FileWriter(self.config.Train.log_dir, flush_secs=10)
        tf.summary.scalar("policy_loss", self.xentropy_loss)
        tf.summary.scalar("value_loss", self.value_loss)
        tf.summary.scalar("total_loss", self.total_loss)
        self.summury_op = tf.summary.merge_all()

    def train_one_step(self, data, policy_labels, value_labels, weights):
        xentropy_loss, value_loss, total_loss, _, sum_res = self.sess.run(
            [self.xentropy_loss, self.value_loss, self.total_loss, self.opt, self.summury_op],
            feed_dict={self.inputs: data, self.policy_labels: policy_labels,
                       self.value_labels: value_labels, self.training: True, self.weights: weights})
        step = self.sess.run(self.global_step)
        self.journalist.add_summary(sum_res, step)
        if step % self.config.Train.step_ckpt == 0:
            self.saver.save(
                self.sess, save_path=os.path.join(self.config.Train.ckpt_path, "chineseChess"), global_step=step)
        if step % self.config.Train.lr_step == 0:
            self.sess.run(tf.assign(self.lr, self.sess.run(self.lr)*self.config.Train.lr_decay))

        return step, xentropy_loss, value_loss, total_loss

    def build_network(self):
        f = tf.layers.conv2d(self.inputs, 256, 5, padding="SAME", data_format=DATA_FORMAT, use_bias=False, name="conv1")
        f = tf.layers.batch_normalization(f, axis=1, training=self.training, name="BN1")
        f = tf.nn.relu(f)
        for i in range(2, 9):  # 7个残差层
            f = self._build_residual_block(f, i)

        with tf.variable_scope("policy"):
            policy = tf.layers.conv2d(f, 8, 1, padding="SAME", data_format=DATA_FORMAT, use_bias=False, name="conv1")
            policy = tf.layers.batch_normalization(policy, axis=1, training=self.training, name="BN1")
            policy = tf.nn.relu(policy)
            last_dim = reduce(lambda x, y: x * y, policy.get_shape().as_list()[1:])
            policy = tf.reshape(policy, (-1, last_dim))
            self.policy = tf.layers.dense(policy, self.n_labels, activation=None, name="fc")

        with tf.variable_scope("value"):
            value = tf.layers.conv2d(f, 4, 1, padding="SAME", data_format=DATA_FORMAT, use_bias=False, name="conv1")
            value = tf.layers.batch_normalization(value, axis=1, training=self.training, name="BN1")
            value = tf.nn.relu(value)
            last_dim = reduce(lambda x, y: x * y, value.get_shape().as_list()[1:])
            value = tf.reshape(value, (-1, last_dim))
            value = tf.layers.dense(value, 256, activation=tf.nn.relu, name="fc1")
            self.value = tf.squeeze(tf.layers.dense(value, 1, activation=tf.nn.tanh, name="fc2"), axis=[1])

    def build_loss_and_optimizer(self):
        xentropy_loss = tf.reduce_sum(tf.multiply(self.policy_labels, tf.nn.log_softmax(self.policy)), axis=1)
        loss1 = tf.negative(tf.reduce_mean(xentropy_loss*self.weights))
        self.xentropy_loss = tf.negative(tf.reduce_mean(xentropy_loss))
        value_loss = tf.squared_difference(self.value, self.value_labels)
        loss2 = tf.reduce_mean(value_loss*self.weights)
        self.value_loss = tf.reduce_mean(value_loss)
        self.L2_loss = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name and "BN" not in v.name])
        # self.total_loss = 0.0001 * self.L2_loss + self.value_loss + self.xentropy_loss
        self.total_loss = 0.0001 * self.L2_loss + loss1 + loss2
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss, self.global_step)

    def _build_residual_block(self, f, index):
        ff = f
        f = tf.layers.conv2d(f, 256, 3, padding="SAME", data_format=DATA_FORMAT, use_bias=False,
                             name="conv" + str(index) + '_1')
        f = tf.layers.batch_normalization(f, axis=1, training=self.training, name="BN" + str(index) + '_1')
        f = tf.nn.relu(f)
        f = tf.layers.conv2d(f, 256, 3, padding="SAME", data_format=DATA_FORMAT, use_bias=False,
                             name="conv" + str(index) + '_2')
        f = tf.layers.batch_normalization(f, axis=1, training=self.training, name="BN" + str(index) + '_2')
        f = tf.nn.relu(tf.add(f, ff))
        return f

    def save_ckpt(self):
        step = self.sess.run(self.global_step)
        self.saver.save(
            self.sess, save_path=os.path.join(self.config.Train.ckpt_path, "chineseChess"), global_step=step)
