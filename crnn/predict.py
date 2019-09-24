#coding:utf-8
import os
import tensorflow as tf
from modules import CRNN
from multiprocessing import Pool
import config as crnn_config
from data_generator import load_images
 
crnn_graph = tf.Graph()
with crnn_graph.as_default():
    crnn = CRNN(image_shape=crnn_config.image_shape,
                min_len=crnn_config.min_len,
                max_len=crnn_config.max_len,
                lstm_hidden=crnn_config.lstm_hidden,
                pool_size=crnn_config.pool_size,
                learning_decay_rate=crnn_config.learning_decay_rate,
                learning_rate=crnn_config.learning_rate,
                learning_decay_steps=crnn_config.learning_decay_steps,
                mode=crnn_config.mode,
                dict=crnn_config.dict,
                is_training=False,
                train_label_path=crnn_config.train_label_path,
                train_images_path=crnn_config.train_images_path,
                charset_path=crnn_config.charset_path)
 
crnn_sess = tf.Session(graph=crnn_graph)
with crnn_sess.as_default():
    with crnn_graph.as_default():
        tf.global_variables_initializer().run()
        crnn_saver = tf.train.Saver(tf.global_variables())
        crnn_ckpt = tf.train.get_checkpoint_state(crnn_config.models_path)
        crnn_saver.restore(crnn_sess, crnn_ckpt.model_checkpoint_path)

        #crnn_saver = tf.train.import_meta_graph('D:/python_project/crnn/models/models-3.meta')
        #crnn_saver.restore(crnn_sess, tf.train.latest_checkpoint("D:/python_project/crnn/models/"))
 
 
def predict(images, batch_size=crnn_config.predict_batch_size):
    """
    predict images
    :param images:images path or list of images ,[list/str]
    :param batch_size: batch size
    :return:
    """
    if isinstance(images, str):
        assert os.path.exists(images), 'path of image or images dir is not exist'
        if os.path.isdir(images):
            test_img_list = os.listdir(images)
            batch_size = len(test_img_list) if len(test_img_list) <= batch_size else batch_size
            test_img_list = [os.path.join(images, i) for i in test_img_list]
            batch_images, batch_image_widths = load_images(
                test_img_list,
                crnn.image_shape
            )
        elif os.path.isfile(images):
            test_img_list = [images]
            batch_size = len(test_img_list) if len(test_img_list) <= batch_size else batch_size
            batch_images, batch_image_widths = load_images(
                test_img_list,
                crnn.image_shape
            )
 
    elif isinstance(images, list):
        assert len(images) > 0, '图片数量不可以为0'
        batch_size = len(images) if len(images) <= batch_size else batch_size
        batch_images, batch_image_widths = load_images(
            images,
            crnn.image_shape
        )
    # 启用多线程
    predict_label_list = list()
    for i in range(0, len(batch_images), batch_size):
        if i + batch_size >= len(batch_images):
            batch_size = len(batch_images) - i
        predict_label_list.append(crnn_sess.run(crnn.dense_predicts,
                                                feed_dict={crnn.images: batch_images[i:i + batch_size],
                                                           crnn.image_widths: batch_image_widths[i:i + batch_size]}))
    result = list()
    for predict_label in predict_label_list:
        for j in range(len(predict_label)):
            text_i = ''.join([crnn.charsets[v] for v in predict_label[j] if v != -1])
            # if text_i.replace(' ', '') != '':
            #     result.append(text_i)
            result.append(text_i)
    return test_img_list, result

    #         dp[]


import numpy as np


def edit_distance(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            temp = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + temp, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]


def simility(word1, word2):
    edis = edit_distance(word1, word2)
    maxLen = max(len(word1), len(word2))
    # return 1-res*1.0/maxLen
    return edis, maxLen

def printresult():
    _,ret = predict(crnn_config.predict_images_path, crnn_config.predict_batch_size)
    print(ret)


def writetofile():
    image_name, result = predict(crnn_config.predict_images_path, crnn_config.predict_batch_size)
    testlabelfile = open(crnn_config.test_label_path, 'r', encoding='utf-8')
    testlabel = []
    for i in testlabelfile.readlines():
        # testlabel.append(i)
        testlabel.append(i.split('\t')[1].strip())
    totl = len(testlabel)
    if len(testlabel) == len(result):
        print("same len")
    with open(crnn_config.predict_label_path, 'w', encoding='utf-8') as retfile:
        sum = 0
        i = 0
        edis_sum = 0
        maxlen_sum = 0
        total_len = 0
        for (name, pre) in zip(image_name, result):
            if pre == testlabel[i]:
                sum += 1
                total_len += len(pre)
            else:
                print(name, '\t', "predict:", pre, '\t', '\t', "true:", testlabel[i])
                edis, maxlen = simility(pre, testlabel[i])
                edis_sum += edis
                maxlen_sum += maxlen
                total_len += maxlen
            retfile.write(
                name.replace(crnn_config.predict_images_path + "/", '') + '\t' + pre + '\t' + testlabel[i] + "\n")
            i += 1
        print("complete match: %d/%d" % (sum, totl))
        print("in %d not complete match senquence" % (totl - sum), " acc:", 1 - edis_sum * 1.0 / maxlen_sum)
        print("total acc:", 1 - edis_sum * 1.0 / total_len)


if __name__ == '__main__':
    # 可以传入本地图片文件夹路径、本地图片路径、ndarray图片列表
    printresult()
    # writetofile()

