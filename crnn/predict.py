# coding:utf-8
import tensorflow as tf
from modules import CRNN
import config as crnn_config
import base64
import io
from PIL import Image
import time
import numpy as np

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
def cmp(s1, s2):
    i1 = int(s1[:s1.find(".")])
    i2 = int(s2[:s2.find(".")])

    if i1 < i2:
        return -1
    elif i1 == i2:
        return 0
    return 1

def perpare_image(image):
    '''
    :param image: iamge is image stram
    :param batch_size: 1 # just one to one
    :return:
    '''
    image = base64.b64decode(image)
    image = io.BytesIO(image)
    image = Image.open(image)
    crnn_imageshape = crnn.image_shape
    img_size = image.size
    batch_images = np.zeros(shape=(1, crnn_imageshape[0], crnn_imageshape[1], crnn_imageshape[2]), dtype=np.float32)
    height_ratio = crnn_imageshape[0] / img_size[1]
    if int(img_size[0] * height_ratio) > crnn_imageshape[1]:
        new_img_size = (crnn_imageshape[1], crnn_imageshape[0])
        image = image.resize(new_img_size, Image.ANTIALIAS).convert('RGB')
        image = np.array(image, np.float32)
        image = image / 255
        batch_images[0,:,:,:] = image
    else:
        new_img_size = (int(img_size[0] * height_ratio), crnn_imageshape[0])
        image = image.resize(new_img_size, Image.ANTIALIAS).convert('RGB')
        image = np.array(image, np.float32)
        image = image / 255
        batch_images[0, :image.shape[0], :image.shape[1], :] = image
    return batch_images,image.shape[1]


def flask_predict(image):
    '''
    :param image:image code by base 64
    :return: predict result
    '''
    batch_images,image_width = perpare_image(image)
    pred_rest = crnn_sess.run(crnn.dense_predicts,feed_dict={crnn.images: batch_images,crnn.image_widths: [image_width]})
    for i in range(len(pred_rest)):
        text = ''.join([crnn.charsets[j] for j in pred_rest[i] if j != -1])
    return text

def get_predict_report(image,strs=None):

    begin_time = time.time()
    perpare = flask_predict(image)
    if(strs is not None):
        edis = edit_distance(perpare, strs)
        edis_sum = edis/len(strs)
        rest_acc = 1-edis_sum
        result = perpare
        ture_str = strs
    else:
        rest_acc = str("not calculate accuracy")
        ture_str = ""
        result = perpare
    end_time = time.time()
    crnn_time = end_time-begin_time
    print(result,ture_str,str(rest_acc),crnn_time)
    return  result,ture_str,str(rest_acc),crnn_time

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

