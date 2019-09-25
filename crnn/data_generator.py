# coding:utf-8
import cv2
import re
import os
import PIL
import math
import numpy as np
from PIL import Image
from config import seed
#from captcha.image import ImageCaptcha


def get_img_label(label_path, images_path):
    """
    获取图像路径列表和图像标签列表
    :param label_path: 图像路径、标签存放文件对应的路径. [str]
    :param images_path: 图像路径. [str]
    :return:
    """
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.read()
    lines = lines.split('\n')
    img_path_list = []
    img_label_list = []
    for line in lines[:-1]:
        this_img_path, this_img_label = line.split('\t')
        this_img_path = os.path.join(images_path, this_img_path)
        img_path_list.append(this_img_path)
        img_label_list.append(this_img_label)
    return img_path_list, img_label_list


def get_charsets(dict=None, mode=1, charset_path=None):
    """
    生成字符集
    :param mode: 当mode=1时，则生成实时验证码进行训练，此时生成验证码的字符集存放在dict路径下的charsets.txt下，
                 当mode=2时，则采用真实场景的图像进行训练，此时会读取data文件夹下label.txt中所有的文本标签，
                 然后汇总去重得到所有的字符集
    :param dict: 字符集文件路径
    :param charset_path: 字符集文件存储路径，only use with mode=2
    :return:
    """
    if mode == 1:
        # with open(dict, 'r', encoding='utf-8') as f:
        #     lines = f.readlines()
        # charsets = ''.join(lines)
        pass
    else:
        with open(charset_path, 'r', encoding='utf-8') as fr:
            charsets = fr.read()
    charsets = re.sub('\n|\t|', '', charsets)
    charsets = list(set(list(charsets)))
    charsets = sorted(charsets)
    charsets = ''.join(charsets)
    charsets = charsets.encode('utf-8').decode('utf-8')
    return charsets


def gen_random_text(charsets, min_len, max_len):
    """
    生成长度在min_len到max_len的随机文本
    :param charsets: 字符集合. [str]
    :param min_len: 最小文本长度. [int]
    :param max_len: 最长文本长度. [int]
    :return:返回生成的文本编码序列和文本字符串
    """
    length = seed.random_integers(low=min_len, high=max_len)
    idxs = seed.randint(low=0, high=len(charsets), size=length)
    str = ''.join([charsets[i] for i in idxs])
    return idxs, str

def scence_batch_gen(batch_img_list, batch_img_label_list,
                     charsets, image_shape, max_len, blank_symbol):
    """
    生成一个batch真实场景数据集，每个batch包含三部分，分别是图像、每张图像的宽度、图像的标签
    :param batch_img_list: 图像路径列表
    :param batch_img_label_list: 图像标签列表
    :param charsets: 字符集字符串
    :param image_shape: 生成的图像尺寸
    :param max_len: 文本序列的最大长度
    :param blank_symbol: 当文本长度小于最大的长度时，对其尾部进行padding的数字
    :return:
    """
    batch_labels = []
    batch_image_widths = []
    batch_size = len(batch_img_label_list)
    batch_images = np.zeros(shape=(batch_size, image_shape[0], image_shape[1], image_shape[2]), dtype=np.float32)

    for i, path, label in zip(range(batch_size), batch_img_list, batch_img_label_list):
        # 对图像进行放缩
        image = Image.open(path)
        img_size = image.size
        height_ratio = image_shape[0] / img_size[1]
        if int(img_size[0] * height_ratio) > image_shape[1]:
            new_img_size = (image_shape[1], image_shape[0])
            image = image.resize(new_img_size, Image.ANTIALIAS).convert('RGB')
            image = np.array(image, np.float32)
            image = image / 255
            batch_images[i, :, :, :] = image
        else:
            new_img_size = (int(img_size[0] * height_ratio), image_shape[0])
            image = image.resize(new_img_size, Image.ANTIALIAS).convert('RGB')
            image = np.array(image, np.float32)
            image = image / 255
            batch_images[i, :image.shape[0], :image.shape[1], :] = image

        # 对标签进行编码
        if len(label) > max_len:
            label = label[:max_len]
        idxs = [charsets.index(i) for i in label]

        # 对标签进行padding
        pad_size = max_len - len(idxs)
        if pad_size > 0:
            idxs = np.pad(idxs, pad_width=(0, pad_size), mode='constant', constant_values=blank_symbol)

        batch_image_widths.append(image.shape[1])
        batch_labels.append(idxs)

    batch_labels = np.array(batch_labels, dtype=np.int32)
    batch_image_widths = np.array(batch_image_widths, dtype=np.int32)

    return batch_images, batch_image_widths, batch_labels


def load_images(batch_img_list, image_shape):
    """
    生成一个batch真实场景数据集，每个batch包含三部分，分别是图像、每张图像的宽度、图像的标签
    :param batch_img_list: 图像路径列表或图像列表[list]
    :param image_shape: 生成的图像尺寸
    :return:
    """
    # 参数为图像路径列表
    if isinstance(batch_img_list[0], str):
        batch_size = len(batch_img_list)
        batch_image_widths = []
        batch_images = np.zeros(shape=(batch_size, image_shape[0], image_shape[1], image_shape[2]), dtype=np.float32)

        for i, path in zip(range(batch_size), batch_img_list):
            # 对图像进行放缩
            image = Image.open(path)
            img_size = image.size
            height_ratio = image_shape[0] / img_size[1]
            if int(img_size[0] * height_ratio) > image_shape[1]:
                new_img_size = (image_shape[1], image_shape[0])
                image = image.resize(new_img_size, Image.ANTIALIAS).convert('RGB')
                image = np.array(image, np.float32)
                image = image / 255
                batch_images[i, :, :, :] = image
            else:
                new_img_size = (int(img_size[0] * height_ratio), image_shape[0])
                image = image.resize(new_img_size, Image.ANTIALIAS).convert('RGB')
                image = np.array(image, np.float32)
                image = image / 255
                batch_images[i, :image.shape[0], :image.shape[1], :] = image
            batch_image_widths.append(image.shape[1])
    # 参数为图像列表
    elif isinstance(batch_img_list[0], PIL.Image.Image):
        batch_size = len(batch_img_list)
        batch_image_widths = []
        batch_images = np.zeros(shape=(batch_size, image_shape[0], image_shape[1], image_shape[2]), dtype=np.float32)

        for i in range(batch_size):
            # 对图像进行放缩
            image = batch_img_list[i]
            img_size = image.size
            height_ratio = image_shape[0] / img_size[1]
            if int(img_size[0] * height_ratio) > image_shape[1]:
                new_img_size = (image_shape[1], image_shape[0])
                image = image.resize(new_img_size, Image.ANTIALIAS).convert('RGB')
                image = np.array(image, np.float32)
                image = image / 255
                batch_images[i, :, :, :] = image
            else:
                new_img_size = (int(img_size[0] * height_ratio), image_shape[0])
                image = image.resize(new_img_size, Image.ANTIALIAS).convert('RGB')
                image = np.array(image, np.float32)
                image = image / 255
                batch_images[i, :image.shape[0], :image.shape[1], :] = image
            batch_image_widths.append(image.shape[1])

    return batch_images, batch_image_widths