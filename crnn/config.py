#coding:utf-8
import time
import numpy as np
 
# data
mode = 2  # mode=1则用验证码进行训练，mode=2则用真实场景进行训练
image_shape = [32, 1024, 1]  # 图像尺寸
# image_shape = [32, 1024, 3]  # 图像尺寸
seed = np.random.RandomState(int(round(time.time())))  # 生成模拟数据时的随机种子
min_len = 1  # 文本的最小长度
max_len = 50  # 文本的最大长度
fonts = ['./crnn/fonts/ch_font/STSONG.TTF']  # 生成模拟数据时的字体文件路径列表
train_images_path = r'./data/train_images'  # 训练集图像存放路径
train_label_path = r'./data/train_label.txt'  # 训练集标签存放路径
test_images_path = r'./data/test_images'  # 测试集图像存放路径
test_label_path = r'./data/test_label.txt'  # 测试集标签存放路径
dict = r'./dict/english.txt'
logs_path = r'./logs'  # 训练日志存放路径
models_path =r'./models/'  # 模型存放路径
 
# data icpr
org_images_path = r'./data/mtwi_2018_train/image_train'  # ICPR数据集原始图像路径
org_labels_path = r'./data/mtwi_2018_train/txt_train'  # ICPR数据集原始label路径
cut_train_images_path = r'./data/train_images'  # 训练集切图的保存路径
cut_train_labels_path = r'./data/train_label.txt'  # 训练集切图对应label的保存路径
cut_test_images_path = r'./data/test_images'  # 测试集切图的保存路径
cut_test_labels_path = r'./data/test_label.txt'  # 测试集切图对应label的保存路径
train_test_ratio = 0.9  # 训练测试集的比例
is_transform = True  # 是否进行仿射变换
angle_range = [-15.0, 15.0]  # 不进行仿射变换的倾斜角度范围
epsilon = 1e-4  # 原始图像的顺时针变换参数
filter_ratio = 1.3  # 图片过滤的高宽比例，高于该比例的图片将被过滤
filter_height = 16  # 高度过滤，切图后的图像高度低于该值的将被过滤掉，[int]
 
# data generate (with base images)
num_samples = 100  # 生成样本总量
base_img_dir = r'./images_base'  # 背景图文件夹路径
# font_style_path = {'ch': r'./fonts/ch_fonts', 'en': r'./fonts/en_fonts'}  # 字体文件夹路径
font_style_path = {'ch': r'./fonts', 'en': r'./fonts'}  # 字体文件夹路径
font_size = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40]  # 字体大小列表
# 字体颜色列表 ,black:0-3 gray:3-6 blue:6-12 green:12-15 brown:15-16 white:16-17
font_color = [[0, 0, 0], [36, 36, 36], [83, 72, 53], [109, 129, 139], [139, 139, 139], [143, 161, 143],
              [106, 160, 194], [97, 174, 238], [191, 234, 255], [118, 103, 221], [198, 120, 221], [64, 148, 216],
              [147, 178, 139], [76, 136, 107], [62, 144, 135], [209, 125, 72], [255, 255, 255]]
dictionary_file = './dict/en_ch.txt'  # 字典文件路径
text_size_limit = [1, 16]  # 生成文本字符范围
margin = 10  # 生成文本离背景图的边距最大值
use_blank = True  # 是否使用多线程，默认False
num_process = 1  # 并行处理数据的进程数，默认1（即单进程）
 
# charset generate
charset_path = r'./data/char_en_gen.txt'
 
# model
lstm_hidden = 128
 
# train
pool_size = 2 * 2  # pool层总共对图像宽度的缩小倍数
batch_size = 64  # batch_size
learning_rate = 1e-3  # 学习率
learning_decay_steps = 3000  # 学习率每多少次递减一次
learning_decay_rate = 0.95  # 学习率每次递减时，变为原来的多少
epoch = 20  # 迭代的次数
 
# predict
predict_batch_size = 64
predict_images_path = r'./data/test_images'
predict_label_path = r'./data/predict_label.txt'