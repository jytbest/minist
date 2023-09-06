import numpy as np
# 导入Pandas数据处理工具箱
import pandas as pd
# 从 Keras中导入 mnist数据集
from keras.datasets import mnist
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras import models,Input,Sequential
# 从 keras.layers 中导入神经网络需要的计算层
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow as tf
import pathlib
import random

BATCH_SIZE=64
def load_and_preprocess_from_path_label(path, label):
    image = tf.io.read_file(path)  # 读取图片
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.cast(image, dtype=tf.float32) / 255.0
    image = tf.image.resize(image, [28,28])
    image /= 255.0  # 归一化到[0,1]范围
    label = tf.cast(label, dtype=tf.int32)
    label = tf.one_hot(label, depth=10)
    return image, label
#加载数据集
def load_data() :
    data_path = pathlib.Path('./data/images')
    print(type(data_path))
    all_image_paths = list(data_path.glob('*/*'))
    #print(type(data_path.glob('*/*')))  # <class 'generator'>
    all_image_paths = [str(path) for path in all_image_paths]  # 所有图片路径的列表
    random.shuffle(all_image_paths)  # 打散
    image_count = len(all_image_paths)
    print('image_count: ', image_count)

    label_names = sorted(item for item in ["0","1","2","3","4","5","6","7","8","9"])
    #print('label_names: ', label_names)
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    #print('label_to_index: ', label_to_index)
    all_image_labels = [label_to_index[pathlib.Path(path).name.split("_")[2].replace(".jpg","")] for path in all_image_paths]
    db_train = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

    db_train.shuffle(len(all_image_labels))

    train = db_train.map(load_and_preprocess_from_path_label)

    # x_list = []
    # y_list = []
    # # 这里的step不能少，多亏了这里可以遍历train_db
    # for step, (x, y) in enumerate(db_train):
    #     x.numpy()  # tensor转为numpy
    #     y.numpy()
    #     x_list.append(x)  # 列表的使用方法，追加
    #     y_list.append(y)
    #
    # x_train = np.array(x_list)  # list转为numpy.ndarray格式。
    # y_train = np.array(y_list)

    #
    train.batch(BATCH_SIZE).repeat(2)

    val_batches = tf.data.experimental.cardinality(train)
    # iterator = train.make_one_shot_iterator()
    # train = iterator.get_next()
    # 0.2
    val_size = int(val_batches // 5)
    val_dataset = train.take(val_size)
    train_dataset = train.skip(val_size)
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #
    # # 导入keras.utils工具箱的类别转换工具
    # # 给标签增加维度,使其满足模型的需要
    # # 原始标签，比如训练集标签的维度信息是[60000, 28, 28, 1]
    # X_train = x_train.reshape(60000,28,28,1)
    # X_test = x_test.reshape(10000,28,28,1)
    #  # 特征转换为one-hot编码
    # y_train = to_categorical(y_train, 10)
    # y_test = to_categorical(y_test, 10)
    return train_dataset,val_dataset

def cnn_model() :
    # 创建
    # 从 keras 中导入模型
    # 构建一个最基础的连续的模型，所谓连续，就是一层接着一层
    model = models.Sequential()
    # 第一层为一个卷积，卷积核大小为(3,3), 输出通道32，使用 relu 作为激活函数
    model.add(Conv2D(32, (3, 3), padding="same",activation='relu',input_shape=(28, 28, 1)))
    # 第二层为一个最大池化层，池化核为（2,2)
    # 最大池化的作用，是取出池化核（2,2）范围内最大的像素点代表该区域
    # 可减少数据量，降低运算量。
    model.add(MaxPooling2D(pool_size=(2, 2),padding="same"))
    # 又经过一个（3,3）的卷积，输出通道变为64，也就是提取了64个特征。
    # 同样为 relu 激活函数
    model.add(Conv2D(64, (3, 3),padding="same", activation='relu'))
    # 上面通道数增大，运算量增大，此处再加一个最大池化，降低运算
    model.add(MaxPooling2D(pool_size=(2, 2),padding="same"))
    # dropout 随机设置一部分神经元的权值为零，在训练时用于防止过拟合
    # 这里设置25%的神经元权值为零
    model.add(Dropout(0.25))
    # 将结果展平成1维的向量
    model.add(Flatten())
    # 增加一个全连接层，用来进一步特征融合
    model.add(Dense(128, activation='relu'))
    # 再设置一个dropout层，将50%的神经元权值为零，防止过拟合
    # 由于一般的神经元处于关闭状态，这样也可以加速训练
    model.add(Dropout(0.5))
    # 最后添加一个全连接+softmax激活，输出10个分类，分别对应0-9 这10个数字
    model.add(Dense(10, activation='softmax'))
    return model

def cnn_model2():
    inputShape = (28, 28, 1)  # (60000, 28, 28, 1)，60000个样本
    # 激活函数：sigmoid，relu，tanh
    # 卷积层可以处理二维数据，矩阵
    # 全连接层只能处理一维数据，向量
    model = Sequential(
        [
            Input(shape=inputShape),
            Conv2D(32, kernel_size=(3, 3), activation='relu'),  # 卷积层：卷积函数：32 过滤器个数
            MaxPooling2D(pool_size=(2, 2)),  # 最大池化层
            Conv2D(32, kernel_size=(3, 3), activation='relu'),  # 卷积层：卷积核尺寸，一般是3*3，或者5*5
            MaxPooling2D(pool_size=(2, 2)),  # 最大池化层
            Flatten(),  # 将输入层的数据压成一维数据，
            Dropout(0.5),  # 深度学习网络训练过程中，按照一定的概率丢弃神经网络单元，防止过拟合，默认0.5，丢弃50%的神经元
            # Softmax 函数能够将一个K维实值向量归一化，所以它主要被用于多分类任务
            # Sigmoid 能够将一个实数归一化，因此它一般用于二分类任务。
            # 当 Softmax 的维数 K=2 时，Softmax 会退化为 Sigmoid 函数
            Dense(10, activation="softmax")  # Dense代表全连接网络，输出维度10，激活函数softmax
        ]
    )
    return model

def train():
    x_train,y_train = load_data()
    for x,y in x_train:
        print(x.shape)

    model = cnn_model2()

    # 训练
    # 编译上述构建好的神经网络模型
    # 指定优化器为 rmsprop
    # 制定损失函数为交叉熵损失
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # 创建一个权重文件保存文件夹logs
    log_dir = "logFiles/"
    # 记录所有训练过程，每隔一定步数记录最大值
    tensorboard = TensorBoard(log_dir=log_dir)
    # Checkpoint是模型的权重
    checkpoint = ModelCheckpoint("./modelFiles/mnist_model.hdf5", monitor='val_loss', verbose=1,
                                 save_best_only=False, mode='min')
    # 防止过拟合
    early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    # 加了一个tensorboard
    callbacks_list = [tensorboard, checkpoint, early]
    model.summary()

    model.fit(x_train,validation_data=y_train,
              epochs=5,  # 训练轮次为5轮
              batch_size=128,callbacks=callbacks_list)  # 以128为批量进行训练
    model.save('./modelFiles/mnist_model.h5', save_format='tf')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()
    #load_data()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
