
import pandas as pd
import  numpy as np
import yaml
from keras.layers import Input,Masking,Bidirectional,GRU,Embedding,Dense,TimeDistributed,concatenate,Conv1D,Reshape,Conv2D,MaxPool2D,GlobalMaxPool2D
from keras.models import Model,model_from_yaml,load_model
from keras import models,Input,Sequential,preprocessing
# from keras.utils import np_utils
import argparse
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    # parser.add_argument('--output', help='the dir to save logs and models')
    parser.add_argument('--data_predict_path', help='dataset path')
    parser.add_argument('--resume', help='the checkpoint file to resume from')
    # parser.add_argument('--step',
    #     type=int,
    #     default=150,
    #     help='number training step')
    args = parser.parse_args()
    return args


def prepare(image,label):
    image /= 255.0  # 归一化到[0,1]范围
    label = tf.cast(label, dtype=tf.int32)
    label = tf.one_hot(label, depth=10)
    return image,label

from keras import models
def predict():
    args = parse_args()
    dataset = preprocessing.image_dataset_from_directory(
        args.data_path+'/1.jpg', image_size=(28, 28))
    dataset = dataset.map(prepare)

    # 打开已经训练好的文件夹，
    with open('./modelFiles/mnist_model.h5', 'r') as m:
        # 加载训练好的模型
        yaml_string = yaml.load(m)
    model = model_from_yaml(yaml_string)
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    y_predict_pro = model.predict(dataset)
    lab = np.argsort(y_predict_pro.numpy())
    print(lab)
    print("本次预测的数字是: ", lab[0][-1])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    predict()