
import pandas as pd
import  numpy as np
import yaml
from keras.layers import Input,Masking,Bidirectional,GRU,Embedding,Dense,TimeDistributed,concatenate,Conv1D,Reshape,Conv2D,MaxPool2D,GlobalMaxPool2D
from keras.models import Model,model_from_yaml,load_model
# from keras.utils import np_utils

from keras import models
def predict():
    test_data='predict/example_6.jpg'
    # 打开已经训练好的文件夹，
    with open('./modelFiles_experiment/ganrao/mnist_model.h5', 'r') as m:
        # 加载训练好的模型
        yaml_string = yaml.load(m)
    model = model_from_yaml(yaml_string)
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    y_predict_pro = model.predict(test_data)
    lab = np.argsort(y_predict_pro.numpy())
    print(lab)
    print("本次预测的数字是: ", lab[0][-1])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    predict()