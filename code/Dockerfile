FROM tensorflow/tensorflow:2.13.0-jupyter
MAINTAINER Jyt
WORKDIR /Users/jyt/docker/tensorflow/mnist
COPY . /Users/jyt/docker/tensorflow/mnist
#RUN pip install -r /shell/pytorch/requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
RUN pip install keras==2.13.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip install pandas==1.5.3 -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip install scikit-learn==1.2.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip install regex
CMD ["python", "model.py"]