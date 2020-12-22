# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import pickle

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
'''
normalize=True : 0~1사이값,False면 0~255사이값(0 : black, 255 : white)
one_hot_label=True : x_train, t_train이 one_hot_incording됨/default = flatten되어있음
x_train.shape : (60000, 784), t_train.shape : (60000, 10), x_test.shape : (10000, 784), t_test.shape: (10000, 10)
''' 

network = TwoLayerNet(input_size=784, hidden_size=500, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0] ## train_size = 60000
batch_size = 100   # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    ## random하게 train_size 60000개에서 batch_size 100개를 choice해 mask하는것?
    x_batch = x_train[batch_mask]
    ## x_batch.shape : [100,784]
    t_batch = t_train[batch_mask]
    ## t_batch.shape : [100,10]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        ## => gradient
    
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 1에폭당 정확도 계산 (600의 배수일 때 정확도 계산해서 print)
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc) ## 그래프 그리려고 list에 append
        test_acc_list.append(test_acc) ## 그래프 그리려고 list에 append
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# pickle 저장
f1 = open('my_pickle.pkl', 'wb')
pickle.dump(network, f1) ## network, weight, bias 모두 저장됨
f1.close()