# -*- coding: utf-8 -*-
"""
框架化反向传播编程
@author: 知乎@Ai酱
"""

import random
class Layer(object):
    '''
    本文中，一层只有一个神经元，一个神经元只有一个输入一个输出
    '''
    def __init__(self,layer_index):
        '''
        layer_index: 第几层
        '''
        self.layer_index = layer_index
        # 初始化权重[0,1] - 0.5 = [-0.5,0.5]保证初始化有正有负
        self.w = random.random() - 0.5 
        # 当前层的输出
        self.output = 0
        
    def forward(self,input_var):
        '''
        前向传播：对输入进行运算，并将结果保存
        input_var: 当前层的输入
        '''
        self.input = input_var
        self.output = self.w * self.input
        
    
    def backward(self, public_value):
        '''
        反向传播：计算上层也会使用的导数值并保存
        假设当前层的计算规则是这样output = f(input)，
        而 input == 前一层的输出，
        因此，根据链式法则损失函数对上一层权重的导数 = 后面层传过来的公共导数* f'(input) * 前一层的导数
        也就是说，后面层传过来的公共导数值* f'(input) 是需要往前传的公用的导数值。
        由于本层中对输入做的运算为：output = f(input) = w*input
        所以, f'(input) = w.
        public_value: 后面传过来的公共导数值
        '''
        # 当前层要传给前面层的公共导数值 = 后面传过来的公共导数值 * f'(input)
        self.public_value = public_value * self.w
        # 损失函数对当前层参数w的导数 = 后面传过来的公共导数值 * f'(input) * doutput/dw
        self.w_grad = self.public_value * self.input
    
    def upate(self, learning_rate):
        '''
        利用梯度下降更新参数w
        参数迭代更新规则（梯度下降）： w = w - 学习率*损失函数对w的导数
        learning_rate: 学习率
        '''
        self.w = self.w - learning_rate * self.w_grad
    
    def display(self):
        print('layer',self.layer_index,'w:',self.w)

class Network(object):
    def __init__(self,layers_num):
        '''
        构造网络
        layers_num: 网络层数
        '''
        self.layers = []
        # 向网络添加层
        for i in range(layers_num):
            self.layers.append(Layer(i+1))#层编号从1开始
    
    def predict(self, sample):
        '''
        sample: 样本输入
        return 最后一层的输出
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return 1 if output>0 else -1
    
    def calc_gradient(self, label):
        '''
        从后往前计算损失函数对各层的导数
        '''
        # 计算最后一层的导数
        last_layer = self.layers[-1]
        # 由于损失函数=0.5*(last_layer.output - label)^2
        # 由于backward中的public_value = 输出对输入的导数
        # 对于损失函数其输入是last_layer.output，损失函数对该输入的导数=last_layer.output - label
        # 所以 最后一层的backward的public_value = last_layer.output - label
        last_layer.backward(last_layer.output - label)
        public_value = last_layer.public_value
        for layer in self.layers:
            layer.backward(public_value) # 计算损失函数对该层参数的导数
            public_value= layer.public_value
            
    def update_weights(self, learning_rate):
        '''
        更新各层权重
        '''
        for layer in self.layers:
            layer.upate(learning_rate)
        
    
    def train_one_sample(self, label, sample, learning_rate):
        self.predict(sample) # 前向传播，使得各层的输入都有值
        self.calc_gradient(label) # 计算各层导数
        self.update_weights(learning_rate) # 更新各层参数
        
    def train(self, labels, data_set, learning_rate, epoch):
        '''
        训练神经网络
        labels: 样本标签
        data_set: 输入样本们
        learning_rate: 学习率
        epoch: 同样的样本反复训练的次数
        '''
        for _ in range(epoch):# 同样数据反复训练epoch次保证权重收敛
            for i in range(len(labels)):#逐样本更新权重
                self.train_one_sample(labels[i], data_set[i], learning_rate)

nn = Network(3)
data_set = [1,-1]
labels   = [-1,1]
learning_rate = 0.05
epoch = 160
nn.train(labels,data_set,learning_rate,epoch)
print(nn.predict(1)) # 输出 -1
print(nn.predict(-1)) # 输出 1
