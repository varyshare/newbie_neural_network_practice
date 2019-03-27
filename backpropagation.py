# -*- coding: UTF-8 -*-
"""
@author 知乎：@Ai酱
"""
class NeuralNetwork:
    
    def __init__(self):
        self.LEARNING_RATE = 0.05 # 设置学习率
        # 初始化网络各层权重（权重的初试值也会影响神经网络是否收敛）
        # 博主试了下权重初始值都为0.2333是不行的
        self.w3 = -0.52133
        self.w2 = -0.233
        self.w1 = 0.2333
        
        self.data = [1, -1] # 输入数据
        self.label= [-1, 1]
    def L(self,w1,w2,w3,x,e_real):
        '''
        损失函数 return 1/2 * (e - e_real)^2
        '''
        return 0.5*(w1*w2*w3*x - e_real)**2
    
    def train(self):
        epoch = 160
        for _ in range(epoch):
            # 逐个样本进行训练模型
            for i in range(len(self.data)):
                x = self.data[i]
                e_real = self.label[i]
                
                self.y = self.w1 * x #计算第1层输出
                self.z = self.w2 * self.y # 计算第2层输出
                self.e = self.w3 * self.z # 计算第3层输出
                
                # 开始反向传播优化权重
                self.result3 = self.e - e_real
                self.w3 = self.w3 - self.LEARNING_RATE * self.result3 * self.z
                                
                self.result2 = self.result3 * self.w3
                self.w2 = self.w2 - self.LEARNING_RATE * self.result2 * self.y
                
                self.w1 = self.w1 - self.LEARNING_RATE * self.result2 * self.w2 * x
                self.check_gradient(x,e_real)
                
                
                
                
    def check_gradient(self,x,e_real):
        # 反向传播所求得的损失函数对各权重的导数
        dw3 = self.result3 * self.z
        dw2 = self.result2 * self.y
        dw1 = self.result2 * self.w2 * x
        
        # 使用定义求损失函数对各权重的导数
        epsilon = 10**-4 # epsilon为10的4次方
        # 求损失函数在w3处的左极限和右极限
        lim_dw3_right = self.L(self.w1, self.w2, self.w3+epsilon, x, e_real)
        lim_dw3_left = self.L(self.w1, self.w2, self.w3-epsilon, x, e_real)
        # 利用左右极限求导
        lim_dw3 = (lim_dw3_right - lim_dw3_left)/(2*epsilon)
        
        lim_dw2_right = self.L(self.w1, self.w2+epsilon, self.w3, x, e_real)
        lim_dw2_left = self.L(self.w1, self.w2-epsilon, self.w3, x, e_real)
        lim_dw2 = (lim_dw2_right - lim_dw2_left)/(2*epsilon)
        
        lim_dw1_right = self.L(self.w1+epsilon, self.w2, self.w3, x, e_real)
        lim_dw1_left = self.L(self.w1-epsilon, self.w2, self.w3, x, e_real)
        lim_dw1 = (lim_dw1_right - lim_dw1_left)/(2*epsilon)
        
        # 比对反向传播求的导数和用定义求的导数是否接近
        print("dl/dw3反向传播求得：%f,定义求得%f"%(dw3,lim_dw3))
        print("dl/dw2反向传播求得：%f,定义求得%f"%(dw2,lim_dw2))
        print("dl/dw1反向传播求得：%f,定义求得%f"%(dw1,lim_dw1))
                   
            
    def predict(self,x):
        self.y = self.w1 * x #计算第1层输出
        self.z = self.w2 * self.y # 计算第2层输出
        self.e = self.w3 * self.z # 计算第3层输出
        return 1 if self.e>0 else -1
        
        

            
nn = NeuralNetwork()
nn.train()
print(1,',',nn.predict(1))
print(-1,',',nn.predict(-1))
'''
输出:
1 , -1
-1 , 1
'''
            
        
        