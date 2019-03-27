## 实例介绍反向传播，为何说深度学习离不开反向传播？

我们专栏之前介绍了[单个神经元](https://zhuanlan.zhihu.com/p/59678480)如何利用随机梯度下降自己调节权重。深度学习指的是数以百千层的层数很深的神经网络，每层又有数以百千个神经元。那么深度学习是否也可以使用这种形式的梯度下降来进行调节权重呢？答：很难。为什么？主要原因是太深了。为何“深”用梯度下降解会有问题呢？主要是因为链式法则导致计算损失函数对前面层权重的导数时，损失函数对后面层权重的导数总是被重复计算，反向传播就是将那些计算的值保存减少重复计算。不明白？那这篇文章就看对了。接下来将解释这个重复计算过程。**反向传播就是梯度下降中的求导环节，它从后往前计算导数重复利用计算过的导数而已。**。梯度下降不懂或者神经网络不懂请先阅读这个文章[单个神经元+随机梯度下降学习逻辑与规则](https://zhuanlan.zhihu.com/p/59678480)。

文章结构：

> 1. 为何要使用反向传播？
> 2. 反向传播优化神经网络权重参数实践。

### 为何要使用反向传播？

我们用一个最简单的三层神经网络（无激活函数）来进行解释为何需要使用反向传播，所使用的三层神经网络如下所示：

![1553584597483](https://pic2.zhimg.com/v2-f99702fa21cb27b336d3b3a837190559_b.jpg)

我们神经网络是可以用公式表示。对于整个网络而言，$x​$是输入，$e​$是输出。对于神经网络的各层而言，第一层神经网络的输入是$x​$ 输出是$y​$。那么其用公式表示就是： $y=w1*x​$ .第二层神经网络用公式表示就是：$z = w2 * y​$ 。第三层用公式表示是：$e = w3 * z​$ 。假设$\hat e​$ 是真实值，那么损失函数$L​$可以写成这样：$L(w1,w2,w3)=\frac{1}2 (e-\hat e )^2​$ 。`这个1/2 只是为了抵消二次方求导所产生的2。其实可以不写的。`  

这里我们有三个要优化的参数：$w1,w2,w3$ 。

**我们先用传统的梯度下降方法来看看哪些地方重复计算了，而反向传播就是将这些重复计算的地方计算结果保存，并且向前面层传播这个值以减少运算复杂度。** 梯度下降不懂或者神经网络不懂请先阅读这个文章[单个神经元+随机梯度下降学习逻辑与规则](https://zhuanlan.zhihu.com/p/59678480)。梯度下降需要先得求损失函数对$w1,w2,w3$ 的导数，然后根据导数来不断迭代更新$w1,w2,w3 $ 的值。

1. 第1层：求损失函数$L(w1,w2,w3)​$对$w1​$ 的导数（更规范的讲是偏导数）。

   我们先看看损失函数$L(w1,w2, w3)$和$w1$之间的关系是什么。

   $L(w1,w2,w3)=\frac{1}2 (e-\hat e )^2 $

   $e = w3 * z​$

   $ z= w2*y​$ 

   $ y = w1 *x ​$ 

   所以这是个复合函数的求导。根据高中学习到的复合函数求导法则（也是大家可能经常听到的链式法则），复合函数求导等于各级复合函数导数的乘积，也就是这样 $ \frac{dL} {dw1} =\frac{dL} {de} * \frac{de} {dz} * \frac{dz} {dy} * \frac {dy} {dw1}​$ 。

2. 第2层：求损失函数$L(w1,w2, w3 )$对$w2$ 的导数。

   我们再看看损失函数$L(w1,w2, w3)​$和$w2​$之间的关系是什么。

   $L(w1,w2, w3)=\frac{1}2 (e-\hat e )^2 ​$

   $e = w3 * z$

   $ z= w2*y​$ 

   根据复合函数求导可得： $ \frac{dL} {dw2} = \frac{dL} {de} *  \frac{de} {dz} * \frac{dz} {dw2}​$ 。

3. 第3层：求损失函数$L(w1,w2, w3 )$对$w3$ 的导数。

   我们再看看损失函数$L(w1,w2, w3)​$和$w3​$之间的关系是什么。

   $L(w1,w2, w3)=\frac{1}2 (e-\hat e )^2 ​$

   $e = w3 * z$

   根据复合函数求导可得： $ \frac{dL} {dw3} = \frac{dL} {de} *  \frac{de} {dw3}$ 。

我们将这三层的损失函数对相应层权重导数列在一起看看哪儿重复计算了：

- 第1层：$ \frac{dL} {dw1} =\frac{dL} {de} * \frac{de} {dz} * \frac{dz} {dy} * \frac {dy} {dw1}​$
- 第2层：$ \frac{dL} {dw2} = \frac{dL} {de} *  \frac{de} {dz} * \frac{dz} {dw2}​$ 
- 第3层： $ \frac{dL} {dw3} = \frac{dL} {de} *  \frac{de} {dw3}$

我们会发现，最前面的那层即第1层使用的$\frac{dL} {de}​$ 和$\frac{de} {dz}​$ 已经在后面的两层中计算过了。并且每层都重复计算了$\frac{dL} {de}​$ 。为了更清晰的突出哪些是重复计算的，我将那些重复计算部分提取出来。

- 第1层：$ \frac{dL} {de} * \frac{de} {dz} * \frac{dz} {dy} ​$
- 第2层：$\frac{dL} {de} *  \frac{de} {dz} ​$ 
- 第3层： $ \frac{dL} {de} ​$

那么，我们是不是把这些公共部分的计算从最后一层开始计算导数，并把结果往前面传？可以计算完了第3层的 $result3 = \frac{dL} {de} $，就把它传到第2层。计算完了第2层的$result2 = \frac{dL} {de} *  \frac{de} {dz} =result3*  \frac{de} {dz}$ ，就将结果传到第1层。然后第一层就可以这样写：$result1 =\frac{dL} {de} * \frac{de} {dz} * \frac{dz} {dy}=result2*  \frac{dz} {dy}$ 。这样就可以不重复计算那些已经算出来的导数。

## 反向传播过程理解

前面我们提到了可以从后面往前面计算，将公共部分的导数往前传。这只是解决了求导问题，那怎么进行参数更新呢？答：参数更新跟梯度下降完全一样，都是这个公式$w_i = w_i - \alpha \frac{dL} {dw_i} $。反向传播就是梯度下降中的求导环节，它重复利用计算过的导数而已。

我们看看反向传播版的使用梯度下降进行参数更新是怎样的。

损失函数$L$ 是这样： $L(w1,w2, w3)=\frac{1}2 (e-\hat e )^2 $

其他的几层函数如下所示：

- 第3层：$e = w3 * z​$

- 第2层：$ z= w2*y​$ 

- 第1层：$ y = w1 *x ​$ 

在这篇文章[“单个神经元+随机梯度下降学习逻辑与规则”](https://zhuanlan.zhihu.com/p/59678480) 介绍了，权重更新是一个不断猜（迭代更新）的过程。`下一次权重值 = 本次权重值 - 学习率*损失函数对该权重的导数`。定义学习率为$\alpha$. 接下来我们只需要知道怎么利用后面层数传递过来的值来求“损失函数对当前层权重$w_i​$的导数”即可。

则各层网络更新权重的步骤为如下所示：

1. 更新第3层权重参数$w3$

   计算损失函数对第3层权重的导数$\frac{dL} {dw3}​$ ，用公式表示是这样， $ \frac{dL} {dw3} = \frac{dL} {de} *  \frac{de} {dw3}​$ 。由于$ \frac{dL} {de} ​$ 还会被前面几层用到，我们用$result3​$保存它。即令$result3 = \frac{dL} {de} = (e-\hat e)​$ 。所以 $ \frac{dL} {dw3} = result3 * \frac{de} {dw3} = result3 *  z​$ 。

   - 所以,权重$w3​$的迭代更新表达式为：$w3 = w3 - \alpha * \frac{dL} {dw3} = w3 - result3 * z​$。

   - 然后**将$result3 == \frac{dL} {de}   $，传递给上一层(即第2层）。省得它们又重新计算一次，这就是反向传播（backpropagation)。** 这也是大家常说的BP神经网络。BP是backpropagation的简写。

2. 更新第2层权重参数$w2​$

   利用后一层（即第3层）传递过来的值$result3 = \frac{dL} {de}   ​$来计算损失函数对$w2​$的导数$ \frac{dL} {dw2} = \frac{dL} {de} *  \frac{de} {dz} * \frac{dz} {dw2}  ​$ 。直接使用了$result3​$这样就不用再计算$\frac{dL} {de}​$ 这个导数。由于前一层(第1层)会使用$\frac{dL} {de} *  \frac{de} {dz}​$ ，于是乎我们将这个公共导数保存。令$result2 = \frac{dL} {de} *  \frac{de} {dz} = result3 * \frac{de} {dz} ​$ 。

   - 那么，$ \frac{dL} {dw2} = result2 * \frac{dz} {dw2}  = result2 * y​$ 。权重$w2​$的迭代更新表达式为：$w2 = w2 - \alpha * \frac{dL} {dw2} = w2 - result2 * y​$。
   - 将$result2 ==  \frac{dL} {de} *  \frac{de} {dz} ​$传递给前一层（第1层）。

3. 更新第1层权重参数$w1 ​$

   到了第一层就不用再向前传播导数了。损失函数对第一层权重$w1$的导数为$\frac{dL} {dw1} =\frac{dL} {de} * \frac{de} {dz} * \frac{dz} {dy} * \frac {dy} {dw1} = result2   * \frac{dz} {dy} * \frac {dy} {dw1}  = result2 * w2 * x$ 。

   - 权重$w1​$的迭代更新表达式为：$w1 = w1 - \alpha * \frac{dL} {dw1} = w1 - result2 * w2 *x​$。
   - 由于已经是最前面这层，所以不用向前面传播。

所以，将上面3步用伪代码写可以写成这样。

> while(同样数据反复训练){
>
> ​	while(逐个取样本训练){
>
> ​	$ y = w1 *x $ 
>
> ​	$ z= w2*y$ 
>
> ​	$e = w3 * z​$
>
> ​	$result3 = \frac{dL} {de} = (e-\hat e)$
>
> ​	$w3 = w3 - \alpha * \frac{dL} {dw3} = w3 - \alpha *result3 * z$
>
> ​	$result2  = result3 * \frac{de} {dz} = result3 * w3 $
>
> ​	$w2  = w2 - \alpha * result2 * y​$
>
> ​	$w1 = w1 - \alpha *result2 * w2 *x$
>
> }
>
> }

## 反向传播实践

我们将上面的伪代码转成Python代码。我们希望这个神经网络能自己学习到的功能是输入$x$输出的$e=x^2$。

我们提供训练集数据：

> 输入数据x	数据标签$\hat {e} = -x​$
> ​	1		-1
> ​	-1		1

重复训练次数epoch = 500。

好开工实现它。

```python
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
```



## 如何检验反向传播是否写对？

### 手动推导，人工判断

前面提到了反向传播本质是梯度下降，那么关键在于导数必须对。我们现在网络比较小可以直接手动计算导数比对代码中对各权重导数是否求对。

比如上面代码中三个参数的导数将代码中的`result*`展开表示就是：

```python
dw3 = (self.e - e_real) * self.z 
    = (self.e - e_real) * self.w2 * self.y 
    =  (self.e - e_real) * self.w2 * self.w1 * x
    = (self. w3 * self.w2 * self.w1 * x - e_real) * self.w2 * self.w1 * x

dw2 = (self.e - e_real) * self.w3* self.y 
	= (self.e - e_real) * self.w3* self.w1 * x 
    = (self. w3 * self.w2 * self.w1 * x - e_real) * self.w3* self.w1 * x
    
dw3 =  self.result3 * self.z
	= (self.e - e_real) * self.w2 * self.w1 * x
    = (self. w3 * self.w2 * self.w1 * x - e_real)* self.w2 * self.w1 * x
```

而损失函数展开可以表示为：$L(w1,w2,w3) =  \frac {1} 2 (w3 * w2 * w1 * x - \hat e)^2​$

对各权重参数求导为：

$\frac{dL} {dw3} = (w3 * w2 * w1 * x - \hat e) * w2 * w1$

$\frac{dL} {dw2} = (w3 * w2 * w1 * x - \hat e) * w3 * w1$

$\frac{dL} {dw1} = (w3 * w2 * w1 * x - \hat e) * w3 * w2​$

可以发现我们代码展开，与我们实际的公式求导是一致的证明我们代码是正确的。

**但是，一旦层数很深，那么我们就不能这么做了**

**我们需要用代码自动判断是否反向传播写对了。**

### 代码自动判断反向传播的导函数是否正确

这个和手工判断方法类似。反向传播是否正确，关键在于$\frac {dL} {dw_i}$ 是否计算正确。根据高中学过的导数的定义,对于位于点$\theta$的导数$f'(\theta)$有：

$f'(\theta) = \lim_{\epsilon\to 0} \frac{f(\theta + \epsilon) + f(\theta - \epsilon)  } {2\epsilon}$

所以我们可以看反向传播求的导函数值和用导数定义求的导函数值是否接近。

即我们需要让代码判断这个式子是否成立：$ \frac{dL} {dwi}  \approx \frac {L(wi+ 10^{-4}) - L(wi- 10^{-4})} {2* 10^{-4}}​$

左边的$ \frac{dL} {dwi} $是反向传播求得，右边是导数定义求的导数值。这两个导数应当大致相同。

**程序自动检验导函数是否正确实践：**

新增了一个梯度检验函数`check_gradient()`，如下所示：

```python
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
dl/dw1反向传播求得：-0.026729,定义求得-0.026727
dl/dw3反向传播求得：0.003970,定义求得0.004164
dl/dw2反向传播求得：-0.032617,定义求得-0.033257
dl/dw1反向传播求得：-0.027502,定义求得-0.027499
dl/dw3反向传播求得：0.004164,定义求得0.004367
dl/dw2反向传播求得：-0.033272,定义求得-0.033932
dl/dw1反向传播求得：-0.028291,定义求得-0.028288
dl/dw3反向传播求得：0.004367,定义求得0.004579
dl/dw2反向传播求得：-0.033947,定义求得-0.034625
dl/dw1反向传播求得：-0.029097,定义求得-0.029094
... ...
1 , -1
-1 , 1
'''
```

可以发现反向传播求得损失函数对各参数求得的导数和我们用高中学的定义法求导数，两者基本一致，证明我们反向传播求得的导数没有问题。












