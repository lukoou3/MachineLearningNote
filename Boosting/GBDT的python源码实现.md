# GBDT的python源码实现

## GBDT的python源码实现
摘抄自知乎：`https://zhuanlan.zhihu.com/p/32181306`

### 1.前言
这次我们来讲解Gradient Bosting Desicion Tree的python实现，关于GBDT的原理我浏览了许多教材和blog都没有发现讲解的非常清晰的，后面翻墙去谷歌看了一篇PPT讲解的非常透彻，豁然开朗，虽然ppt是全英的，但阅读难度真心不大，大家可以去看看[ccs.neu.edu/home/vip/te](http://www.ccs.neu.edu/home/vip/teach/MLcourse/4_boosting/slides/gradient_boosting.pdf "")

**建议阅读顺序**：先阅读源代码，再来看源码关键方法的讲解，源码地址

不知为何知乎上的**代码格式**没有原文章便于理解，大家可在[cs229论坛社区|深度学习社区|机器学习社区|人工智能社区](http://www.dmlearning.cn/single/a5bf33e7b2c44e499a1cb7b2d5f8fbfa.html "cs229论坛社区|深度学习社区|机器学习社区|人工智能社区")上阅读

### 2.源码讲解
GBDT与随机森林一样需要使用到决策树的子类，对于决策树子类的代码讲解在我上一篇文章中。若是大家之前没有了解过决策树可以看我这一篇文章[随机森林，gbdt，xgboost的决策树子类讲解](http://www.dmlearning.cn/single/c5f5c23878b04500a9ed74cf0e6e07bf.html "随机森林，gbdt，xgboost的决策树子类讲解")。

#### init()
```python
"""Parameters:
-----------
n_estimators: int
    树的数量
    The number of classification trees that are used.
learning_rate: float
    梯度下降的学习率
    The step length that will be taken when following the negative gradient during
    training.
min_samples_split: int
    每棵子树的节点的最小数目（小于后不继续切割）
    The minimum number of samples needed to make a split when building a tree.
min_impurity: float
    每颗子树的最小纯度（小于后不继续切割）
    The minimum impurity required to split the tree further.
max_depth: int
    每颗子树的最大层数（大于后不继续切割）
    The maximum depth of a tree.
regression: boolean
    是否为回归问题
    True or false depending on if we're doing regression or classification.
"""
 
def __init__(self, n_estimators, learning_rate, min_samples_split,
             min_impurity, max_depth, regression):
 
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate
    self.min_samples_split = min_samples_split
    self.min_impurity = min_impurity
    self.max_depth = max_depth
    self.regression = regression
 
    # 进度条 processbar
    self.bar = progressbar.ProgressBar(widgets=bar_widgets)
 
    self.loss = SquareLoss()
    if not self.regression:
        self.loss = SotfMaxLoss()
 
    # 分类问题也使用回归树，利用残差去学习概率
    self.trees = []
    for i in range(self.n_estimators):
        self.trees.append(RegressionTree(min_samples_split=self.min_samples_split,
                                         min_impurity=self.min_impurity,
                                         max_depth=self.max_depth))
```
创建n_estimators棵树的GBDT，注意这里的分类问题也使用回归树，利用残差去学习概率

#### fit()
```python
def fit(self, X, y):
    # 让第一棵树去拟合模型
    self.trees[0].fit(X, y)
    y_pred = self.trees[0].predict(X)
    for i in self.bar(range(1, self.n_estimators)):
        gradient = self.loss.gradient(y, y_pred)
        self.trees[i].fit(X, gradient)
        y_pred -= np.multiply(self.learning_rate, self.trees[i].predict(X))
```
**for**循环的过程就是不断让下一棵树拟合上一颗树的"残差"(梯度)。

而"残差"是由梯度求出。在**square loss**中，gradient = yi - F(xi),此时梯度刚好等于残差(这里是真正的残差)。

在其他的损失函数中其实拟合的是梯度，具体的细节可以查看我上面推荐的ppt，讲的非常详细。

#### predict()
```python
def predict(self, X):
    y_pred = self.trees[0].predict(X)
    for i in range(1, self.n_estimators):
        y_pred -= np.multiply(self.learning_rate, self.trees[i].predict(X))
 
    if not self.regression:
        # Turn into probability distribution
        y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
        # Set label to the value that maximizes probability
        y_pred = np.argmax(y_pred, axis=1)
    return y_pred
```
**for**循环的过程就是汇总各棵树的残差得到最后的结果
    
### 3.源码地址
[github.com/RRdmlearning](https://github.com/RRdmlearning/Machine-Learning-From-Scratch/tree/master/gradient_boosting_decision_tree "")

直接运行[gbdt_classifier_example.py](https://github.com/RRdmlearning/Machine-Learning-From-Scratch/blob/master/gradient_boosting_decision_tree/gbdt_classifier_example.py "")或[gbd_regressor_example.py](https://github.com/RRdmlearning/Machine-Learning-From-Scratch/blob/master/gradient_boosting_decision_tree/gbd_regressor_example.py "")文件即可。


直接运行项目包括了许多机器学习算法的简洁实现    
    
此文章为记录自己一路的学习路程，也希望能给广大初学者们一点点帮助，如有错误,疑惑欢迎一起交流。

## 通俗讲解GBDT
摘抄自知乎：`https://zhuanlan.zhihu.com/p/93605767`

看《统计学习方法》GBDT那一章，被一堆公式弄晕了，网上找了很多博客，都没怎么弄懂。

还是看到这篇博客：GBDT的python源码实现  [https://zhuanlan.zhihu.com/p/32..](https://zhuanlan.zhihu.com/p/32181306 "")，结合代码才大概了解，在这里分享一下本人的粗浅理解，如有错误欢迎指出。

给定样本集： $(x_1,y_1)...(x_n,y_n)$ ；

损失函数： $L(y,f(x))$ ；

树模型： $T(x;\Theta)$ ；

我们的目的：用较弱的树模型组合出一个强模型 $f(x)$ ，来进行样本的预测。

**直接进入算法流程：**

**第一步**：初始化 $f(x)$ ，先找到最合适的参数 $\Theta_1$ ， 使得 $\sum_{i=1}^{n} L(y_i,T(x_i;\Theta_1))$ 最小，模型初始化， $f(x)=T(x;\Theta_1)$ ;

**第二步**：

1. 如果这里的损失函数是均方误差，就如书中P169那样，我们通过 $y_i-f(x_i)$ 能计算得到各样本点预测的残差，我们要拟合的样本集就变成了 $(x_1,y_1-f(x_1))...(x_n,y_n-f(x_n))$ ，按照老套路找最合适的 $T(x;\Theta_2)$ 来拟合这个集合，最后模型升级， $f(x)=T(x;\Theta_1)+T(x;\Theta_2)$ ；

2. 再推广到一般的情况，利用残差的近似值，书中P171给出的负梯度 $\frac{\partial L(y_i, f(x_i)))}{\partial f(x_i)}$ 来代替，我们要拟合的样本集就变成了 $ (x_1,\frac{\partial L(y_1, f(x_1)))}{\partial f(x_1)})...(x_n,\frac{\partial L(y_n, f(x_n)))}{\partial f(x_n)})$ ，后面的方法就跟上面一样了；

**第三步**：设定一个停止条件，比如树的数量 $M$ ，利用第二步的思路一直拟合下去，最终会得到我们需要的强模型， $f(x)=\sum_{i=1}^{M}T(x;\Theta_i)$ 。











