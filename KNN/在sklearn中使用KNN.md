# 在sklearn中使用KNN

## 基本使用

### sklearn.neighbors
sklearn.neighbors提供监督的基于邻居的学习方法的功能，sklearn.neighbors.KNeighborsClassifier是一个最近邻居分类器。那么KNeighborsClassifier是一个类，我们看一下实例化时候的参数
```python
class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs)**
  """
  :param n_neighbors：int，可选（默认= 5），k_neighbors查询默认使用的邻居数

  :param algorithm：{'auto'，'ball_tree'，'kd_tree'，'brute'}，可选用于计算最近邻居的算法：'ball_tree'将会使用 BallTree，'kd_tree'将使用 KDTree，“野兽”将使用强力搜索。'auto'将尝试根据传递给fit方法的值来决定最合适的算法。

  :param n_jobs：int，可选（默认= 1),用于邻居搜索的并行作业数。如果-1，则将作业数设置为CPU内核数。不影响fit方法。

  """
```


```python
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
```

### Method
#### fit(X, y)
使用X作为训练数据拟合模型，y作为X的类别值。X，y为数组或者矩阵
```python
X = np.array([[1,1],[1,1.1],[0,0],[0,0.1]])
y = np.array([1,1,0,0])
neigh.fit(X,y)
```

#### predict(X)
预测提供的数据的类标签
```python
neigh.predict(np.array([[0.1,0.1],[1.1,1.1]]))
```

#### predict_proba(X)
返回测试数据X属于某一类别的概率估计
```python
neigh.predict_proba(np.array([[1.1,1.1]]))
```

#### kneighbors(X=None, n_neighbors=None, return_distance=True)
找到指定点集X的n_neighbors个邻居，return_distance为False的话，不返回距离
```python
neigh.kneighbors(np.array([[1.1,1.1]]),return_distance= False)

neigh.kneighbors(np.array([[1.1,1.1]]),return_distance= False,an_neighbors=2)
```

## KNeighborsClassifier类参数和方法

### 参数
```python
class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=None, **kwargs)
```

* **n_neighbors**: int, 可选参数(默认为 5)。用于kneighbors查询的默认邻居的数量

* **weights（权重）**: str or callable(自定义类型), 可选参数(默认为 ‘uniform’)。用于预测的权重参数，可选参数如下：

    * uniform : 统一的权重. 在每一个邻居区域里的点的权重都是一样的。

    * distance : 权重点等于他们距离的倒数。使用此函数，更近的邻居对于所预测的点的影响更大。

    * [callable] : 一个用户自定义的方法，此方法接收一个距离的数组，然后返回一个相同形状并且包含权重的数组。

* **algorithm（算法）**: {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, 可选参数（默认为 ‘auto’）。计算最近邻居用的算法：

    * ball_tree 使用算法BallTree

    * kd_tree 使用算法KDTree

    * brute 使用暴力搜索

    * auto 会基于传入fit方法的内容，选择最合适的算法。

    * 注意 : 如果传入fit方法的输入是稀疏的，将会重载参数设置，直接使用暴力搜索。

* **leaf_size（叶子数量）**: int, 可选参数(默认为 30)。传入BallTree或者KDTree算法的叶子数量。此参数会影响构建、查询BallTree或者KDTree的速度，以及存储BallTree或者KDTree所需要的内存大小。此可选参数根据是否是问题所需选择性使用。

* **p: integer**, 可选参数(默认为 2)。用于Minkowski metric（闵可夫斯基空间）的超参数。p = 1, 相当于使用曼哈顿距离，p = 2, 相当于使用欧几里得距离]，对于任何 p ，使用的是闵可夫斯基空间。

* **metric（矩阵）**: string or callable, 默认为 ‘minkowski’。用于树的距离矩阵。默认为闵可夫斯基空间，如果和p=2一块使用相当于使用标准欧几里得矩阵. 所有可用的矩阵列表请查询 DistanceMetric 的文档。

* **metric_params（矩阵参数）**: dict, 可选参数(默认为 None)。给矩阵方法使用的其他的关键词参数。

* **n_jobs: int**, 可选参数(默认为 1)。用于搜索邻居的，可并行运行的任务数量。如果为-1, 任务数量设置为CPU核的数量。不会影响fit


### 方法
对于**KNeighborsClassifier**的方法：

| 方法名                                       | 含义                                                     |
| -------------------------------------------- | -------------------------------------------------------- |
| fit(X, y)                                    | 使用X作为训练数据，y作为目标值（类似于标签）来拟合模型。 |
| get_params([deep])                           | 获取估值器的参数。                                       |
| neighbors([X, n_neighbors, return_distance]) | 查找一个或几个点的K个邻居。                              |
| kneighbors_graph([X, n_neighbors, mode])     | 计算在X数组中每个点的k邻居的（权重）图。                 |
| predict(X)                                   | 给提供的数据预测对应的标签。                             |
| predict_proba(X)                             | 返回测试数据X的概率估值                                  |
| score(X, y[, sample_weight])                 | 返回给定测试数据和标签的平均准确值。                     |
| set_params(**params)                         | 设置估值器的参数。                                       |

## k-近邻算法案例分析
本案例使用最著名的”鸢尾“数据集，该数据集曾经被Fisher用在经典论文中，目前作为教科书般的数据样本预存在Scikit-learn的工具包中。

### 读入Iris数据集细节资料
```python
from sklearn.datasets import load_iris
# 使用加载器读取数据并且存入变量iris
iris = load_iris()

# 查验数据规模
iris.data.shape

# 查看数据说明（这是一个好习惯）
print iris.DESCR
```
通过上述代码对数据的查验以及数据本身的描述，我们了解到Iris数据集共有150朵鸢尾数据样本，并且均匀分布在3个不同的亚种；每个数据样本有总共4个不同的关于花瓣、花萼的形状特征所描述。由于没有制定的测试集合，因此按照惯例，我们需要对数据进行随即分割，25%的样本用于测试，其余75%的样本用于模型的训练。

由于不清楚数据集的排列是否随机，可能会有按照类别去进行依次排列，这样训练样本的不均衡的，所以我们需要分割数据，已经默认有随机采样的功能。

### 对Iris数据集进行分割
```python
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=42)
```

### 对特征数据进行标准化
```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
```

### 完整代码
K近邻算法是非常直观的机器学习模型，我们可以发现K近邻算法没有参数训练过程，也就是说，我们没有通过任何学习算法分析训练数据，而只是根据测试样本训练数据的分布直接作出分类决策。因此，K近邻属于无参数模型中非常简单一种。

```python
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def knniris():
    """
    鸢尾花分类
    :return: None
    """

    # 数据集获取和分割
    lr = load_iris()

    x_train, x_test, y_train, y_test = train_test_split(lr.data, lr.target, test_size=0.25)

    # 进行标准化

    std = StandardScaler()

    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # estimator流程
    knn = KNeighborsClassifier()

    # # 得出模型
    # knn.fit(x_train,y_train)
    #
    # # 进行预测或者得出精度
    # y_predict = knn.predict(x_test)
    #
    # # score = knn.score(x_test,y_test)

    # 通过网格搜索,n_neighbors为参数列表
    param = {"n_neighbors": [3, 5, 7]}

    gs = GridSearchCV(knn, param_grid=param, cv=10)

    # 建立模型
    gs.fit(x_train,y_train)

    # print(gs)

    # 预测数据

    print(gs.score(x_test,y_test))

    # 分类模型的精确率和召回率

    # print("每个类别的精确率与召回率：",classification_report(y_test, y_predict,target_names=lr.target_names))

    return None

if __name__ == "__main__":
    knniris()
```




```python

```