## 高层封装


### estimator

* [参考资料](03.intro_tensorflow/3.-Estimator-API.pdf)

![](03.intro_tensorflow/tensorflow_hierarchy.png)
* why
    * Create production-ready machine learning models using an API
    * Train on large datasets that do not fit in memory
    * Quickly monitor your training metrics in Tensorboard
* 体系构建

![](03.intro_tensorflow/tensorflow_estimator_01.png)

1. 读取大数据， 加载读取方式，构建图，不是真的读取

![](03.intro_tensorflow/tensorflow_dataset_01.png)

2. 数据的分布: 如何划分数据，数据的使用方式

![](03.intro_tensorflow/tensorflow_estimator_02.png)
3. 可视化

![](03.intro_tensorflow/tensorflow_estimator_04.png)
![](03.intro_tensorflow/tensorflow_estimator_05.png)
4. 发布

![](03.intro_tensorflow/tensorflow_estimator_06.png)
![](03.intro_tensorflow/tensorflow_estimator_03.png)
    
* 总结

![](03.intro_tensorflow/tensorflow_estimator_07.png)

* [code](study_api/study_estimator.py)



### contrib.learn

* 引入
```
import tensorflow as tf
from tensorflow.contrib import learn
```
* 函数

![](03.intro_tensorflow/tensorflow_contrib_01.png)
* 调用四步

![](03.intro_tensorflow/tensorflow_contrib_02.png)


### Pretrained models with TF-Slim

* 函数

![](03.intro_tensorflow/tensorflow_slim_01.png)
* 导入
```
from tensorflow.contrib import slim
```

#### Downloading and using a pretrained model

* 下载模型

![](03.intro_tensorflow/tensorflow_slim_02.png)
* 准备数据

![](03.intro_tensorflow/tensorflow_slim_03.png)

* 使用你的模型
![](03.intro_tensorflow/tensorflow_slim_04.png)

