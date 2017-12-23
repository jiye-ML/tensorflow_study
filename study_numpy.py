import numpy as np

# http://blog.csdn.net/huruzun/article/details/39801217
# http://blog.chinaunix.net/uid-21633169-id-4408596.html

"""
NumPy函数和方法分类排列目录:
    
    创建数组
        arange, array, copy, empty, empty_like, eye, fromfile, 
        fromfunction, identity, linspace, logspace, mgrid, 
        ogrid, ones, ones_like, r , zeros, zeros_like 
    转化
        astype, atleast 1d, atleast 2d, atleast 3d, mat 
    操作
        array split, column stack, concatenate, diagonal, 
        dsplit, dstack, hsplit, hstack, item, newaxis, ravel, 
        repeat, reshape, resize, squeeze, swapaxes, take, 
        transpose, vsplit, vstack 
    询问
        all, any, nonzero, where 
    排序
        argmax, argmin, argsort, max, min, ptp, searchsorted, 
        sort 
    运算
        choose, compress, cumprod, cumsum, inner, fill, imag, 
        prod, put, putmask, real, sum 
    基本统计
        cov, mean, std, var 
    基本线性代数
        cross, dot, outer, svd, vdot 
    数组文件输入输出
        load, save, loadtxt
        
"""

# 多维数组的类型是：numpy.ndarray
arr_a = [[1, 2], [3, 4]]
tup_a = (1, 2, 3, 4)
np_a = np.array(arr_a)
np_b = np.array(tup_a)
print(type(np_a))
print(type(np_b))
print(np_a)
print(np_b)


# 数据类型设定与转换
np_string = np.array(['12', '23', '43'], dtype=np.string_)
print(np_string)
np_int = np_string.astype(int)
print(np_int)
np_float = np_string.astype(float)
print(np_float)
np_int = np_float.astype(int)
print(np_int)
print(np_string)


# 索引与切片(切片属于浅拷, like a view)
array = np.array([[1, 2, 3],
                  [4, 5, 6]])
# index
print(array[1, 2])
# slice
print(array[1, :])
print(array[:, 2])
array[1, 1] = 100
print(array)
array[1, 1] = 5
# this is a copy, not is a view
array_slice_copy = array[1, :].copy()
array_slice_copy[1] = 100
print(array_slice_copy)
print(array)


# 布尔型索引
names = np.array(['Bob', 'joe', 'Bob', 'will'])
print(names[names == 'Bob'])
data = np.array([[0.36762706, -1.556689452,  0.84316735, -0.11648442],
                 [1.34023966,  1.127664186,  1.12507441, -0.68689309],
                 [1.27392366, -0.433996417, -0.80444728,  1.60731881],
                 [0.23361565,  1.387724715,  0.69129479, -1.19228023],
                 [0.51353082,  0.176964698, -0.06753478,  0.80448168],
                 [0.21773096,  0.605842802, -0.46446071,  0.83131122],
                 [0.50569072,  0.044341685, -0.69358155, -0.96249124]])
data[data < 0] = 0
print(data)


# arange
print(np.arange(1, 20, 3, float))
print(np.arange(20))
print(np.arange(20).reshape(4, 5))

# linspace
print(np.linspace(start=1, stop=20, num=30, endpoint=True, retstep=True, dtype=float))

# zeros / ones / eye / empty
print(np.zeros((3, 8), dtype=float))
print(np.ones((3, 8), dtype=int))
print(np.eye(N=5, M=5, k=0, dtype=int))
print(np.eye(N=5, M=6, k=0, dtype=int))
print(np.eye(N=5, M=6, k=2, dtype=int))
# 函数empty创建一个内容随机并且依赖与内存状态的数组
print(np.empty((3, 15), dtype=float))


# 获取数组的属性
np_array = np.ones((10, 12), dtype=int)
print(np_array.ndim)
print(np_array.shape)
print(np_array.size)
print(np_array.dtype)
print(np_array.itemsize)
print(np_array.nbytes)
print(np_array.base)
print(np_array.ctypes)
print(np_array.strides)
print(np_array.flags)
print(np_array.flat)
print(np_array.T)
print(np_array.imag)
print(np_array.real)


# methods
np_array = np.ones((2, 4), dtype=int)
# sum
print(np_array.sum(axis=0))
print(np_array.sum(axis=1))
print(np_array.sum(axis=1, keepdims=True))
# 合并数组: vstack / hstack (深拷贝)
np_array_2 = np.zeros((2, 4), dtype=int)
print(np.vstack((np_array, np_array_2)))
print(np.hstack((np_array, np_array_2)))


# 基本的矩阵运算
data = np.array([[0.36762706, -1.556689452,  -0.84316735, -0.11648442],
                 [1.34023966,  1.127664186,  1.12507441, -0.68689309],
                 [1.27392366, -0.433996417, -0.80444728,  1.60731881],
                 [0.23361565,  1.387724715,  0.69129479, -1.19228023]])
data[data > 0] = 5
data[data < 0] = 1
print(data)
print(data.T)
print(data.transpose())


# numpy.linalg模块
det = np.linalg.det(np.array([[1, 2], [1, 4]]))
print(det)


# numpy.random模块
print(np.random.rand(10))


# 设置print options参数来更改打印选项
np.set_printoptions(threshold=np.nan)
print(np.ones((50, 200), dtype=int))


# 数组的算术运算是按元素的
data = np.array([[1, 2, 5], [2, 3, 4], [3, 4, 5]], dtype=int)
print(data)
data_b = data * 2
print(data_b)
data_c = data ** 2
print(data_c)
print(data_c - data_b)
print(data_c > 10)
# NumPy中的乘法运算符*指示按元素计算
print(data_b * data_c)
# 矩阵乘法可以使用dot函数或创建矩阵对象实现
print(np.dot(a=data_b, b=data_c))
print(np.dot(a=data_c, b=data_b))


print(data.sum())
print(data.sum(axis=0))
print(data.sum(axis=1))
print(data.max())
print(data.max(axis=0))
print(data.max(axis=1))
print(data.min())
print(data.min(axis=0))
print(data.min(axis=1))


# 通用函数(ufunc)
print(np.exp(np.arange(4)))
print(np.sqrt(np.arange(4)))
print(np.add(np.array(4), np.array(4)))


# slice : 1D
data = np.arange(10)**3
print(data)
# equivalent to a[0:6:2] = -1000;
# from start to position 6, exclusive,
# set every 2nd element to -1000
data[: 8: 2] = 1000
print(data)
# reversed a
print(data[:: -1])

# slice : 2D
data = np.fromfunction(lambda x, y: 10*x+y, (5, 4), dtype=int)
print(data)
print(data[2, :])
print(data[1:4, 1])
print(data[2:4, :])
# 当少于轴数的索引被提供时，确失的索引被认为是整个切片：
print(data[2:])
print(data[2:-1])
print(data[::-1])
print(data[::-1, ::-1])
# 点(…)代表许多产生一个完整的索引元组必要的分号
data = np.fromfunction(lambda x, y, z: 100*x+10*y+z, (3, 3, 3), dtype=int)
print(data)
print(data[..., 1])
print(data[1, ..., 2])
# 迭代多维数组是就第一个轴而言的
for row in data:
    for row2 in row:
        print(row2)
# 如果想对每个数组中元素进行运算，可以使用flat属性.
# 该属性是数组元素的一个迭代器
for element in data.flat:
    print(element)


# reshape
data = np.fromfunction(lambda x, y: 10*x+y, (3, 5), dtype=int)
print(data)
print(data.shape)
print(data.reshape(5, 3))
# 如果在改变形状操作中一个维度被给做-1，其维度将自动被计算
print(data.reshape(5, -1))
print(data.reshape(-1, 3))
print(data.ravel())


# 组合(stack)不同的数组
data_1 = np.fromfunction(lambda x, y: 5*x+y, (3, 5), dtype=int)
data_2 = np.fromfunction(lambda x, y: 6*x+y, (3, 5), dtype=int)
print(data_1)
print(data_2)
print(np.vstack((data_1, data_2)))
print(np.hstack((data_1, data_2)))
# concatenate允许可选参数给出组合时沿着的轴
print(np.concatenate((data_1, data_2), axis=0))
print(np.concatenate((data_1, data_2), axis=1))
# 函数column_stack以列将一维数组合成二维数组，它等同与vstack对一维数组.
# row_stack函数，另一方面，将一维数组以行组合成二维数组.
data_1 = np.fromfunction(lambda x: 5*x, (3,), dtype=int)
data_2 = np.fromfunction(lambda x: 6*x, (3,), dtype=int)
print(data_1)
print(data_2)
print(np.column_stack((data_1, data_2)))
print(np.row_stack((data_1, data_2)))
# 在复杂情况下，r_[]和c_[]对创建沿着一个方向组合的数很有用,
# 它们允许范围符号(“:”)
print(np.r_[1:10, 0, 10])
print(np.c_[4:10])


# split
data_1 = np.fromfunction(lambda x, y: 5*x+y, (6, 6), dtype=int)
print(data_1)
# hsplit沿着数组的水平轴分割
# 指定返回相同形状数组的个数,
print(np.hsplit(data_1, 3))
# 指定在哪些列后发生分割
print(np.hsplit(data_1, (3, 5)))
# vsplit沿着数组纵向轴分割
# 指定返回相同形状数组的个数,
print(np.vsplit(data_1, 3))
# 指定在哪些列后发生分割
print(np.vsplit(data_1, (3, 5)))


# view
# 同的数组对象分享同一个数据.
# 视图方法创造一个新的数组对象指向同一数据。
# 切片数组返回它的一个视图
data_1 = np.fromfunction(lambda x, y: 5*x+y, (6, 6), dtype=int)
data_2 = data_1.view()
print(data_1)
print(data_2)
data_1[1, 1] = 1000
print(data_1)
print(data_2)
data_2[2, 2] = 1000
print(data_1)
print(data_2)
print(data_1.flags.owndata)
print(data_2.flags.owndata)

# copy
# 这个复制方法完全复制数组和它的数据
data_1 = np.fromfunction(lambda x, y: 5*x+y, (6, 6), dtype=int)
data_2 = data_1.copy()
print(data_1)
print(data_2)
data_1[1, 1] = 1000
print(data_1)
print(data_2)
data_2[2, 2] = 1000
print(data_1)
print(data_2)
print(data_1.flags.owndata)
print(data_2.flags.owndata)


# 数据文件输入输出
arr = np.arange(10)
np.save('some_array',arr)
np.load('some_array.npy')
arr = np.loadtxt('dataMatrix.txt',delimiter=' ')


"""
广播法则能使通用函数有意义地处理不具有相同形状的输入。

    广播第一法则:
        如果所有的输入数组维度不都相同，一个“1”将被重复地添加在维度较小的
        数组上直至所有的数组拥有一样的维度。

    广播第二法则:
        确定长度为1的数组沿着特殊的方向表现地好像它有沿着那个方向最大形状
        的大小。对数组来说，沿着那个维度的数组元素的值理应相同。

应用广播法则之后，所有数组的大小必须匹配.
"""


# 数组索引
# 数组可以被整数数组和布尔数组索引


