"""
@Project   : DuReader
@Module    : det.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/10/18 1:32 PM
@Desc      :
Numba is an Open Source NumPy-aware optimizing compiler for Python sponsored
by Anaconda, Inc. It uses the remarkable LLVM compiler infrastructure to
compile Python syntax to machine code.

It is aware of NumPy arrays as typed memory regions and so can speed-up code
using NumPy arrays. Other, less well-typed code will be translated to
Python C-API calls effectively removing the "interpreter" but not removing
the dynamic indirection.
"""

"""
Created on Sat May 20 18:27:41 2017

@author: Liuzimu
"""
"""
算法：
1.直接计算行列式的展开式，把各种排列组合搞出来，并根据逆序数计算是奇排列还是偶排列；
2.行列式的m行 * k加到n行之后行列式的值不变，化简为上三角行列式，并求对角线的乘积
3.拉普拉斯定理，化为C(k, n)个k阶子式和代数余子式，递归实现
采用算法2
"""

from numba import jit


@jit
def get_det(mat):
    # 为了节省空间，直接在输入的行列式上进行了化简，而没有使用copy
    mat = mat.astype('float')
    n = mat.shape[0]
    res = 1
    # 遍历列
    for col in range(n):
        row = col
        res *= mat[row][col]
        # 寻找不是0的位置row
        while mat[row][col] == 0 and row < n - 1:
            row += 1
        # 化简mat[row,col]下面的每一元素为0
        for i in range(row + 1, n):
            if mat[i][col] == 0:
                pass
            else:
                k = - mat[i][col] / mat[row][col]
                for j in range(col ,n):
                    mat[i][j] += mat[row][j] * k
    return res


"""
测试：与numpy自带的计算行列式函数进行结果比较
"""
from numpy.random import rand, seed
from numpy.linalg import det
# 生成随机行列式
seed(100)
n = 5
A = rand(n ** 2)
mat = A.reshape(n, n)
# 打印numpy函数和自定义函数的结果
print(det(mat), get_det(mat))

'''
#利用iPython的命令对比运行时间，猜猜谁快^
%timeit det(mat) #100000 loops, best of 3: 7.31 µs per loop
%timeit get_det(mat) #10000 loops, best of 3: 52.4 µs per loop
#jit 100000 loops, best of 3: 3.52 µs per loop
'''
