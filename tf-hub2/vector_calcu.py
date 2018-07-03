"""
@Project   : DuReader
@Module    : vector_calcu.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/15/18 10:44 AM
@Desc      : 
"""

"""
Created on Sun Aug 20 14:40:29 2017

@author: zimuliu
"""

from functools import reduce
from math import acos, pi
import numpy as np


class Vector:
    def __init__(self, coordinates):
        self.coordinates = tuple(coordinates)
        self.dimension = len(coordinates)

    def __str__(self):
        return "%dD Vector: %s" % (self.dimension,
                                   ', '.join(["%.3f" % round(x, 3)
                                              for x in self.coordinates]))

    def __eq__(self, v):
        """两向量相等"""
        return self.coordinates is v.coordinates

    def _eq_dim(self, v):
        """两向量维度相同"""
        assert self.dimension is v.dimension, \
            "The dimensions of vectors must be equal!"

    def _zero_vec(self):
        """零向量"""
        assert self.magnitude() != 0, "Encount with zero vector!"

    def plus(self, v):
        """两向量相加"""
        self._eq_dim(v)
        return Vector([x + y for x, y in zip(self.coordinates, v.coordinates)])

    def plus2(self, v):
        self._eq_dim(v)
        temp = np.array(self.coordinates) + np.array(v.coordinates)
        return Vector(temp.tolist())

    def minus(self, v):
        """两向量相减"""
        self._eq_dim(v)
        return Vector([x - y for x, y in zip(self.coordinates, v.coordinates)])

    def minus2(self, v):
        self._eq_dim(v)
        temp = np.array(self.coordinates) - np.array(v.coordinates)
        return Vector(temp.tolist())

    def scalar_mult(self, m):
        """向量乘以标量"""
        return Vector([x * m for x in self.coordinates])

    def scalar_mult2(self, m):
        temp = np.array(self.coordinates)*m
        return Vector(temp.tolist())

    def magnitude(self, *args):
        """求向量的norm"""
        return reduce(lambda x, y: x + y,
                      map(lambda z: z ** 2, self.coordinates)) ** 0.5

    def magnitude2(self):
        return np.linalg.norm(self.coordinates)

    def direction(self, *args):
        """转化为向量所在方向的方向向量; 或者说，求单位向量"""
        self._zero_vec()
        return self.scalar_mult(1 / self.magnitude())

    def dot_product(self, v):
        """求向量的点乘，与矩阵的内积有关联"""
        self._eq_dim(v)
        return reduce(lambda x, y: x + y,
                      [a * b for a, b in zip(self.coordinates, v.coordinates)])

    def dot_product2(self, v):
        self._eq_dim(v)
        a = np.array(self.coordinates)
        b = np.array(v.coordinates)
        temp = np.dot(a, b)
        print('temp in dot_product2:', temp)
        print('type of temp:', type(temp))
        print('type of temp.tolist():', type(temp.tolist()))
        return temp.tolist()

    def multiply_elementwise(self, v):
        self._eq_dim(v)
        return Vector([a * b for a, b in zip(self.coordinates, v.coordinates)])

    def multiply_elementwise2(self, v):
        self._eq_dim(v)
        temp = np.multiply(self.coordinates, v.coordinates)
        return temp.tolist()

    def cross_product(self, v):
        def cross(a, b):
            c = [a[1] * b[2] - a[2] * b[1],
                 a[2] * b[0] - a[0] * b[2],
                 a[0] * b[1] - a[1] * b[0]]

            return c
        self._eq_dim(v)
        a0 = self.coordinates
        b0 = v.coordinates
        return cross(a0, b0)

    def cross_product2(self, v):
        self._eq_dim(v)
        a = np.array(self.coordinates)
        b = np.array(v.coordinates)
        temp = np.cross(a, b)
        return temp.tolist()

    def angle(self, v, degree=False):
        """求两个向量的夹角大小，可以表征两个向量的相似度;
        可以选择用实数表示还是用度数表示"""
        self._zero_vec()
        v._zero_vec()
        measurement = pi / 180 if degree else 1
        return acos(self.dot_product(v) / (self.magnitude() * v.magnitude())) \
               / measurement

    def parallelism(self, v, threshold=10e-6):
        """判断两个向量是否平行"""
        self._eq_dim(v)
        res = False
        if self.magnitude() < threshold or v.magnitude() < threshold:
            res = True
        else:
            ang = self.angle(v)
            if ang < threshold or (pi - ang) < threshold:
                res = True
        return res

    def orthogonality(self, v, threshold=10e-6):
        """判断两个向量是否垂直"""
        return abs(self.dot_product(v)) < threshold

    def projection(self, v):
        """求一个向量在另一个向量方向上的投影"""
        _v = v.direction()
        weight = self.dot_product(_v)
        return _v.scalar_mult(weight)


if __name__ == '__main__':
    a = Vector([1, 2])
    b = Vector([3, 4])
    print(a.magnitude())
    print(a.magnitude2())
    print(a.plus(b))
    print(a.plus2(b))
    print(a.minus(b))
    print(a.minus2(b))
    print(a.scalar_mult(2))
    print(a.scalar_mult2(2))
    print(a.dot_product(b))
    print(a.dot_product2(b))
    print(a.multiply_elementwise(b))
    print(a.multiply_elementwise2(b))
    print(a.angle(b))
    print(a.parallelism(b))
    print(a.orthogonality(b))
    print(a.projection(b))

    c = Vector([1, 2, 3])
    d = Vector([4, 5, 6])
    print(c.cross_product(d))
    print(c.cross_product2(d))
