"""
@Project   : DuReader
@Module    : two_sum.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/11/18 6:00 PM
@Desc      : 
"""
'''
1. Two Sum

Given an array of integers, return indices of the two numbers such 
that they add up to a specific target.

You may assume that each input would have exactly one solution, 
and you may not use the same element twice.

Example: 

Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
'''

'''
自己写的代码：

是将数据和对应的下标组成二维数组排序，并过滤出不可能的值。
比如最小值-2，目标值是6，那么大于8的数据就应该被过滤掉。
遍历筛选后的数组进行查找，并pop被遍历的值，从而降低下一次查找的复杂度。运行时间68ms。

'''

from bisect import bisect_left


def index(a, x):
    'binary search: Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError


class Solution:
    def twoSum(self, nums, target):
        """
        time complexity O(nlogn), space complexity O(n)
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        n = len(nums)
        A = [[i, num] for i, num in enumerate(nums)]
        # space O(n)
        A = sorted(A, key=lambda x: x[1], reverse=True)
        # sort a list of list
        # space O(n)
        A_min = target - A[0][1]
        A_max = target - A[n - 1][1]
        A = filter(lambda x: A_min <= x[1] <= A_max, A)
        # space O(n)

        # idxs = []
        # numbers = []
        # for idx, number in A:
        #     idxs.append(idx)
        #     numbers.append(number)
        idxs, numbers = zip(*A)
        idxs = list(idxs)
        numbers = list(numbers)
        # space O(n)

        idx_left = None
        while 1:
            idx_right = idxs.pop()
            num_right = numbers.pop()
            num_left = target - num_right
            try:
                # idx_left = idxs[numbers.index(num_left)]
                idx_left = idxs[index(numbers, num_left)]
                # binary search
                break
            except:
                pass

        return [idx_left, idx_right]


'''
受讨论区启发后写的代码：

建立字典，遍历数组，每个值的匹配值如果不在字典中就把这个值放入字典，直到找到匹配值为止。
运行时间49ms。
'''


class Solution2:
    def twoSum(self, nums, target):
        """time comlexity O(n), space complexity O(n)"""
        dic = {}
        # extra space k*n, k is a large number
        for idx, num in enumerate(nums):
            if target - num in dic:
                # hash search
                return [dic[target - num], idx]
            else:
                dic[num] = idx


if __name__ == '__main__':
    print(Solution().twoSum([2, 5, 3, 10], 5))
    print(Solution2().twoSum([2, 5, 3, 10], 5))
