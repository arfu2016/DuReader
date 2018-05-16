"""
@Project   : DuReader
@Module    : add_two_numbers.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/16/18 10:29 AM
@Desc      : 
"""
'''
2. Add Two Numbers

You are given two non-empty linked lists representing two non-negative 
integers. The digits are stored in reverse order and each of their nodes 
contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, 
except the number 0 itself.

Example

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
自己写的代码：

首先定义valNext方法，返回当前Node的值，以及next Node。遍历l1, l2，
每次相加的时候如果大于9，则把进位存储到m中，直到l1，l2都没有元素，
且m=0循环停止。运行时间196 ms。
'''


class ListNode:

    def __init__(self, value):
        self.val = value
        self.next = None


class Solution:
    def valNext(self, l):
        if l:
            res = l.val
            l = l.next
        else:
            res = 0
        return res, l

    def addTwoNumbers(self, l1, l2):
        res = ListNode(0)
        tmp = res
        m = 0
        while l1 or l2 or m:
            # If a class defines neither __len__() nor __bool__(),
            # all its instances are considered true.
            # l1, l2有可能是None，也就是链表的末尾
            val1, l1 = self.valNext(l1)
            # 进入下一个list node
            val2, l2 = self.valNext(l2)
            k = val1 + val2 + m
            if k > 9:
                k -= 10
                m = 1
                # 进位
            else:
                m = 0
            tmp.next = ListNode(k)
            tmp = tmp.next
            # 先让res.next指向ListNode(k)，然后让tmp指向该节点，使得链表可以延续下去
        return res.next
    # res.next就放着linked list的引用


'''
受讨论区启发后写的代码：

思路与我自己想的几乎一样，只是用了divmod函数让代码更加简洁。运行时间195 ms。
'''


class Solution2:
    def addTwoNumbers(self, l1, l2):
        res = tmp = ListNode(0)
        m = 0
        while l1 or l2 or m:
            if l1:
                m += l1.val
                l1 = l1.next
            if l2:
                m += l2.val
                l2 = l2.next
            tmp.next = ListNode(0)
            tmp = tmp.next
            m, tmp.val = divmod(m, 10)
        return res.next
