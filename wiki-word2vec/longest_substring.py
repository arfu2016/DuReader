"""
@Project   : DuReader
@Module    : longest_substring.py
@Author    : Deco [deco@cubee.com]
@Created   : 5/23/18 2:08 PM
@Desc      : 
"""

'''
3. Longest Substring Without Repeating Characters

Given a string, find the length of the longest substring without repeating 
characters.

Examples:

Given "abcabcbb", the answer is "abc", which the length is 3.

Given "bbbbb", the answer is "b", with the length of 1.

Given "pwwkew", the answer is "wke", with the length of 3. Note that the answer 
must be a substring, "pwke" is a subsequence and not a substring.


自己写的代码：

遍历字符串，用列表存储已经被遍历且不重复的字符，遍历下一个字符的时候判断一下这个字符是否
在列表中，如果在的话，就把列表中的这个字符以及这个字符前面的字符都删掉。运行时间145ms。
也试过用字典存储已经遍历过的字符及其下标，虽然判断字符是否在字典中存在的时候效率会高一些，
但是遇到重复字符的时候更新字典的效率却很低。运行时间高达314ms，最后还是放弃了字典，使用列表。
'''

from time import time


def func_runtime(func, n_iter, *args):
    """[summary]

    Arguments:
        func {function object} -- the function to test
        n_iter {int} -- number of iterations

    Returns:
        str -- test result
    """

    start = time()
    for _ in range(n_iter):
        func(*args)
    runtime = (time() - start) / n_iter
    return "Average %.5fs in %d loops" % (runtime, n_iter)


class Solution:
    def lengthOfLongestSubstring(self, s):
        sub = []
        res = 0
        for char in s:
            if char in sub:
                res = max(res, len(sub))
                sub = sub[sub.index(char)+1:]
            sub.append(char)
        res = max(res, len(sub))
        return res


class Solution1:
    def lengthOfLongestSubstring(self, s):
        sub = []
        sub_set = set(sub)
        res = 0
        for char in s:
            if char in sub_set:
                res = max(res, len(sub))
                sub = sub[sub.index(char)+1:]
            sub.append(char)
            sub_set = set(sub)
            # 在循环中不断产生新的set，时间开销太大
        res = max(res, len(sub))
        return res


'''
受讨论区启发后写的代码：

思路也是用字典，但是不去更新这个字典，我只想到了字典，但是非得要更新它。遇到重复字符之后，
还要判断重复的字符的下标是否大于不重复字符串的起始下标。这个思路确实妙啊。
'''


class Solution2:
    def lengthOfLongestSubstring(self, s):
        if s is "":
            return 0
        dic = {}
        res = start = 0
        for i, char in enumerate(s):
            if char in dic:
                j = dic[char]
                if j >= start:
                    res = max(res, i - start)
                    start = j + 1
            dic[char] = i
        res = max(res, i + 1 - start)
        return res


"""
而讨论区的代码更加简洁，虽然在没有遇到重复的时候也在计算最大长度，有点浪费计算，
但实际运行时效率并没有差异。
"""


class Solution3:
    # @return an integer
    def lengthOfLongestSubstring(self, s):
        start = maxLength = 0
        usedChar = {}

        for i in range(len(s)):
            if s[i] in usedChar and start <= usedChar[s[i]]:
                start = usedChar[s[i]] + 1
            else:
                maxLength = max(maxLength, i - start + 1)

            usedChar[s[i]] = i

        return maxLength


if __name__ == '__main__':
    print(func_runtime(Solution().lengthOfLongestSubstring,
                       10 ** 4, "abcabcbb"))
    print(func_runtime(Solution1().lengthOfLongestSubstring,
                       10 ** 4, "abcabcbb"))
    print(func_runtime(Solution2().lengthOfLongestSubstring,
                       10 ** 4, "abcabcbb"))
    print(func_runtime(Solution3().lengthOfLongestSubstring,
                       10 ** 4, "fabcaffcbb"))
