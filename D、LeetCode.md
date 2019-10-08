<span id="re_"></span>
# LeetCode

[1、两数之和](#两数之和)  
[2、两个排序数组的中位数](#两个排序数组的中位数)  
[5、最长回文子串](#最长回文子串)  
[69、X的平方根](#X的平方根)   
[386、字典序排数](#字典序排数)

<span id="两数之和"></span>
## [1、两数之和](#re_)
```py
"""
题目：
# 给定一个整数数组和一个目标值，找出数组中和为目标值的两个数。
# 你可以假设每个输入只对应一种答案，且同样的元素不能被重复利用。

# 给定 nums = [2, 7, 11, 15], target = 9
# 因为 nums[0] + nums[1] = 2 + 7 = 9
# 所以返回 [0, 1]
"""


class Solution(object):
    def twoSum(self, nums, target):
        """
        :param nums: List[int]
        :param target: int
        :return: List[int]
        """
        l = len(nums)
        dict_ = {nums[i]: i for i in range(l)}
        for a in range(l):
            other_num = target - nums[a]
            if other_num in dict_ and a != dict_[other_num]:
                return [a, dict_[other_num]]


nums = [2, 7, 11, 15]
target = 9
s = Solution()
result = s.twoSum(nums, target)
print("result", result)

"""
result [0, 1]
"""
```

<span id="两个排序数组的中位数"></span>
## [2、两个排序数组的中位数](#re_)
```py
"""
给定两个大小为 m 和 n 的有序数组 nums1 和 nums2 。
请找出这两个有序数组的中位数。 要求算法的时间复杂度为 O(log (m+n))  。
你可以假设 nums1 和 nums2 不同时为空。

# 示例1
nums1 = [1, 3]
nums2 = [2]
中位数是 2.0

# 示例2
nums1 = [1, 2]
nums2 = [3, 4]
中位数是 (2 + 3)/2 = 2.5
"""


class Solution(object):
    def findMedianSortArrays(self, nums1, nums2):
        nums = nums1 + nums2
        nums.sort()
        mid = len(nums) // 2  # 整除
        if mid % 2 == 1:
            return float(nums[mid])
        else:
            return (nums[mid - 1] + nums[mid]) / 2.0


nums1 = [1, 2]
nums2 = [3, 4]
s = Solution()
result = s.findMedianSortArrays(nums1, nums2)
print("result", result)

"""
result 2.5
"""
```

<span id="最长回文子串"></span>
## [5、最长回文子串](#re_)
```py
"""
给定一个字符串 s，找到 s 中最长的回文子串。
你可以假设 s 的最大长度为1000。

示例 1：
输入: "babad"
输出: "bab"
注意: "aba"也是一个有效答案。
示例 2：
输入: "cbbd"
输出: "bb"
"""


class Solution(object):
    longest_s = ""  # 最长回文子串
    maxLen = 0  # 长度

    def longestPalindrome(self, s):
        """
        :param s: str
        :return: str
        """
        len_s = len(s)
        if len_s == 1:
            return s
        for i in range(len_s):
            # 单核
            self.find_longest_Palindrome(s, i, i)
            self.find_longest_Palindrome(s, i, i + 1)
        return self.longest_s

    def find_longest_Palindrome(self, s, low, high):
        # 从中间向两端延伸
        while low >= 0 and high < len(s):
            if s[low] == s[high]:
                low -= 1
                high += 1
            else:
                break
        # high - low - 1表示当前字符串长度
        if high - low - 1 >= self.maxLen:
            self.maxLen = high - low - 1
            self.longest_s = s[low + 1:high]


str = "cbbd"
s = Solution()
result = s.longestPalindrome(str)
print("result:", result)

"""
result: bb
"""
```

```cpp
class Solution {
public:
    string longestPalindrome(string s) {
        const int len = s.size();
        if(len <= 1)return s;
        int start, maxLen = 0;
        for(int i = 1; i < len; i++)
        {
            //寻找以i-1,i为中点偶数长度的回文
            int low = i-1, high = i;
            while(low >= 0 && high < len && s[low] == s[high])
            {
                low--;
                high++;
            }
            if(high - low - 1 > maxLen)
            {
                maxLen = high - low -1;
                start = low + 1;
            }
             
            //寻找以i为中心的奇数长度的回文
            low = i- 1; high = i + 1;
            while(low >= 0 && high < len && s[low] == s[high])
            {
                low--;
                high++;
            }
            if(high - low - 1 > maxLen)
            {
                maxLen = high - low -1;
                start = low + 1;
            }
        }
        return s.substr(start, maxLen);
    }
};
```

<span id="X的平方根"></span>
## [69、X的平方根](#re_)
```cpp
// 二分查找
class Solution {
public:
    int mySqrt(int x) {
        if (x <= 1) return x;
        int left = 0, right = x;
        while (left < right) {
            int mid = left + right >> 1;
            if (x / mid >= mid) left = mid + 1;
            else right = mid;
        }
        return right - 1;
    }
};
```

<span id="字典序排数"></span>
## [386、字典序排数](#re_)
```
题目：
给定一个整数 n, 返回从 1 到 n 的字典顺序。

例如：
给定 n =1 3，返回 [1,10,11,12,13,2,3,4,5,6,7,8,9] 。
```
```cpp
class Solution {
public:
    vector<int> lexicalOrder(int n) {
        vector<int> res(n);
        int cur = 1;
        for (int i = 0; i < n; i++) {
            res[i] = cur;
            if (cur * 10 <= n) cur *= 10;
            else {
                if (cur >= n) cur /= 10;
                cur += 1;
                while (cur % 10 == 0) cur /= 10;
            }
        }
        return res;
    }
};
```
