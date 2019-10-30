<span id="re_"></span>
# LeetCode

[1、两数之和](#两数之和)  
[2、两数相加](#两数相加)  
[3、无重复字符的最长子串](#无重复字符的最长子串)  
[2、两个排序数组的中位数](#两个排序数组的中位数)  
[5、最长回文子串](#最长回文子串)  
[6、Z字形变换](#Z字形变换)  
[7、整数反转](#整数反转)  
[8、字符串转换整数（atoi）](#字符串转换整数（atoi）)  
[9、回文数](#回文数)  
[10、正则表达式匹配](#正则表达式匹配)  
[11、盛最多水的容器](#盛最多水的容器)  
[12、整数转罗马数字](#整数转罗马数字)  
[14、最长公共前缀](#最长公共前缀)  
[69、X的平方根](#X的平方根)    
[386、字典序排数](#字典序排数)

<span id="两数之和"></span>
## [1、两数之和](#re_)
```cpp
给定一个整数数组 nums 和一个目标值 target，
请你在该数组中找出和为目标值的那两个整数，并返回他们的数组下标。

给定 nums = [2, 7, 11, 15], target = 9
因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int len = nums.size();
        map<int, int> mp;
        vector<int> vec(2,0);
        for(int i = 0; i < len; i++){
            int other = target - nums[i];
            if(mp.count(other) > 0 && mp[other] != i){
                vec[0] = mp[other];
                vec[1] = i;
                break;
            }
            mp[nums[i]] = i;
        }
        return vec;
    }
};
```

<span id="两数相加"></span>
## [2、两数相加](#re_)
```cpp
给出两个非空的链表用来表示两个非负的整数。
其中，它们各自的位数是按照逆序的方式存储的，并且它们的每个节点只能存储一位数字。
如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
您可以假设除了数字0之外，这两个数都不会以0开头。

输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807

// 各个节点相加，当大于9时产生进位1,加在下一个节点中
// 最后一组节点若产生进位，则需要再次新增一个节点值为1；没有进位则结束
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode* L = new ListNode(0);
        ListNode* result = L;
        int flag = 0;
        while(l1 != NULL || l2 != NULL){
            int sum_ = 0;
            if(l1){
                sum_ += l1->val;
                l1 = l1->next;
            }
            if(l2){
                sum_ += l2->val;
                l2 = l2->next;
            }
            if(flag) sum_+=1;
            L->next = new ListNode(sum_ % 10);
            if(sum_ > 9) flag = 1;
            else flag = 0;
            L = L->next;
        }
        if(flag) L->next = new ListNode(1);
        return result->next;
    }
};
```


<span id="无重复字符的最长子串"></span>
## [3、无重复字符的最长子串](#re_)
```cpp
给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。

输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。

输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        // 双指针; aim,end
        if(s.size() == 0 || s.size() == 1) return s.size();
        int p = 0, max_len = 1;
        for(int end = 1;end < s.size(); end++){
            for(int aim = p; aim < end; aim++){
                if(s[aim] == s[end]){
                    p = aim + 1;
                    if(max_len < end - aim) max_len = end - aim;
                    break;
                }
            }
            if(max_len < end - p + 1) max_len = end - p + 1;
        }
        return max_len;
    }
};
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
    // 中心扩散
    string longestPalindrome(string s) {
        int len_1, len_2, len_ = 1;
        int start = 0, end = 0;
        for(int i = 0; i < s.size(); i++){
            len_1 = expandAroundCenter(s, i, i);
            len_2 = expandAroundCenter(s, i, i + 1);
            len_ = max(len_, max(len_1, len_2));
            if(len_ > end - start + 1){
                start = i - (len_ - 1) / 2;
                end  = i + len_ / 2;
            }   
        }
        return s.substr(start, len_);
    }
    
private:
    int expandAroundCenter(string s, int low, int high){
        while(low >= 0 && high < s.size()){
            if(s[low] == s[high]){
                low--;
                high++;
            }
            else break;
        }
        return high - low - 1;
    }
};
```

<span id="Z字形变换"></span>
## [6、Z字形变换](#re_)
```cpp
将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。
比如输入字符串为 "LEETCODEISHIRING" 行数为 3 时，排列如下：
L   C   I   R
E T O E S I I G
E   D   H   N
之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："LCIRETOESIIGEDHN"。

输入: s = "LEETCODEISHIRING", numRows = 4
输出: "LDREOEIIECIHNTSG"
解释:
L     D     R
E   O E   I I
E C   I H   N
T     S     G

class Solution {
public:
    string convert(string s, int numRows) {
        if(numRows == 1) return s;
        vector<string> res(min(int(s.size()), numRows));
        bool direction = false;
        int cur_row= 0;
        for(char s_ : s){
            res[cur_row] += s_;
            if(cur_row == 0 || cur_row == numRows - 1) direction = !direction;
            cur_row += direction ? 1 : -1;
        }
        string str;
        for(string res_: res) str += res_;
        return str;
    }
};
```

<span id="整数反转"></span>
## [7、整数反转](#re_)
```cpp
给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。

输入: 123
输出: 321

输入: -123
输出: -321

class Solution {
public:
    int reverse(int x) {
        // 考虑溢出问题
        int ret = 0;
        while(x){
            int pop = x % 10;
            x = x / 10;
            if(ret > INT_MAX / 10 || (ret == INT_MAX / 10 && pop > 7)) return 0;  // 正数溢出
            if(ret < INT_MIN / 10 || (ret == INT_MIN / 10 && pop < -8 )) return 0;  // 负数溢出
            ret = ret * 10 + pop;
        }
        return ret;
    }
};
```

<span id="字符串转换整数（atoi）"></span>
## [8、字符串转换整数（atoi）](#re_)
```cpp
输入: "42"
输出: 42

输入: "4193 with words"
输出: 4193
解释: 转换截止于数字 '3' ，因为它的下一个字符不为数字。

输入: "words and 987"
输出: 0
解释: 第一个非空字符是 'w', 但它不是数字或正、负号。因此无法执行有效的转换。
     
输入: "-91283472332"
输出: -2147483648
解释: 数字 "-91283472332" 超过 32 位有符号整数范围。因此返回 INT_MIN (−231) 。

class Solution {
public:
    int myAtoi(string str) {
        int flag = 1, i = 0,res = 0;
        while (str[i] == ' ') i++;
        if (str[i] == '-') flag = -1; 
        if (str[i] == '-' || str[i] == '+') i++; 
        while (i < str.size() && isdigit(str[i])){
            int r = str[i] - '0';
            if (res > INT_MAX / 10 || (res == INT_MAX / 10 && r > 7)) {  // 处理溢出
                return flag > 0 ? INT_MAX : INT_MIN;
            }
            res = 10 * res + r;
            i++;
        }
        return flag > 0 ? res : -res;
    }
};
```

<span id="回文数"></span>
## [9、回文数](#re_)
```cpp
输入: 121
输出: true

输入: -121
输出: false
解释: 从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。

输入: 10
输出: false
解释: 从右向左读, 为 01 。因此它不是一个回文数。

class Solution {
public:
    bool isPalindrome(int x) {
        if(x < 0 || (x != 0 && x % 10 == 0)) return false;
        int expect_num = 0;
        // 反转一半数字
        while(x > expect_num){
            expect_num = expect_num * 10 + x % 10;
            x = x / 10;
        }
        return x == expect_num || x == expect_num / 10;
    }
};
```

<span id="正则表达式匹配"></span>
## [10、正则表达式匹配](#re_)
```cpp
给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。
'.' 匹配任意单个字符
'*' 匹配零个或多个前面的那一个元素

输入:
s = "aab"
p = "c*a*b"
输出: true
解释: 因为 '*' 表示零个或多个，这里 'c' 为 0 个, 'a' 被重复一次。因此可以匹配字符串 "aab"。

class Solution {
public:
    vector<vector<int>> f;
    int n, m;
    bool isMatch(string s, string p) {
        n = s.size();
        m = p.size();
        f = vector<vector<int>> (n + 1, vector<int>(m + 1, -1));
        return dp(0, 0, s, p);
    }
    
    // 动态规划 
    bool dp(int x, int y, string &s, string &p) {
        if (f[x][y] != -1) return f[x][y];
        if(y == m){
            return f[x][y] = x == n;
        }
        bool first_con = (x < n) && (s[x] == p[y] || p[y] == '.');
        bool ans;
        if (y + 1 < m && p[y + 1] == '*') {
            ans = dp(x, y + 2, s, p) || first_con && dp(x + 1, y, s, p);
        }
        else {
            ans = first_con  && dp(x + 1, y + 1, s, p);
        }
        return f[x][y] = ans;
    }
};
```

<span id="盛最多水的容器"></span>
## [11、盛最多水的容器](#re_)
```cpp
给定n个非负整数a1，a2，...，an，每个数代表坐标中的一个点(i, ai)。
在坐标内画n条垂直线，垂直线i的两个端点分别为(i, ai)和(i, 0)。
找出其中的两条线，使得它们与x轴共同构成的容器可以容纳最多的水。
说明：你不能倾斜容器，且 n 的值至少为 2。

class Solution {
public:
    int maxArea(vector<int>& height) {
        int i = 0, j = height.size() - 1;
        int area = (int(height.size()) - 1) * min(height[i], height[j]);
        while (i < j) {
            area = max(area, (j - i) * min(height[i], height[j]));
            if (height[i] < height[j]) i++;
            else j--;
        }
        return area;
    }
};
```

<span id="整数转罗马数字"></span>
## [12、整数转罗马数字](#re_)
```cpp
罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。

字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 
27 写做  XXVII, 即为 XX + V + II 。
通常情况下，罗马数字中小的数字在大的数字的右边。
但也存在特例，例如 4 不写做 IIII，而是 IV。
数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。
同样地，数字 9 表示为 IX。

输入: 3
输出: "III"

输入: 58
输出: "LVIII"
解释: L = 50, V = 5, III = 3.

class Solution {
public:
    string intToRoman(int num) {
        map<int, string> mp;
        mp[1] = "I";
        mp[4] = "IV";
        mp[5] = "V";
        mp[9] = "IX";
        mp[10] = "X";
        mp[40] = "XL";
        mp[50] = "L";
        mp[90] = "XC";
        mp[100] = "C";
        mp[400] = "CD";
        mp[500] = "D";
        mp[900] = "CM";
        mp[1000] = "M";    
        int nums[13] = {1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000};
        int t = 12;
        string str="";
        while (num) {
            if (num >= nums[t]) {
                str += mp[nums[t]];
                num -= nums[t];
            }
            else t--;
        }
        return str;
    }
};
```

<span id="最长公共前缀"></span>
## [14、最长公共前缀](#re_)
```py
"""
编写一个函数来查找字符串数组中的最长公共前缀。
如果不存在公共前缀，返回空字符串 ""。

示例 1:
输入: ["flower","flow","flight"]
输出: "fl"

示例 2:
输入: ["dog","racecar","car"]
输出: ""
"""


class Solution(object):
    # 方法1
    def longestCommonPrefix1(self, strs):
        """
        :param strs: List[str]
        :return: str
        """
        len_str = len(strs)
        if len_str == 0:
            return ''
        list_len = []
        for i in range(len_str):
            list_len.append(len(strs[i]))
        list_len.sort()
        # 取出最短的子串
        # 我这里是直接取第一个子串的前min_len
        min_len = list_len[0]
        b0 = strs[0][0:min_len]  # 最短的子串
        com = b0
        for s in strs:
            for j in range(list_len[0]):
                if s[j] != com[j]:
                    # 判断到有不等的地方
                    a0 = s[0:j]
                    if len(b0) >= len(a0):  # 上一个最长公共前缀是否比现在长
                        b0 = a0
        return b0

    # 方法2:高级
    def longestCommonPrefix2(self, strs):
        res = ""
        if len(strs) == 0:
            return ""
        for each in zip(*str):
            if (len(set(each))) == 1:
                res += each[0]
            else:
                return res
        return res


str = ["flower", "flow", "flight"]
s = Solution()
result = s.longestCommonPrefix2(str)
print("result:", result)
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

// 扩展，给定精度
double getSqrt(int x,double precision) {
	 double left = 0, right = x;
	 while (1) {
		 double mid = left + (right - left) / 2;
		 if (abs(x /mid - mid) < precision)	return mid;
		 else if (x / mid > mid)	left = mid + 1;
		 else right = mid - 1;
	 }
}
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
