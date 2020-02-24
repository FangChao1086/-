<span id="back"></span>
# LeetCode_CPP版题目及答案
||||||
|-|-|-|-|-|
|[1、两数之和](#两数之和)|[2、两数相加](#两数相加) |[3、无重复字符的最长子串](#无重复字符的最长子串)| [4、寻找两个有序数组的中位数](#寻找两个有序数组的中位数)  |[5、最长回文子串](#最长回文子串) | 
|[6、Z字形变换](#Z字形变换)  |[7、整数反转](#整数反转)  |[8、字符串转换整数(atoi)](#字符串转换整数（atoi）)  |[9、回文数](#回文数)  |[10、正则表达式匹配](#正则表达式匹配)  |
|[11、盛最多水的容器](#盛最多水的容器)  |[12、整数转罗马数字](#整数转罗马数字)  |[13、罗马数字转整数](#罗马数字转整数)  |[14、最长公共前缀](#最长公共前缀)  |[15、三数之和](#三数之和)  |
|[16、最接近的三数之和](#最接近的三数之和)  |[17、电话号码的字母组合](#电话号码的字母组合)  |[18、四数之和](#四数之和)  |[19、删除链表的倒数第N个节点](#删除链表的倒数第N个节点)  |[20、有效的括号](#有效的括号)  |
|[21、合并两个有序链表](#合并两个有序链表)  |[22、括号生成](#括号生成)  |[23、合并K个排序链表](#合并K个排序链表)  |[24、两两交换链表中的节点](#两两交换链表中的节点)  |[25、K 个一组翻转链表](#K个一组翻转链表)  
|[26、删除排序数组中的重复项](#删除排序数组中的重复项)  |[27、移除元素](#移除元素)  |[28、实现str()](#实现str())  |[29、两数相除](#两数相除)  |[30、串联所有单词的子串](#串联所有单词的子串)  
|[31、下一个排列](#下一个排列)  |[32、最长有效括号](#最长有效括号)  |[33、搜索旋转排序数组](#搜索旋转排序数组)  |[34、排序数组查找元素的第一和最后一个位置](#排序数组查找元素的第一和最后一个位置)  |[35、搜索插入位置](#搜索插入位置)  
|[36、有效的数独](#有效的数独)  |[37、解数独](#解数独)  |[38、报数](#报数)  |[39、组合总和](#组合总和)  |[40、组合总和 II](#组合总和2)  
|[41、缺失的第一个正数](#缺失的第一个正数)  |[42、接雨水](#接雨水)  |[43、字符串相乘](#字符串相乘)  |[44、通配符匹配](#通配符匹配)  |[45、跳跃游戏 II](#跳跃游戏2)  
|[46、全排列](#全排列)  |[47、全排列 II](#全排列2)  |[48、旋转图像](#旋转图像)  |[49、 字母异位词分组](#字母异位词分组)  |[50、Pow(x, n)](#Pow)  
|[51、N皇后](#N皇后)  |[52、N皇后 II](#N皇后2)  |[53、最大子序和](#最大子序和)  |[54、螺旋矩阵](#螺旋矩阵)  |[55、跳跃游戏](#跳跃游戏)  
|[56、合并区间](#合并区间)  |[57、插入区间](#插入区间)  |[58、最后一个单词的长度](#最后一个单词的长度)  |[59、螺旋矩阵 II](#螺旋矩阵2)  |[60、第k个排列](#第k个排列)  
|[61、旋转链表](#旋转链表)  |[62、不同路径](#不同路径)  |[63、不同路径 II](#不同路径2)  |[64、最小路径和](#最小路径和)  |[65、有效数字](#有效数字)  
|[66、加一](#加一)  |[67、二进制求和](#二进制求和)  |[68、文本左右对齐](#文本左右对齐)  |[69、X的平方根](#X的平方根)  |[70、爬楼梯](#爬楼梯)  
|[71、简化路径](#简化路径)  |[72、编辑距离](#编辑距离)  |[73、矩阵置零](#矩阵置零)  |[74、搜索二维矩阵](#搜索二维矩阵)  |[75、颜色分类](#颜色分类)  
|[76、最小覆盖子串](#最小覆盖子串)  |[77、组合](#组合)  |[78、子集](#子集)  |[79、单词搜索](#单词搜索)  |[80、删除排序数组中的重复项 II](#删除排序数组中的重复项2)  
|[81、搜索旋转排序数组 II](#搜索旋转排序数组2)  |[82、删除排序链表中的重复元素 II](#删除排序链表中的重复元素2)  |[83、删除排序链表中的重复元素](#删除排序链表中的重复元素)  |[84、柱状图中最大的矩形](#柱状图中最大的矩形)  |[85、最大矩形](#最大矩形)  
|[86、分隔链表](#分隔链表)  |[87、扰乱字符串](#扰乱字符串)  |[88、合并两个有序数组](#合并两个有序数组)  |[89、格雷编码](#格雷编码)  |[90、子集 II](#子集2)  
|[91、解码方法](#解码方法)  |[92、反转链表 II](#反转链表2)  |[93、复原IP地址](#复原IP地址)  |[94、二叉树的中序遍历](#二叉树的中序遍历)  |[95、不同的二叉搜索树 II](#不同的二叉搜索树2)  
|[96、不同的二叉搜索树](#不同的二叉搜索树)  |[97、交错字符串](#交错字符串)  |[98、验证二叉搜索树](#验证二叉搜索树)  |[99、恢复二叉搜索树](#恢复二叉搜索树)  |[100、相同的树](#相同的树)  
|[101、对称二叉树](#对称二叉树)  |[102、二叉树的层次遍历](#二叉树的层次遍历)  |[103、二叉树的锯齿形层次遍历](#二叉树的锯齿形层次遍历)  |[104、二叉树的最大深度](#二叉树的最大深度)  |[105、从前序与中序遍历序列构造二叉树](#从前序与中序遍历序列构造二叉树)  
|[106、从中序与后序遍历序列构造二叉树](#从中序与后序遍历序列构造二叉树)  |[107、二叉树的层次遍历 II](#二叉树的层次遍历2)  |[108、将有序数组转换为二叉搜索树](#将有序数组转换为二叉搜索树)  |[109、有序链表转换二叉搜索树](#有序链表转换二叉搜索树)  |[110、平衡二叉树](#平衡二叉树)  
|[111、二叉树的最小深度](#二叉树的最小深度)  |[112、路径总和](#路径总和)  |[113、路径总和 II](#路径总和2)  |[114、二叉树展开为链表](#二叉树展开为链表)  |[115、不同的子序列](#不同的子序列)  
|[116、填充每个节点的下一个右侧节点指针](#填充每个节点的下一个右侧节点指针)  |[117、填充每个节点的下一个右侧节点指针 II](#填充每个节点的下一个右侧节点指针2)  |[118、杨辉三角](#杨辉三角)|[119、杨辉三角 II](#杨辉三角2)|[120、三角形最小路径和](#三角形最小路径和)|
|[121、买卖股票的最佳时机(easy)](#买卖股票的最佳时机)|
|[386、字典序排数](#字典序排数)  

<span id="两数之和"></span>
## [1、两数之和](#back)
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
## [2、两数相加](#back)
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
## [3、无重复字符的最长子串](#back)
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

<span id="寻找两个有序数组的中位数"></span>
## [4、两个有序数组的中位数](#back)
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
```cpp
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        // 参考：https://leetcode-cn.com/problems/median-of-two-sorted-arrays/solution/4-xun-zhao-liang-ge-you-xu-shu-zu-de-zhong-wei-shu/
        int n = nums1.size();
        int m = nums2.size();
        if (n > m) return findMedianSortedArrays(nums2, nums1);
        int L_max1, L_max2, R_min1, R_min2, c1, c2, lo = 0, hi = 2 * n;
        while (lo <= hi) {
            c1 = lo + hi >> 1;  // 二分
            c2 = n + m - c1;

            L_max1 = (c1 == 0) ? INT_MIN : nums1[(c1 - 1) / 2];
            R_min1 = (c1 == 2 * n) ? INT_MAX : nums1[c1 / 2];
            L_max2 = (c2 == 0) ? INT_MIN : nums2[(c2 - 1) / 2];
            R_min2 = (c2 == 2 * m) ? INT_MAX : nums2[c2 / 2];

            if (L_max1 > R_min2) hi = c1 - 1;
            else if (L_max2 > R_min1) lo = c1 + 1;
            else break;  // L_max1 <= R_min2 && L_max2 <= R_min1 时停止
        } 
        return (max(L_max1, L_max2) + min(R_min1, R_min2)) / 2.0;
    }
};
```

<span id="最长回文子串"></span>
## [5、最长回文子串](#back)
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
## [6、Z字形变换](#back)
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
## [7、整数反转](#back)
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
## [8、字符串转换整数（atoi）](#back)
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
## [9、回文数](#back)
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
## [10、正则表达式匹配](#back)
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
## [11、盛最多水的容器](#back)
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
## [12、整数转罗马数字](#back)
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

<span id="罗马数字转整数"></span>
## [13、罗马数字转整数](#back)
```cpp
罗马数字包含以下七种字符: I， V， X， L，C，D 和 M。
字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000

输入: "III"
输出: 3

输入: "LVIII"
输出: 58
解释: L = 50, V= 5, III = 3

class Solution {
public:
    int romanToInt(string s) {
        int total[256];
        total['I'] = 1;
        total['V'] = 5;
        total['X'] = 10;
        total['L'] = 50;
        total['C'] = 100;
        total['D'] = 500;
        total['M'] = 1000;
        int num = 0;
        for (int i = 0; i < s.size(); i++){
            if((i == s.size() - 1) || total[s[i]] >= total[s[i + 1]]) {
                num += total[s[i]];
            }
            else 
                num -= total[s[i]];
        }
        return num;
    }
};
```

<span id="最长公共前缀"></span>
## [14、最长公共前缀](#back)
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
```cpp
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        int len_ = strs.size();
        int count_ = INT_MAX;
        string result = "";
        for (int i = 0; i < len_; i++) {
            if(count_ >= strs[i].size()) count_ = strs[i].size();
        }
        for (int i = 0; i < count_; i++){
            int j;
            for(j = 1; j < len_; j++){
                if(strs[0][i] == strs[j][i]) continue;
                else break;
            }
            if (j == len_) result += strs[0][i];
            else break;
        }
        return result;
    }
};
```

<span id="三数之和"></span>
## [15、三数之和](#back)
```cpp
给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。
注意：答案中不可以包含重复的三元组。
例如, 给定数组 nums = [-1, 0, 1, 2, -1, -4]，
满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]

class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int len_ = nums.size();
        int left,right,i;
        vector<vector<int>> res;
        for (i = 0; i < len_ - 2; i++) {
            if (nums[i] > 0) break;
            if (i > 0 && nums[i - 1] == nums[i]) continue;
            left = i + 1, right = len_ - 1;
            while(left < right){
                if (nums[i] + nums[left] + nums[right] == 0) {
                    res.push_back({nums[i], nums[left], nums[right]});
                    left++;
                    while (left < right && nums[left] == nums[left-1]) left++;
                    right--;
                    while (left < right && nums[right] == nums[right+1]) right--;
                }
                else if(nums[i] + nums[left] + nums[right] < 0) left++;
                else right--;
            }
        }
        return res;
    }
};
```

<span id="最接近的三数之和"></span>
## [16、最接近的三数之和](#back)
```cpp
给定一个包括 n 个整数的数组 nums 和 一个目标值 target。
找出 nums 中的三个整数，使得它们的和与 target 最接近。
返回这三个数的和。假定每组输入只存在唯一答案。

例如，给定数组 nums = [-1，2，1，-4], 和 target = 1.
与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).

class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        // 和与目标值接近
        sort(nums.begin(), nums.end());
        int close_with_target = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.size() - 2; i++) {
            int digit1 = i;
            int digit2 = i + 1;
            int digit3 = nums.size() - 1;
            int tmp = nums[digit1] + nums[digit2] + nums[digit3];
            while (digit2 < digit3) {
                if (abs(tmp - target) < abs(close_with_target - target)) {
                    close_with_target = tmp;
                }
                int dif = target - tmp;
                if (dif == 0) return target;
                if (dif > 0) {
                    digit2++;
                }
                else digit3--;
                tmp = nums[digit1] + nums[digit2] + nums[digit3];
            }
        }
        return close_with_target;
    }
};
```

<span id="电话号码的字母组合"></span>
## [17、电话号码的字母组合](#back)
```cpp
给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

输入："23"
输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].

class Solution {
public:
    vector<string> letterCombinations(string digits) {
        vector<string> res;
        map<char, string> mp = {{'2', "abc"}, {'3', "def"}, {'4', "ghi"},
                                {'5', "jkl"}, {'6', "mno"}, {'7', "pqrs"},
                                {'8', "tuv"}, {'9', "wxyz"}};
        int size_ = digits.size();
        queue<string> que;
        for (int i = 0; i < mp[digits[0]].size(); i++) {
            string str;
            str.push_back(mp[digits[0]][i]);  // char转string
            que.push(str);
        }
        for (int i = 1; i < size_; i++) {
            int len_ = que.size();
            while (len_--) {
                for (int j = 0; j < mp[digits[i]].size(); j++) {
                    string s_front = que.front();
                    s_front += mp[digits[i]][j];
                    que.push(s_front);
                }
                que.pop();
            }
        }
        while (!que.empty()) {
            res.push_back(que.front());
            que.pop();
        }
        return res;
    }
};
```

<span id="四数之和"></span>
## [18、四数之和](#back)
```cpp
给定一个包含 n 个整数的数组 nums 和一个目标值 target，
判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？
找出所有满足条件且不重复的四元组。
注意：答案中不可以包含重复的四元组。

给定数组 nums = [1, 0, -1, 0, -2, 2]，和 target = 0。
满足要求的四元组集合为：
[
  [-1,  0, 0, 1],
  [-2, -1, 1, 2],
  [-2,  0, 0, 2]
]

思路：
使用四个指针(a<b<c<d)。固定最小的a和b在左边，c=b+1,d=_size-1 移动两个指针包夹求解。
 保存使得nums[a]+nums[b]+nums[c]+nums[d]==target的解。偏大时d左移，偏小时c右移。c和d相
 遇时，表示以当前的a和b为最小值的解已经全部求得。b++,进入下一轮循环b循环，当b循环结束后。
 a++，进入下一轮a循环。 即(a在最外层循环，里面嵌套b循环，再嵌套双指针c,d包夹求解)

class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        int size_ = nums.size();
        for (int a = 0; a < size_ - 3; a++) {
            if(a > 0 && nums[a] == nums[a - 1]) continue;
            for (int b = a + 1; b < size_ - 2; b++) {
                if (b > a + 1 && nums[b] == nums[b - 1]) continue;
                int c = b + 1;
                int d = size_ - 1;
                while (c < d) {
                    if (nums[a] + nums[b] + nums[c] + nums[d] < target) {
                        c++;
                    }
                    else if (nums[a] + nums[b] + nums[c] + nums[d] > target) {
                        d--;
                    }
                    else {
                        res.push_back({nums[a], nums[b], nums[c], nums[d]});
                        c++;
                        while (c < d && nums[c - 1] == nums[c]) c++;
                        d--;
                        while (c < d && nums[d + 1] == nums[d]) d--;
                    }
                }
            }
        }
        return res;
    }
};
```

<span id="删除链表的倒数第N个节点"></span>
## [19、删除链表的倒数第N个节点](#back)
```cpp
给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。

给定一个链表: 1->2->3->4->5, 和 n = 2.
当删除了倒数第二个节点后，链表变为 1->2->3->5.

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
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* p1 = head;
        ListNode* p2 = head;
        int i;
        for (i = 0; p1 != NULL; i++) {
            if (i > n) p2 = p2 -> next;
            p1 = p1 -> next;
        }
        if (i == n) return head -> next;
        ListNode* p3 = p2 -> next;
        p2 -> next = p3 -> next;
        return head;
    }
};
```

<span id="有效的括号"></span>
## [20、有效的括号](#back)
```cpp
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
有效字符串需满足：
左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
注意空字符串可被认为是有效字符串。

输入: "([)]"
输出: false
输入: "{[]}"
输出: true

class Solution {
public:
    bool isValid(string s) {
        stack<int> stk;
        int len_ = s.size();
        for (int i = 0; i < len_; i++) {
            if (s[i] == '(' || s[i] == '[' || s[i] == '{') {
                stk.push(s[i]);
            }
            else{
                if (stk.empty()) return false;
                else {
                    if (s[i] == ')' && stk.top() != '(') return false;
                    if (s[i] == ']' && stk.top() != '[') return false;
                    if (s[i] == '}' && stk.top() != '{') return false;
                    stk.pop();
                }
            }
        }
        return stk.empty();
    }
};
```

<span id="合并两个有序链表"></span>
## [21、合并两个有序链表](#back)
```cpp
将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4

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
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* head = new ListNode(0);
        ListNode* p = head;
        if (l1 == NULL) return l2;
        if (l2 == NULL) return l1;
        while (l1 && l2) {
            if (l1 -> val <= l2 -> val) {
                head -> next = l1;
                l1 = l1 -> next;
            }
            else {
                head -> next = l2;
                l2 = l2 -> next;
            }
            head = head -> next;
        }
        if (l1) {
            head -> next = l1;
        }
        else {
            head -> next = l2;
        }
        return p -> next;
    }
};
```

<span id="括号生成"></span>
## [22、括号生成](#back)
```cpp
给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。

例如，给出 n = 3，生成结果为：
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]

class Solution {
public:
    // 回溯法
    vector<string> generateParenthesis(int n) {
        vector<string> res;
        backTrack(res, "", n, 0);
        return res; 
    }

    // n:可用的左括号(的个数;   index:可用的右括号)的个数;
    void backTrack(vector<string> &res, string track, int n, int index){
        if (n == 0 && index == 0) res.push_back(track);
        else {
            if (n > 0) backTrack(res, track + "(", n - 1, index + 1);  // 使用一个左括号，就会生成一个右括号要使用
            if (index > 0) backTrack(res, track + ")", n, index - 1);  // 使用右括号
        }

    }
};
```

<span id="合并K个排序链表"></span>
## [23、合并K个排序链表](#back)
```cpp
合并 k 个排序链表，返回合并后的排序链表。请分析和描述算法的复杂度。

输入:
[
  1->4->5,
  1->3->4,
  2->6
]
输出: 1->1->2->3->4->4->5->6

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
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        // merge两两合并
        int size_ = lists.size();
        if (size_ == 0) return NULL;
        if (size_ == 1) return lists[0];
        ListNode* p = lists[0];
        for (int i = 1; i < size_; i++) {
            p = merge(p, lists[i]); 
        }
        return p;
    }

    ListNode* merge(ListNode* L1, ListNode* L2){
        ListNode* Head = new ListNode(0);
        ListNode* pHead = Head;
        while (L1 && L2) {
            if (L1 -> val < L2 -> val) {
                Head -> next = L1;
                L1 = L1 -> next;
            }
            else {
                Head -> next = L2;
                L2 = L2 -> next;
            }
            Head = Head -> next;
        }
        Head -> next = L1 ? L1 : L2;
        return pHead -> next;
    }
};
```

<span id="两两交换链表中的节点"></span>
## [24、两两交换链表中的节点](#back)
```cpp
给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

给定 1->2->3->4, 你应该返回 2->1->4->3.

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
    ListNode* swapPairs(ListNode* head) {
        // // 递归
        // if(head==NULL || head->next==NULL) return head;
        // ListNode* pNext = head->next;
        // head->next = swapPairs(pNext->next);
        // pNext->next = head;
        // return pNext;

        // 非递归
        ListNode* p = new ListNode(0);
        p->next = head;
        ListNode* tmp = p; 
        while (tmp->next != NULL && tmp->next->next !=NULL) {
            ListNode* start = tmp->next;
            ListNode* end = tmp->next->next;
            start->next = end->next;
            end->next = start;
            tmp->next = end;
            tmp = start;
        }
        return p->next;
    }
};
```

<span id="K个一组翻转链表"></span>
## [25、K 个一组翻转链表](#back)
```cpp
给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。
k 是一个正整数，它的值小于或等于链表的长度。
如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

给定这个链表：1->2->3->4->5
当 k = 2 时，应当返回: 2->1->4->3->5
当 k = 3 时，应当返回: 3->2->1->4->5

说明 :
你的算法只能使用常数的额外空间。
你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

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
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* pre = dummy;
        ListNode* end = dummy;
        while (end->next) {
            for (int i = 0; i < k && end!=NULL; i++) end = end->next;
            if (end == NULL) break;
            ListNode* start = pre->next;
            ListNode* next_ = end->next;
            end->next = NULL;
            pre->next = reverse(start);
            start->next = next_;
            pre = start;
            end = start;
        }
        return dummy->next;   
    }

    ListNode* reverse(ListNode* head){
        ListNode* p = head;
        ListNode* pre = NULL;
        while (head) {
            ListNode* tmp = head->next;
            head->next = pre;
            pre = head;
            head = tmp;
        }
        return pre;
    }
};
```

<span id="删除排序数组中的重复项"></span>
## [26、删除排序数组中的重复项](#back)
```cpp
给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。

给定 nums = [0,0,1,1,1,2,2,3,3,4],
函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。
你不需要考虑数组中超出新长度后面的元素。

class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        // 数组已经排完序，双指针
        int len_ = nums.size();
        if (len_ == 0) return 0;
        int i = 0;
        for (int j = 1; j < len_; j++) {
            if (nums[i] != nums[j]) {
                i++;
                nums[i] = nums[j];
            }
        }
        return i + 1;
    }
};
```

<span id="移除元素"></span>
## [27、移除元素](#back)
```cpp
给定一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，返回移除后数组的新长度。
不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。
元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

给定 nums = [0,1,2,2,3,0,4,2], val = 2,
函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。
注意这五个元素可为任意顺序。
你不需要考虑数组中超出新长度后面的元素。

class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        // 双指针
        int i = 0;
        for (int j = 0; j < nums.size(); j++) {
            if (nums[j] != val) {
                nums[i++] = nums[j];
            }
        }
        return i;
    }
};
```

<span id="实现str()"></span>
## [28、实现str()](#back)
```cpp
实现 strStr() 函数。
给定一个 haystack 字符串和一个 needle 字符串，
在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。
如果不存在，则返回  -1。

输入: haystack = "hello", needle = "ll"
输出: 2

class Solution {
public:
    vector<int> getnext(string p) {
        int len_p = p.size();
        vector<int> next;
        next.push_back(-1);  // 初始值
        int j = 0, k = -1;  // j:后缀； k:前缀
        while (j < len_p - 1) {
            if ((k == -1) || (p[j] == p[k])){
                j++;
                k++;
                next.push_back(k);
            }
            else {
                k = next[k];
            }
        }
        return next;
    }

    int strStr(string haystack, string needle) {
        // KMP字符串匹配
        // 源串不回溯，模式串回溯
        int  i = 0, j = 0;  // i:源串； j:模式串
        int len_ha = haystack.size();
        int len_ne = needle.size();
        vector<int> next;  // 存储模式串匹配时的跳转情况
        next = getnext(needle);
        while ((i < len_ha) && (j < len_ne)) {
            if ((j == -1) || (haystack[i] == needle[j])){
                i++;
                j++;
            }
            else {
                j = next[j];  // 跳转模式串的下一次匹配位置
            }
        }
        if(j == len_ne) return i - j;
        return -1;
    }
};
```

<span id="两数相除"></span>
## [29、两数相除](#back)
```cpp
给定两个整数，被除数 dividend 和除数 divisor。将两数相除，要求不使用乘法、除法和 mod 运算符。
返回被除数 dividend 除以除数 divisor 得到的商。

输入: dividend = 10, divisor = 3
输出: 3

输入: dividend = 7, divisor = -3
输出: -2

说明:
被除数和除数均为 32 位有符号整数。
除数不为 0。
假设我们的环境只能存储 32 位有符号整数，其数值范围是 [−2^31,  {2^31} − 1]。
本题中，如果除法结果溢出，则返回 {2^31} − 1。

class Solution {
public:
    int divide(int dividend, int divisor) {
        // 全部转换为负数，可以避免正数移位的边界问题
        int flag = dividend > 0 == divisor > 0;  // dividend 与 divisor 同号返回1，异号返回0；
        if (dividend > 0) dividend = -dividend;
        if (divisor > 0) divisor = -divisor;
        int result = 0;
        while (dividend <= divisor) {
            unsigned int temp_result = -1;
            unsigned int temp_divisor = divisor;
            while (dividend <= (temp_divisor << 1)) {
                if (temp_result <= (INT_MIN >> 1)) break;
                temp_result = temp_result << 1;
                temp_divisor = temp_divisor << 1;
            }
            result += temp_result;
            dividend -= temp_divisor;
        }
        if (flag){
            if (result <= INT_MIN) return INT_MAX;
            return -result;
        }
        return result;
    }
};
```

<span id="串联所有单词的子串"></span>
## [30、串联所有单词的子串](#back)
```cpp
给定一个字符串 s 和一些长度相同的单词 words。找出 s 中恰好可以由 words 中所有单词串联形成的子串的起始位置。
注意子串要与 words 中的单词完全匹配，中间不能有其他字符，但不需要考虑 words 中单词串联的顺序。

输入：
  s = "barfoothefoobarman",
  words = ["foo","bar"]
输出：[0,9]
解释：
从索引 0 和 9 开始的子串分别是 "barfoo" 和 "foobar" 。
输出的顺序不重要, [9,0] 也是有效答案。

输入：
  s = "wordgoodgoodgoodbestword",
  words = ["word","good","best","word"]
输出：[]

class Solution {
public:
    vector<int> findSubstring(string s, vector<string>& words) {
        // 滑动窗口
        // hash表存单词表单词个数，与字符串滑动窗口中单词个数比较 
        vector<int>  res;
        unordered_map<string, int> m1;  // 单词->单词个数
        if (s.empty() || words.empty()) return {};
        int word_size = words[0].size();  // 单词的大小
        int words_num = words.size();  // 单词的个数
        for (auto w : words) m1[w]++;

        for (int i = 0; i < word_size; i++){
            int left = i, right = i, count_ = 0;
            unordered_map<string, int> m2;
            while (right + word_size <= s.size()) {  // 滑动窗口
                string w = s.substr(right, word_size);  // 从S中提取一个单词拷贝到w
                right += word_size;  // 有边界右移一个单词的长度；
                if (m1.count(w) == 0) {  // 单词不在words中
                    count_ = 0;
                    left = right;
                    m2.clear();
                }
                else {  // 单词在words中，添加到m2中
                    m2[w]++;
                    count_++;
                    while (m2.at(w) > m1.at(w)) {  // 一个单词匹配多次，需要缩小窗口
                        string t_w = s.substr(left, word_size);
                        count_--;
                        m2[t_w]--;
                        left += word_size;
                    }
                    if (count_ == words_num) res.push_back(left);
                }
            }
        }
        return res;        
    }
};
```

<span id="下一个排列"></span>
## [31、下一个排列](#back)

<div align=center><img src="https://github.com/FangChao1086/LeetCode_Solutions/blob/master/依赖文件/下一个排列.gif"></div>

```cpp
实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。
如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。
必须原地修改，只允许使用额外常数空间。

以下是一些例子，输入位于左侧列，其相应输出位于右侧列。
1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1

class Solution {
public:
    // 翻转nums数组start之后的数，包含nums[start]
    void reverse(vector<int>& nums, int start){
        int i = start, j = nums.size() - 1;
        while (i < j) {
            swap(nums[i], nums[j]);
            i++;
            j--;
        }
    }

    void nextPermutation(vector<int>& nums) {
        int i = nums.size() - 1;
        while (i-1 >= 0 && nums[i-1] >= nums[i]) {  // 找到nums[i-1]
            i--;
        }
        if (i-1 >= 0) {
            int j = nums.size() - 1;
            while (j >= 0 && nums[j] <= nums[i-1]) {  // 找到与nums[i-1]反转的nums[j]
                j--;
            }
            swap(nums[i-1],  nums[j]);
        }
        reverse(nums, i);
    }
};
```

<span id="最长有效括号"></span>
## [32、最长有效括号](#back)
```cpp
给定一个只包含 '(' 和 ')' 的字符串，找出最长的包含有效括号的子串的长度。

输入: "(()"
输出: 2
解释: 最长有效括号子串为 "()"

输入: ")()())"
输出: 4
解释: 最长有效括号子串为 "()()"

// 思路：
// 利用两个计数器 left 和 right 。
// 首先，我们从左到右遍历字符串，对于遇到的每个 ‘(’，我们增加 left ;
// 对于遇到的每个 ‘)’ ，我们增加 right 计数器。
// 每当 left == right ，计算有效字符串的长度，记录找到的最长子字符串。
// 如果 right > left，我们将 left 和 right 计数器同时变回 0 。
// 接下来，我们从右到左做一遍类似的工作

// 时间复杂度： O(n) 。遍历两遍字符串。
// 空间复杂度： O(1) 。仅有两个额外的变量 left 和 right 。

class Solution {
public:
    int longestValidParentheses(string s) {
        // 双边遍历; left,right计数
        int left = 0, right = 0, max_len = 0;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(') left++;
            else right++;
            if (left == right) max_len = max(max_len, right * 2);
            else if (right > left) {
                left = 0;
                right = 0;
            }

        }
        left = 0, right = 0;
        for (int i = s.size() - 1; i >= 0; i--) {
            if (s[i] == ')') right++;
            else left++;
            if (left == right) max_len = max(max_len, left * 2);
            else if (left > right) {
                left = 0;
                right = 0;
            }

        }
        return max_len;
    }
};
```

<span id="搜索旋转排序数组"></span>
## [33、搜索旋转排序数组](#back)
```cpp
假设按照升序排序的数组在预先未知的某个点上进行了旋转。
( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。
你可以假设数组中不存在重复的元素。
你的算法时间复杂度必须是 O(log n) 级别。

输入: nums = [4,5,6,7,0,1,2], target = 0
输出: 4

输入: nums = [4,5,6,7,0,1,2], target = 3
输出: -1

class Solution {
public:
    int search(vector<int>& nums, int target) {
        // 二分查找; 查看[left, mid]是否升序，再根据其他条件判断查找范围
        int left = 0, mid = 0, right = nums.size() - 1;
        while (left <= right) {
            mid = (left + right) >> 1; 
            if (nums[mid] == target) return mid;
            if (nums[left] <= nums[mid]) {  // 左边升序,或者mid == left
                if (nums[left] <= target && target <= nums[mid]) right = mid - 1;  // 在左边范围内
                else left = mid + 1;
            }
            else {  // 右边升序
                if (nums[mid] <= target && target <= nums[right]) left = mid + 1;  // 在右边范围内
                else right = mid - 1;
            }
        }
        return -1;
    }
};
```

<span id="排序数组查找元素的第一和最后一个位置"></span>
## [34、排序数组查找元素的第一和最后一个位置](#back)
```cpp
给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
你的算法时间复杂度必须是 O(log n) 级别。
如果数组中不存在目标值，返回 [-1, -1]。

输入: nums = [5,7,7,8,8,10], target = 8
输出: [3,4]

输入: nums = [5,7,7,8,8,10], target = 6
输出: [-1,-1]

class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        // 二分查找
        int left = 0, mid = 0,  right = nums.size() - 1;
        vector<int> res = {-1,-1};
        if (nums.size() == 0 || nums[left] > target || nums[right] < target) return res;  // 边界
        while (left < right) {  // 找到左边第一个元素
            mid = left + right >> 1;
            if (nums[mid] >= target) right = mid;
            else left = mid + 1;
        }
        if (nums[right] == target) res[0] = right;
        right = nums.size();  // 原因：考虑数组长度为1的情况，不能设置为right = nums.size()-1
        while (left < right) {  // 查找右边的第一个元素
            mid = left + right >> 1;
            if (nums[mid] > target) right = mid;
            else left = mid + 1;
        }
        if (nums[right - 1] == target) res[1] = right - 1;
        return res; 
    }
};
```

<span id="搜索插入位置"></span>
## [35、搜索插入位置](#back)
```cpp
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。
如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
你可以假设数组中无重复元素。

输入: [1,3,5,6], 5
输出: 2

输入: [1,3,5,6], 2
输出: 1

输入: [1,3,5,6], 7
输出: 4

输入: [1,3,5,6], 0
输出: 0

class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        // 二分查找
        int left = 0, mid = 0, right = nums.size() - 1;
        if (target < nums[left]) return 0;
        if (target > nums[right]) return nums.size();
        while (left < right) {
            mid = left + right >> 1;
            if (nums[mid] > target) right = mid;
            else if(nums[mid] == target) return mid;
            else left = mid + 1;
        }
        return left;
    }
};
```

<span id="有效的数独"></span>
## [36、有效的数独](#back)
```cpp
判断一个 9x9 的数独是否有效。只需要根据以下规则，验证已经填入的数字是否有效即可。

数字 1-9 在每一行只能出现一次。
数字 1-9 在每一列只能出现一次。
数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
数独部分空格内已填入了数字，空白格用 '.' 表示。

输入:
[
  ["5","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
输出: true

输入:
[
  ["8","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
输出: false
解释: 除了第一行的第一个数字从 5 改为 8 以外，空格内其他数字均与 示例1 相同。
     但由于位于左上角的 3x3 宫内有两个 8 存在, 因此这个数独是无效的。
     
说明:
一个有效的数独（部分已被填充）不一定是可解的。
只需要根据以上规则，验证已经填入的数字是否有效即可。
给定数独序列只包含数字 1-9 和字符 '.' 。
给定数独永远是 9x9 形式的。

class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        // 9个宫格，宫格内统计字符的数量，每个宫格3*3大小
        // -宫格0-|-宫格1-|-宫格2-
        // -宫格3-|-宫格4-|-宫格5-
        // -宫格6-|-宫格7-|-宫格8-
        map<int, map<char, int>> mp_rows;
        map<int, map<char, int>> mp_cols;
        map<int, map<char, int>> mp_boxes;
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    int n = (i / 3) * 3+ (j / 3);
                    mp_rows[i][board[i][j]]++;  // 行方向计数
                    mp_cols[j][board[i][j]]++;  // 列方向计数
                    mp_boxes[n][board[i][j]]++;  // box宫格内计数
                    if (mp_rows[i][board[i][j]] > 1 || mp_cols[j][board[i][j]] > 1 || mp_boxes[n][board[i][j]] > 1){
                        return false;
                    }
                }
            }
        }
        return true;
    }
};
```

<span id="解数独"></span>
## [37、解数独](#back)
```cpp
编写一个程序，通过已填充的空格来解决数独问题。
一个数独的解法需遵循如下规则：
数字 1-9 在每一行只能出现一次。
数字 1-9 在每一列只能出现一次。
数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
空白格用 '.' 表示。

Note:
给定的数独序列只包含数字 1-9 和字符 '.' 。
你可以假设给定的数独只有唯一解。
给定数独永远是 9x9 形式的。

class Solution {
public:
    vector<set<int>> rows, cols, boxes;

    void update(vector<vector<char>> & board) {
        set<int> s = {1,2,3,4,5,6,7,8,9};
        for (int i = 0 ; i < 9; i++) {
            rows.push_back(s);
            cols.push_back(s);
            boxes.push_back(s);
        }
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    int tmp = board[i][j] - '0';
                    rows[i].erase(tmp);
                    cols[j].erase(tmp);
                    boxes[i / 3 + j / 3 * 3].erase(tmp);
                }
            }
        }
        return ;
    }

    bool check(vector<vector<char>> &board, const int &i, const int &j, int num) {
        if (rows[i].find(num) != rows[i].end() && cols[j].find(num) != cols[j].end() && boxes[i / 3 + j / 3 * 3].find(num) != boxes[i / 3 + j / 3 * 3].end()) return true;
        return false;
    }

    int flag = 0;

    // dfs + 回溯
    void dfs(vector<vector<char>> & board, int count) {
        if(count == 81) {
            flag = 1;
            return ;
        }
        int i = count / 9, j = count % 9;
        if ( board[i][j] == '.' ){
            for (int k = 1; k < 10; k++) {  // 检查 1 ～ 9 中数字哪一个可以放入该位置
                if (check(board, i, j, k)) {
                    rows[i].erase(k);
                    cols[j].erase(k);
                    boxes[i / 3 + j / 3 * 3].erase(k);
                    board[i][j] = k + '0';
                    dfs(board, count + 1);
                    if (!flag) {
                    rows[i].insert(k);
                    cols[j].insert(k);
                    boxes[i / 3 + j / 3 * 3].insert(k);
                    board[i][j] = '.';
                    }
                    else return ;
                }
            }
        }
        else dfs(board, count + 1);
        return ;
    }

    void solveSudoku(vector<vector<char>>& board) {
        update(board);
        dfs(board, 0); 
    }
};
```

<span id="报数"></span>
## [38、报数](#back)
```cpp
报数序列是一个整数序列，按照其中的整数的顺序进行报数，得到下一个数。其前五项如下：
1.     1
2.     11
3.     21
4.     1211
5.     111221
1 被读作  "one 1"  ("一个一") , 即 11。
11 被读作 "two 1s" ("两个一"）, 即 21。
21 被读作 "one 2",  "one 1" （"一个二" ,  "一个一") , 即 1211。
给定一个正整数 n（1 ≤ n ≤ 30），输出报数序列的第 n 项。
注意：整数顺序将表示为一个字符串。

输入: 1
输出: "1"

输入: 4
输出: "1211"

class Solution {
public:
    string countAndSay(int n) {
        // 递归，通过n-1个获取第n个
        if (n == 1) return "1";
        string str_old = countAndSay(n - 1);
        string str_new = "";
        char begin = str_old[0];
        int count_ = 1;
        for (int i = 1; i < str_old.size(); i++) {
            if (str_old[i] == begin) {
                count_++;
            }
            else {
                str_new += to_string(count_) + begin;
                begin = str_old[i];
                count_ = 1;
            }
        }
        if (begin == str_old[str_old.size() - 1]) str_new += to_string(count_) + begin;
        return str_new;
    }
};
```

<span id="组合总和"></span>
## [39、组合总和](#back)
```cpp
给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
candidates 中的数字可以 无限制重复被选取。

说明：
所有数字（包括 target）都是正整数。
解集不能包含重复的组合。 

输入: candidates = [2,3,6,7], target = 7,
所求解集为:
[
  [7],
  [2,2,3]
]

输入: candidates = [2,3,5], target = 8,
所求解集为:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]

class Solution {
public:
    vector<vector<int>> res;  // 所有满足条件的数组集合
    vector<int> t;  // 单个满足条件的数组

    void helper(vector<int>& candidates, int target) {
        for(int i = 0; i < candidates.size(); i++){
            if (candidates[i] > target || (t.size() && candidates[i] < t[t.size() - 1])) continue;  // 防止出现重复的数
            t.push_back(candidates[i]);
            if (candidates[i] == target) res.push_back(t); 
            else helper(candidates, target - candidates[i]);
            t.pop_back();
        }
    }

    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        res.clear();
        t.clear();
        helper(candidates, target);
        return res;
    }
};
```

<span id="组合总和2"></span>
## [40、组合总和 II](#back)
```cpp
给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
candidates 中的每个数字在每个组合中 只能使用一次。

说明：
所有数字（包括目标数）都是正整数。
解集不能包含重复的组合。 

输入: candidates = [10,1,2,7,6,1,5], target = 8,
所求解集为:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]

输入: candidates = [2,5,2,1,2], target = 5,
所求解集为:
[
  [1,2,2],
  [5]
]

class Solution {
public:
    vector<int> candidates;
    vector<vector<int>> res;
    vector<int> path;

    void DFS(int start, int target) {
        if (target == 0) res.push_back(path);
        for (int i = start; i < candidates.size() && target >= candidates[i]; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) continue;
            path.push_back(candidates[i]);
            DFS(i + 1, target - candidates[i]);
            path.pop_back();
        }
    }

    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        this->candidates = candidates;
        DFS(0, target);
        return res;
    }
};
```

<span id="缺失的第一个正数"></span>
## [41、缺失的第一个正数](#back)
```cpp
给定一个未排序的整数数组，找出其中没有出现的最小的正整数。

说明:
你的算法的时间复杂度应为O(n)，并且只能使用常数级别的空间。

输入: [1,2,0]
输出: 3

输入: [3,4,-1,1]
输出: 2

输入: [7,8,9,11,12]
输出: 1

class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        // 首次出现最小的正整数一定是小于等于size_ + 1
        int size_ = nums.size();
        vector<int> tmp(size_ + 1, 0);
        for(int i = 0; i < size_; i++) {
            if (nums[i] > 0 && nums[i] <= size_) {
                tmp[nums[i]] = 1;
            }
        }
        int i = 1;
        while (i <= size_ && tmp[i] != 0) {
            i++;
        }
        return i;
    }
};
```

<span id="接雨水"></span>
## [42、接雨水](#back)
<div align=center><img src="https://github.com/FangChao1086/LeetCode_Solutions/blob/master/依赖文件/接雨水.jpg"/></div>  

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        // 双指针
        int left = 0, right = height.size() - 1;
        int ans = 0, left_max = 0, right_max = 0;
        while (left < right) {
            if (height[left] < height[right]) {
                if (height[left] >= left_max) left_max = height[left];
                else ans += left_max - height[left];
                left++;
            }
            else {
                if (height[right] >= right_max) right_max = height[right];
                else ans += right_max - height[right];
                right--;
            }
        } 
        return ans;
    }
};
```

<span id="字符串相乘"></span>
## [43、字符串相乘](#back)
<div align=center><img src="https://github.com/FangChao1086/LeetCode_Solutions/blob/master/依赖文件/字符串相乘.jpg"/></div>  

```cpp
给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。

输入: num1 = "2", num2 = "3"
输出: "6"
示例 2:

输入: num1 = "123", num2 = "456"
输出: "56088"

说明：
num1 和 num2 的长度小于110。
num1 和 num2 只包含数字 0-9。
num1 和 num2 均不以零开头，除非是数字 0 本身。
不能使用任何标准库的大数类型（比如 BigInteger）或直接将输入转换为整数来处理

class Solution {
public:
    string multiply(string num1, string num2) {
        int len1 = num1.size();
        int len2 = num2.size();
        string res(len1 + len2, '0');
        for (int i = len2 - 1; i >= 0; i--) {
            for (int j = len1 - 1; j >= 0; j--) {
                int temp = (res[i + j + 1] - '0') + (num1[j] - '0') * (num2[i] - '0');
                res[i + j + 1] = temp % 10 + '0';  // 当前位
                res[i + j] += temp / 10;  // 前一位加进位 
            }
        }
        // 去除首位的‘0’
        for (int i = 0; i < len1 + len2; i++) {
            if (res[i] != '0') return res.substr(i);
        }
        return "0";
    }
};
```

<span id="通配符匹配"></span>
## [44、通配符匹配](#back)
```cpp
给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配。

'?' 可以匹配任何单个字符。
'*' 可以匹配任意字符串（包括空字符串）。
两个字符串完全匹配才算匹配成功。

说明:
s 可能为空，且只包含从 a-z 的小写字母。
p 可能为空，且只包含从 a-z 的小写字母，以及字符 ? 和 *。

输入:
s = "aa"
p = "a"
输出: false
解释: "a" 无法匹配 "aa" 整个字符串。

输入:
s = "aa"
p = "*"
输出: true
解释: '*' 可以匹配任意字符串。

输入:
s = "cb"
p = "?a"
输出: false
解释: '?' 可以匹配 'c', 但第二个 'a' 无法匹配 'b'。

输入:
s = "adceb"
p = "*a*b"
输出: true
解释: 第一个 '*' 可以匹配空字符串, 第二个 '*' 可以匹配字符串 "dce".

输入:
s = "acdcb"
p = "a*c?b"
输入: false

class Solution {
public:
    bool isMatch(string s, string p) {
        // dp[i][j]:s中的前i位是否与P中的第j位相匹配
        int n = s.size(), m = p.size();
        bool dp[n + 1][m + 1] = {false};  // 初始化的bool数组是false
        dp[0][0] = true;
        for (int j = 0; j < m; j++) {  // 初始化
            dp[0][j + 1] = dp[0][j] && p[j] == '*';
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (s[i] == p[j] || p[j] == '?')  // i - 1与j - 1匹配
                    dp[i + 1][j + 1] = dp[i][j];
                if (p[j] == '*') {
                    dp[i + 1][j + 1] = dp[i + 1][j] || dp[i][j + 1];  // p的第j位匹配空：dp[i + 1][j]；  p的第j位匹配不为空:dp[i][j + 1];
                }
            }
        }
        return dp[n][m];
    }
};
```

<span id="跳跃游戏2"></span>
## [45、跳跃游戏 II](#back)
```cpp
给定一个非负整数数组，你最初位于数组的第一个位置。
数组中的每个元素代表你在该位置可以跳跃的最大长度。
你的目标是使用最少的跳跃次数到达数组的最后一个位置。

输入: [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。

说明:
假设你总是可以到达数组的最后一个位置。

class Solution {
public:
    int jump(vector<int>& nums) {
        // 贪心法
        // 参考：https://leetcode-cn.com/problems/jump-game-ii/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-10/
        int end = 0;
        int maxPosition = 0;
        int steps = 0;
        for (int i = 0; i < nums.size() - 1; i++) {  // 在i==0的时候，steps增加了1，所以i < nums.size() - 1
            // 找到能跳的最远的
            maxPosition = max(maxPosition, nums[i] + i);
            if (i == end) {  // 遇到边界，更新边界
                end = maxPosition;
                steps++;
            }
        }
        return steps;
    }
};
```

<span id="全排列"></span>
## [46、全排列](#back)
```cpp
给定一个没有重复数字的序列，返回其所有可能的全排列。

输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]

class Solution {
public:
    // 回溯
    void backTrack(int n, vector<int> &nums, vector<vector<int>> &res, int first) {
        if (first == n) res.push_back(nums);
        for(int i = first; i < n; i++) {
            swap(nums[first], nums[i]);
            backTrack(n, nums, res, first + 1);
            swap(nums[first], nums[i]);
        }
    } 

    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> res;
        backTrack(nums.size(), nums, res, 0);
        return res;
    }
};
```

<span id="全排列2"></span>
## [47、全排列 II](#back)
```cpp
给定一个可包含重复数字的序列，返回所有不重复的全排列。

输入: [1,1,2]
输出:
[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]

class Solution {
public:
    // 回溯（辅助函数）
    void backTrack(int n, vector<int> &nums, vector<vector<int>> &res, int first) {
        if (first == n) res.push_back(nums);
        for (int i = first; i < n; i++) {
            bool flag = false;
            for (int j = first; j < i; j++) {
                if (nums[j] == nums[i]) {
                    flag = true;
                    break;
                } 
            } 
            if (flag) continue;
            swap(nums[first], nums[i]);
            backTrack(n, nums, res, first + 1);
            swap(nums[first], nums[i]);            
        }
    }


    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> res;
        backTrack(nums.size(), nums, res, 0);
        return res;
    }
};
```

<span id="旋转图像"></span>
## [48、旋转图像](#back)
<div align=center><img src="https://github.com/FangChao1086/LeetCode_Solutions/blob/master/依赖文件/48_旋转图像.jpg"/></div>  

```cpp
给定一个 n × n 的二维矩阵表示一个图像。
将图像顺时针旋转 90 度。

说明：
你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。

给定 matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],
原地旋转输入矩阵，使其变为:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]

给定 matrix =
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 
原地旋转输入矩阵，使其变为:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]

class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        for (int loop = 0; loop < n / 2 ; loop++)  // 外层循环
            for (int i = loop, j = loop; i < n - 1 - loop; i++) {  // 中层循环
                int temp = matrix[i][j];
                for (int z = 0; z < 4; z++){  // 4个位置旋转
                    int tmpi = i;
                    i = j;
                    j = n - 1 - tmpi;
                    swap(temp,matrix[i][j]);
                }
            }
    }
};
```

<span id="字母异位词分组"></span>
## [49、字母异位词分组](#back)
```cpp
给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]

说明：
所有输入均为小写字母。
不考虑答案输出的顺序。

class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        vector<vector<string>> res;
        map<string, vector<string>> mp;
        for(auto str:strs){
            string tmp = str;
            sort(tmp.begin(), tmp.end());
            mp[tmp].push_back(str);
        }
        for(auto m:mp){
            res.push_back(m.second);
        }
        return res;
    }
};
```

<span id="Pow"></span>
## [50、Pow(x, n)](#back)
```cpp
实现 pow(x, n) ，即计算 x 的 n 次幂函数。

输入: 2.00000, 10
输出: 1024.00000

输入: 2.10000, 3
输出: 9.26100

输入: 2.00000, -2
输出: 0.25000
解释: 2-2 = 1/22 = 1/4 = 0.25

说明:
-100.0 < x < 100.0
n 是 32 位有符号整数，其数值范围是 [−231, 231 − 1] 。

class Solution {
public:
    double myPow(double x, int n) {
        double res = 1;
        long long m = n;
        if (m < 0) {
            x = 1 / x;
            m = -m;
        }
        while (m) {
            if (m & 1) {  // E % 2 == 1
                res *= x;
            }
            x *= x;
            m >>= 1;  // m /= 2
        }
        return res;
    }
};

复杂度分析:
时间复杂度：O(logn) 对每一个 n 的二进制位表示，我们都至多需要累乘 1 次，所以总的时间复杂度为 O(logn) 。
空间复杂的：O(1) 我们只需要用到 1 个变量来保存当前的乘积结果。
```

<span id="N皇后"></span>
## [51、N皇后](#back)
```cpp
n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
给定一个整数 n，返回所有不同的 n 皇后问题的解决方案。
每一种解法包含一个明确的 n 皇后问题的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
（横竖与左右斜线上仅仅只能出现1个皇后）

输入: 4
输出: [
 [".Q..",  // 解法 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // 解法 2
  "Q...",
  "...Q",
  ".Q.."]
]
解释: 4 皇后问题存在两个不同的解法。

实现方式：回溯
hills： \ 向右的斜线 row - col == 常数(同一斜线上)
dales： / 向左的斜线 row + col == 常数(同一斜线上)

class Solution {
public:
    int n;
    int rows[100];
    int hills[100];
    int dales[100];
    int queens[100];
    vector<vector<string>> output;

    bool isNotUnderAttack(int row, int col) {
        int res = rows[col] + hills[row - col + n] + dales[row + col];
        return  (res == 0) ? true : false;
    }

    void placeQueen(int row, int col) {
        queens[row] = col;
        rows[col] = 1;
        hills[row - col + n] = 1;
        dales[row + col] = 1;
    }

    void addSolution() {
        vector<string> solution;
        for (int i = 0; i < n; i++) {
            int col = queens[i];
            string s_;
            for (int j = 0; j < col; j++) s_ += ".";
            s_ += "Q";
            for (int j = col + 1; j < n; j++) s_ += ".";
            solution.push_back(s_);
        }
        output.push_back(solution);
    }

    void removeQueen(int row, int col) {
        queens[row] = 0;
        rows[col] = 0;
        hills[row - col + n] = 0;
        dales[row + col] = 0;
    }

    // 回溯
    void backTrack(int row) {
        for (int col = 0; col < n; col++) {
            if (isNotUnderAttack(row, col)) {
                placeQueen(row, col);
                if (row == n - 1) addSolution();
                else backTrack(row + 1);
                removeQueen(row, col);
            }
        }
    }

    vector<vector<string>> solveNQueens(int n) {
        // hills \
        // dales /       
        this -> n = n;
        backTrack(0);
        return output;
    }
};
```

<span id="N皇后2"></span>
## [52、N皇后 II](#back)
```cpp
n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
给定一个整数 n，返回 n 皇后不同的解决方案的数量。

输入: 4
输出: 2
解释: 4 皇后问题存在如下两个不同的解法。
[
 [".Q..",  // 解法 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // 解法 2
  "Q...",
  "...Q",
  ".Q.."]
]

class Solution {
public:
    int n, res = 0;
    int rows[100];
    int pie[100];
    int na[100];

    bool isNotAttrack(int row, int col) {
        return rows[col] + pie[row + col] + na[row - col + n] == 0;
    }

    void placeQueen(int row, int col) {
        rows[col] = 1;
        pie[row + col] = 1;
        na[row - col + n] = 1;
    }

    void removeQueen(int row, int col) {
        rows[col] = 0;
        pie[row + col] = 0;
        na[row - col + n] = 0;
    }

    // 回溯
    void backTrack(int row) {
        for (int col = 0; col < n; col++) {
            if (isNotAttrack(row, col)) {
                placeQueen(row, col);
                if (row == n - 1) {
                    res++;
                }
                else backTrack(row + 1);
                removeQueen(row, col);
            }
        }
    }
    
    int totalNQueens(int n) {
        this -> n = n;
        backTrack(0);
        return res;
    }
};
```

<span id="最大子序和"></span>
## [53、最大子序和](#back)
```cpp
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。

进阶:
如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的分治法求解。

class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        if (nums.size() == 0) return 0;
        int max_1 = nums[0], max_all = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            max_1 = max(max_1 + nums[i], nums[i]);
            max_all = max(max_1, max_all);
        }
        return max_all;
    }
};

复杂度分析
时间复杂度：O(N) 只遍历一次数组。
空间复杂度：O(1) 只使用了常数空间。
```

<span id="螺旋矩阵"></span>
## [54、螺旋矩阵](#back)
```cpp
给定一个包含 m x n 个元素的矩阵（m 行, n 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。

输入:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
输出: [1,2,3,6,9,8,7,4,5]

输入:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
输出: [1,2,3,4,8,12,11,10,9,5,6,7]

class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> res;
        if (matrix.size() == 0) return res;
        int rows = matrix.size();
        int cols = matrix[0].size();
        int left = 0, right = cols - 1, top = 0, bottom = rows - 1;
        while (left <= right && top <= bottom) {
            if (left <= right)  // ->
                for (int i = left; i <= right; i++) 
                    res.push_back(matrix[top][i]);
            if (top < bottom && left <= right)  // 向下
                for (int i = top + 1; i <= bottom; i++)
                    res.push_back(matrix[i][right]);
            if (top < bottom && left < right)  // <-
                for (int i = right - 1; i >= left; i--)
                    res.push_back(matrix[bottom][i]);
            if (top + 1 < bottom && left < right)  // 向上
                for (int i = bottom - 1; i > top; i--)
                    res.push_back(matrix[i][left]);
            left++, right--, top++, bottom--;
        }
        return res;
    }
};
```

<span id="跳跃游戏"></span>
## [55、跳跃游戏](#back)
```cpp
给定一个非负整数数组，你最初位于数组的第一个位置。
数组中的每个元素代表你在该位置可以跳跃的最大长度。
判断你是否能够到达最后一个位置。

输入: [2,3,1,1,4]
输出: true
解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。

输入: [3,2,1,0,4]
输出: false
解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。

class Solution {
public:
    bool canJump(vector<int>& nums) {
        // 贪心法
        int index = nums.size() - 1;
        for (int i = index; i >= 0 ; i--) 
            if (i + nums[i] >= index)  // 表示从 i 可以跳到 index 位置，则更新 index ; 其中 index 最初指向终点
                index = i;
        return index == 0;
    }
};
```

<span id="合并区间"></span>
## [56、合并区间](#back)
```cpp
给出一个区间的集合，请合并所有重叠的区间。

输入: [[1,3],[2,6],[8,10],[15,18]]
输出: [[1,6],[8,10],[15,18]]
解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

输入: [[1,4],[4,5]]
输出: [[1,5]]
解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。

class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>> res;
        int len_in = intervals.size();
        if(len_in == 0) return res;
        sort(intervals.begin(), intervals.end());
        res.push_back(intervals[0]);
        for (int i = 1, j = 0; i < len_in; i++) {
            if (res[j][1] >= intervals[i][0]) {
                if (res[j][1] < intervals[i][1]) {
                    res[j][1] = intervals[i][1];
                }
            }
            else {
                j++;
                res.push_back(intervals[i]);
            }
        }
        return res; 
    }
};
```

<span id="插入区间"></span>
## [57、插入区间](#back)
```cpp
给出一个无重叠的 ，按照区间起始端点排序的区间列表。
在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

输入: intervals = [[1,3],[6,9]], newInterval = [2,5]
输出: [[1,5],[6,9]]

输入: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
输出: [[1,2],[3,10],[12,16]]
解释: 这是因为新的区间 [4,8] 与 [3,5],[6,7],[8,10] 重叠。

class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        intervals.push_back(newInterval);
        vector<vector<int>> res;
        sort(intervals.begin(), intervals.end());
        res.push_back(intervals[0]);
        for (int i = 1, j = 0; i < intervals.size(); i++) {
            if (res[j][1] >= intervals[i][0]) {
                if (res[j][1] < intervals[i][1]) {
                    res[j][1] = intervals[i][1];
                }
            }
            else {
                j++;
                res.push_back(intervals[i]);
            }
        }
        return res;
    }
};
```

<span id="最后一个单词的长度"></span>
## [58、最后一个单词的长度](#back)
```cpp
给定一个仅包含大小写字母和空格 ' ' 的字符串，返回其最后一个单词的长度。
如果不存在最后一个单词，请返回 0 。
说明：一个单词是指由字母组成，但不包含任何空格的字符串。

输入: "Hello World"
输出: 5

class Solution {
public:
    int lengthOfLastWord(string s) {
        int end = 0, start = 0, flag = 0;
        for (int i = s.size() - 1; i >= 0; i--) {
            if(flag == 0 && s[i] != ' ') {
                end = i + 1;
                flag = 1;
            }
            if(flag == 1 && s[i] == ' ') {
                start = i + 1;
                break;
            }
        }
        return end - start;
    }
};
```

<span id="螺旋矩阵2"></span>
## [59、螺旋矩阵 II](#back)
```cpp
给定一个正整数 n，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。

输入: 3
输出:
[
 [ 1, 2, 3 ],
 [ 8, 9, 4 ],
 [ 7, 6, 5 ]
]

class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> res(n, vector<int> (n, 0));
        int num = 0;
        int left = 0, right = n - 1, top = 0, bottom = n - 1;
        while (left <= right && top <= bottom) {
            for (int i = left; i <= right; i++) {
                res[top][i] = ++num;
            }
            if (left <= right && top < bottom) {
                for (int i = top + 1; i <= bottom; i++) {
                    res[i][right] = ++num;
                }
            }
            if (left < right && top < bottom ) {
                for (int i = right - 1; i >= left; i--) {
                    res[bottom][i] = ++num;
                }
            }
            if (left < right && top + 1 < bottom) {
                for (int i = bottom - 1; i > top; i--) {
                    res[i][left] = ++num;
                }
            }
            left++, right--, top++, bottom--;
        }
        return res;
    }
};
```

<span id="第k个排列"></span>
## [60、第k个排列](#back)
```cpp
给出集合 [1,2,3,…,n]，其所有元素共有 n! 种排列。
按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下：
"123"
"132"
"213"
"231"
"312"
"321"
给定 n 和 k，返回第 k 个排列。

说明：
给定 n 的范围是 [1, 9]。
给定 k 的范围是[1,  n!]。

输入: n = 3, k = 3
输出: "213"

输入: n = 4, k = 9
输出: "2314"

int factorial(int n){  // 计算阶乘
    int res= 1;
    for (int i = 2; i <= n; i++) {
        res *= i;
    }
    return res;
}

class Solution {
private:
    list<char> ls;
    int this_k;

    int get_k(int n) {  // 假设还剩n个数没有添加，返回需要以该字符开头的那个字符
        int tot = factorial(n - 1);
        int res = 1;
        while (this_k > tot) {
            res++;
            this_k -= tot;
        }
        return res;
    }

    char get_list_k(int k) {  // 获取链表中的第 k 个元素
        auto it = ls.begin();
        for (int i = 0; i < k - 1; i++) 
            it++;
        char res = *it;
        ls.erase(it);
        return res;
    }

public:
    string getPermutation(int n, int k) {
        // 由于集合中的元素没有重复，每一个元素开头的排列的数量均相同，且均为(n-1)!。
        // 由此，我们可以得知，1开头的排列总数为2!=2，同理2,3开头的排列总数也都为2。因此，第3个排列的第一位应当是2。
        this_k = k;
        string res = "";
        for (int i = 1; i <= n; i++) 
            ls.push_back(48 + i);
        for (int i = n; i >= 1; i--)
            res += get_list_k(get_k(i));  // 在字符串末尾添加新的字符
        return res;
    }
};
```

<span id="旋转链表"></span>
## [61、旋转链表](#back)
```cpp
给定一个链表，旋转链表，将链表每个节点向右移动 k 个位置，其中 k 是非负数。

输入: 1->2->3->4->5->NULL, k = 2
输出: 4->5->1->2->3->NULL
解释:
向右旋转 1 步: 5->1->2->3->4->NULL
向右旋转 2 步: 4->5->1->2->3->NULL

输入: 0->1->2->NULL, k = 4
输出: 2->0->1->NULL
解释:
向右旋转 1 步: 2->0->1->NULL
向右旋转 2 步: 1->2->0->NULL
向右旋转 3 步: 0->1->2->NULL
向右旋转 4 步: 2->0->1->NULL

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
    ListNode* rotateRight(ListNode* head, int k) {
        // 头尾闭环
        // 拆分
        if (head == NULL) return head;
        int count_ = 1;
        ListNode* pHead = head;
        while (pHead -> next) {
            count_++;
            pHead = pHead -> next;
        }
        pHead -> next = head;  // 形成闭环
        int cut_num = count_ - k % count_;
        while (cut_num - 1) {  // 找到尾部的数
            head = head -> next;
            cut_num--;
        }
        pHead = head;  // 尾部非空节点
        head = head -> next;  // 新的头
        pHead -> next = NULL;
        return head;
    }
};
```

<span id="不同路径"></span>
## [62、不同路径](#back)
```cpp
一个机器人位于一个 m x n 网格的左上角。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角。
问总共有多少条不同的路径？
说明：m 和 n 的值均不超过 100。

输入: m = 3, n = 2
输出: 3
解释:
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向右 -> 向下
2. 向右 -> 向下 -> 向右
3. 向下 -> 向右 -> 向右

输入: m = 7, n = 3
输出: 28

class Solution {
public:
    int uniquePaths(int m, int n) {
        // // 动态规划
        // vector<vector<int>> dp(m, vector<int> (n, 1));
        // for (int i = 1; i < m; i++) {
        //     for (int j = 1; j < n; j++) {
        //         dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        //     }
        // }
        // return dp[m -1 ][n - 1];

        // // 优化1
        // vector<int> pre(n, 1);
        // vector<int> cur(n, 1);
        // for (int i = 1; i < m; i++) {
        //     for (int j = 1; j < n; j++) {
        //         cur[j] = pre[j] + cur[j - 1];
        //     }
        //     pre = cur;
        // }
        // return cur[n - 1];

        // 优化2
        vector<int> cur(n, 1);
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                cur[j] += cur[j-1];
            }
        }
        return cur[n - 1];
    }
};
```

<span id="不同路径2"></span>
## [63、不同路径 II](#back)
```cpp
一个机器人位于一个 m x n 网格的左上角。
机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角。
现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
网格中的障碍物和空位置分别用 1 和 0 来表示。

说明：m 和 n 的值均不超过 100。

输入:
[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
输出: 2
解释:
3x3 网格的正中间有一个障碍物。
从左上角到右下角一共有 2 条不同的路径：
1. 向右 -> 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右 -> 向右

class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        if(obstacleGrid[0][0] == 1) return 0;
        int rows = obstacleGrid.size(), cols = obstacleGrid[0].size();
        long dp[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (0 == i && 0 == j) dp[0][0] = 1;
                else if (0 == i && 0 != j) 
                    dp[i][j] = (obstacleGrid[i][j] == 1) ? 0 : dp[i][j - 1];
                else if (0 != i && 0 == j) 
                    dp[i][j] = (obstacleGrid[i][j] == 1) ? 0 : dp[i - 1][j];
                else dp[i][j] = (obstacleGrid[i][j] == 1) ? 0 : (dp[i - 1][j] + dp[i][j - 1]);
            }
        }
        return dp[rows - 1][cols - 1];
    }
};
```

<span id="最小路径和"></span>
## [64、最小路径和](#back)
```cpp
给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
说明：每次只能向下或者向右移动一步。

输入:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 7
解释: 因为路径 1→3→1→1→1 的总和最小。

class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int rows = grid.size();
        int cols = grid[0].size();
        for (int i = 1; i < rows; i++) {
            grid[i][0] += grid[i - 1][0];  
        }
        for (int i = 1; i < cols ;i++) {
            grid[0][i] += grid[0][i - 1];
        }
        for (int i = 1; i < rows; i++) {
            for (int j = 1; j < cols; j++) {
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1]);
            }
        }
        return grid[rows - 1][cols - 1];
    }
};
```

<span id="有效数字"></span>
## [65、有效数字](#back)
```cpp
验证给定的字符串是否可以解释为十进制数字。

"0" => true
" 0.1 " => true
"abc" => false
"1 a" => false
"2e10" => true
" -90e3   " => true
" 1e" => false
"e3" => false
" 6e-1" => true
" 99e2.5 " => false
"53.5e93" => true
" --6 " => false
"-+3" => false
"95a54e53" => false

说明: 我们有意将问题陈述地比较模糊。在实现代码之前，你应当事先思考所有可能的情况。
这里给出一份可能存在于有效十进制数字中的字符列表：
数字 0-9
指数 - "e"
正/负号 - "+"/"-"
小数点 - "."
当然，在输入中，这些字符的上下文也很重要。

class Solution {
public:
    bool isNumber(string s) {
        int len_s = s.size();
        if (len_s == 0) return false;
        int i = 0, j = s.size() - 1;
        while (s[i] == ' ') i++;
        while (s[j] == ' ') j--;
        if (j == -1) return false;
        s = s.substr(i, j + 1 - i);
        if (s[0] == '+' || s[0] == '-') s = s.substr(1, s.size() - 1);
        if (s.size() == 1 && s[0] == '.') return false;
        int count_n = 0, count_dot = 0, count_e = 0;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] >= '0' && s[i] <= '9') {
                count_n = 1;
                continue;
            }
            else if (s[i] == '.') {
                if (count_dot != 0 || count_e != 0) return false;
                count_dot = 1;
            }
            else if (s[i] == 'e' || s[i] == 'E') {
                if (count_e != 0 || count_n == 0) return false;
                count_e = 1;
                int c = i + 1;
                if (s[c] == '+' || s[c] == '-') c = c + 1;
                if (s[c] == '\0') return false;
                i = c - 1;
            }
            else return false;
        }
        return true;
    }
};
```

<span id="加一"></span>
## [66、加一](#back)
```cpp
给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
你可以假设除了整数 0 之外，这个整数不会以零开头。

输入: [1,2,3]
输出: [1,2,4]
解释: 输入数组表示数字 123。

输入: [4,3,2,1]
输出: [4,3,2,2]
解释: 输入数组表示数字 4321。

class Solution {
public:
    vector<int> plusOne(vector<int>& digits) {
        int len_digits = digits.size();
        if (len_digits == 1) {
            if (digits[0] == 9) {
                digits[0] = 0;
                digits.insert(digits.begin(), 1);
            }
            else digits[0]++;
        }
        else digits[len_digits - 1] += 1;
        for (int i = len_digits - 1; i > 0; i--) {
            if (digits[i] % 10 == 0) {
                digits[i] = 0;
                digits[i - 1] = (digits[i - 1] + 1) % 10;
            }
            else {
                break;
            }
        }
        if(digits[0] == 0) digits.insert(digits.begin(), 1);
        return digits;
    }
};
```

<span id="二进制求和"></span>
## [67、二进制求和](#back)
```cpp
给定两个二进制字符串，返回他们的和（用二进制表示）。
输入为非空字符串且只包含数字 1 和 0。

输入: a = "11", b = "1"
输出: "100"

输入: a = "1010", b = "1011"
输出: "10101"

class Solution {
public:
    string addBinary(string a, string b) {
        // 补全长度
        // 相加
        int len_a = a.size(), len_b = b.size();
        while (len_a < len_b) {
            a = '0' + a;
            len_a++;
        }
        while (len_a > len_b) {
            b = '0' + b;
            len_b++;
        }
        for (int i = len_a - 1; i > 0; i--) {
            a[i] = a[i] - '0' + b[i];
            if ( a[i] >= '2' ) {
                a[i] = a[i] % 2 + '0';
                a[i - 1] += 1;
            }
        }
        a[0] = a[0] - '0' + b[0];
        if (a[0] >= '2') {
            a[0] = (a[0] - '0') % 2 + '0';
            a = '1' + a;
        }
        return a;
    }
};
```

<span id="文本左右对齐"></span>
## [68、文本左右对齐](#back)
```cpp

给定一个单词数组和一个长度 maxWidth，重新排版单词，使其成为每行恰好有 maxWidth 个字符，且左右两端对齐的文本。
你应该使用“贪心算法”来放置给定的单词；也就是说，尽可能多地往每行中放置单词。必要时可用空格 ' ' 填充，使得每行恰好有 maxWidth 个字符。
要求尽可能均匀分配单词间的空格数量。如果某一行单词间的空格不能均匀分配，则左侧放置的空格数要多于右侧的空格数。
文本的最后一行应为左对齐，且单词之间不插入额外的空格。

说明:
单词是指由非空格字符组成的字符序列。
每个单词的长度大于 0，小于等于 maxWidth。
输入单词数组 words 至少包含一个单词。

输入:
words = ["This", "is", "an", "example", "of", "text", "justification."]
maxWidth = 16
输出:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]

输入:
words = ["What","must","be","acknowledgment","shall","be"]
maxWidth = 16
输出:
[
  "What   must   be",
  "acknowledgment  ",
  "shall be        "
]
解释: 注意最后一行的格式应为 "shall be    " 而不是 "shall     be",
     因为最后一行应为左对齐，而不是左右两端对齐。       
     第二行同样为左对齐，这是因为这行只包含一个单词。

输入:
words = ["Science","is","what","we","understand","well","enough","to","explain",
         "to","a","computer.","Art","is","everything","else","we","do"]
maxWidth = 20
输出:
[
  "Science  is  what we",
  "understand      well",
  "enough to explain to",
  "a  computer.  Art is",
  "everything  else  we",
  "do                  "
]

class Solution {
public:
    vector<string> fullJustify(vector<string>& words, int maxWidth) {
        vector<string> res;
        int n = words.size();
        for ( int i = 0; i < n; ) {
            int len = 0, num = -1;  // len：当前行的单词总长度；num：当前行包含的空格的数量，单词数 - 1
            while ( len + num <= maxWidth && i < n ) {  // 当前行尚未达到最大，还可以增加单词
                len += words[i].size();
                num++;
                i++;
            }
            if ( len + num > maxWidth ) {  // 当前行已经达到最大，不能增加新的单词
                i--;
                num--;
                len -= words[i].size();
            }
            if ( i != n ) {  // 尚未遍历到最后一个单词
                i -= num + 1;
                int blank = maxWidth - len;
                if ( num > 0) {
                     vector<int> blanks(num, blank / num);
                    for ( int j = 0; j < blank % num; j++ ) blanks[j] += 1;
                    string s;
                    int j;
                    for ( j = 0; j < blanks.size(); j++) {  // 当前行去除最后一个单词后添加
                        s.append(words[i + j]);
                        s.append(blanks[j], ' ');
                    }
                    for ( ; j < num + 1; j++) s.append(words[i+j]);  // 当前行的最后一个单词添加
                    res.push_back(s);
                }
                else {  // 当前行只能有一个单词
                    string s = words[i];
                    s.append(blank, ' ');
                    res.push_back(s);
                }
            }
            else {  // 遍历到最后一个单词
                i -= num + 1;
                string s;
                for ( int j = 0; j < num; j++ ) {  // 当前行去除最后一个单词后的添加
                    s.append(words[i + j]);
                    s.append(" ");
                }
                s.append(words[i + num]);
                s.append(maxWidth - len - num, ' ');
                res.push_back(s);
            }
            i += num + 1;
        }
        return res;
    }
};
```

<span id="X的平方根"></span>
## [69、X的平方根](#back)
```cpp
实现 int sqrt(int x) 函数。
计算并返回 x 的平方根，其中 x 是非负整数。
由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

输入: 4
输出: 2

输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 
     由于返回类型是整数，小数部分将被舍去。

// 二分查找
class Solution {
public:
    int mySqrt(int x) {
        int left = 0, mid = 0, right = x;
        if (x <= 1) return x;
        while (left < right) {
            mid = left + right >> 1;
            if (mid <= x / mid) left = mid + 1;
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

<span id="爬楼梯"></span>
## [70、爬楼梯](#back)
```cpp
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
注意：给定 n 是一个正整数。

输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。
1.  1 阶 + 1 阶
2.  2 阶

输入： 3
输出： 3
解释： 有三种方法可以爬到楼顶。
1.  1 阶 + 1 阶 + 1 阶
2.  1 阶 + 2 阶
3.  2 阶 + 1 阶

class Solution {
public:
    int climbStairs(int n) {
        int f1 = 1;
        long f2 = 1;  // 防止出现溢出
        while ( n-- ) {
            f2 = f1 + f2;
            f1 = f2 - f1;
        }
        return f1;
    }
};
```

<span id="简化路径"></span>
## [71、简化路径](#back)
```cpp
以 Unix 风格给出一个文件的绝对路径，你需要简化它。或者换句话说，将其转换为规范路径。
在 Unix 风格的文件系统中，一个点（.）表示当前目录本身；
此外，两个点 （..） 表示将目录切换到上一级（指向父目录）；
两者都可以是复杂相对路径的组成部分。
请注意，返回的规范路径必须始终以斜杠 / 开头，并且两个目录名之间必须只有一个斜杠 /。
最后一个目录名（如果存在）不能以 / 结尾。此外，规范路径必须是表示绝对路径的最短字符串。

输入："/home/"
输出："/home"
解释：注意，最后一个目录名后面没有斜杠。

输入："/../"
输出："/"
解释：从根目录向上一级是不可行的，因为根是你可以到达的最高级。

输入："/home//foo/"
输出："/home/foo"
解释：在规范路径中，多个连续斜杠需要用一个斜杠替换。

输入："/a/./b/../../c/"
输出："/c"

输入："/a/../../b/../c//.//"
输出："/c"

输入："/a//b////c/d//././/.."
输出："/a/b/c"

class Solution {
public:
    string simplifyPath(string path) {
        stack<string> st;
        string dir;
        path += "/";
        for (auto c : path) {
            if (c == '/') {
                if (dir == ".." && !st.empty()) st.pop();
                else if (dir != ".." && dir != "." && !dir.empty())
                    st.push(dir);
                dir.clear();
            }
            else dir += c;
        }
        string res;
        while (!st.empty()) {
            string t = st.top();
            st.pop();
            res += string(t.rbegin(), t.rend()) + "/";
        }
        reverse(res.begin(), res.end());
        if (res.empty()) return "/";
        return res;
    }
};
```

<span id="编辑距离"></span>
## [72、编辑距离](#back)
```cpp
给定两个单词 word1 和 word2，计算出将 word1 转换成 word2 所使用的最少操作数 。
你可以对一个单词进行如下三种操作：
插入一个字符
删除一个字符
替换一个字符

输入: word1 = "horse", word2 = "ros"
输出: 3
解释: 
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')

输入: word1 = "intention", word2 = "execution"
输出: 5
解释: 
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')

class Solution {
public:
    int minDistance(string word1, string word2) {
        int n = word1.size();
        int m = word2.size();
        vector< vector<int> > dp(n + 1, vector<int>(m + 1, 0));
        for (int i = 1; i <= n; i++) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= m; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + 1;
                if (word1[i - 1] == word2[j - 1]) {
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1]);
                }
                else {
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + 1);
                }
            }
        }
        return dp[n][m];
    }
};
```

<span id="矩阵置零"></span>
## [73、矩阵置零](#back)
```cpp
给定一个 m x n 的矩阵，如果一个元素为 0，则将其所在行和列的所有元素都设为 0。请使用原地算法。

输入: 
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]
输出: 
[
  [1,0,1],
  [0,0,0],
  [1,0,1]
]

输入: 
[
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]
输出: 
[
  [0,0,0,0],
  [0,4,5,0],
  [0,3,1,0]
]

class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        bool row = false;  // 第一行是否需要置零
        bool col = false;  // 第一列是否需要置零
        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix[i].size(); j++) {
                if (matrix[i][j] == 0) {
                    if (i == 0) row = true;  // 第 1 行需要置零
                    if (j == 0) col = true;  // 第 1 列需要置零
                    matrix[i][0] = 0;  // 第 i 行的第一个元素置零，表示第 i 行需要全部置零
                    matrix[0][j] = 0;  // 第 j 列的第一个元素置零，表示第 j 列需要全部置零
                }
            }
        }
        for (int i = 1; i < matrix.size(); i++)  // 第 i 行的第一个元素置零，表示第 i 行需要全部置零
            if (matrix[i][0] == 0) 
                for (int j = 1; j < matrix[i].size(); j++) 
                    matrix[i][j] = 0;
        for (int j = 1; j < matrix[0].size(); j++)  // 第 j 列的第一个元素置零，表示第 j 列需要全部置零
            if (matrix[0][j] == 0)
                for (int i = 1; i < matrix.size(); i++) 
                    matrix[i][j] = 0;
        if (row == true)  // 第 1 行置零
            for (int i = 0; i < matrix[0].size(); i++) 
                matrix[0][i] = 0;
        if (col == true)  // 第 1 列置零
            for (int j = 0 ; j < matrix.size(); j++) 
                matrix[j][0] = 0;
    }
};
```

<span id="搜索二维矩阵"></span>
## [74、搜索二维矩阵](#back)
```cpp
编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
每行中的整数从左到右按升序排列。
每行的第一个整数大于前一行的最后一个整数。

输入:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 3
输出: true

输入:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 13
输出: false

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int rows = matrix.size();
        if (rows == 0) return false;
        int cols = matrix[0].size();
        // 二分查找
        int left = 0, right = rows * cols - 1;
        while (left <= right) {
            int mid = left + right >> 1;
            if (matrix[mid / cols][mid % cols] == target) return true;
            else if (matrix[mid / cols][mid % cols] > target)
                right = mid - 1;
            else left = mid + 1;
        }
        return false;
    }
};
```

<span id="颜色分类"></span>
## [75、颜色分类](#back)
```cpp
给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
注意:不能使用代码库中的排序函数来解决这道题。

输入: [2,0,2,1,1,0]
输出: [0,0,1,1,2,2]

进阶：
一个直观的解决方案是使用计数排序的两趟扫描算法。
首先，迭代计算出0、1 和 2 元素的个数，然后按照0、1、2的排序，重写当前数组。
你能想出一个仅使用常数空间的一趟扫描算法吗？

class Solution {
public:
    void sortColors(vector<int>& nums) {
        int start = 0;
        int end = nums.size() - 1;
        if (nums.size() == 0) return ;
        for (int i = 0; start <= i && i <= end; i++) {
            if (nums[i] == 0  && start != i) {
                swap(nums[start], nums[i]);
                start++;
                i--;
            }
            if (nums[i] == 2 && end != i) {
                swap(nums[end], nums[i]);
                end--;
                i--;
            }
        }
    }
};
```

<span id="最小覆盖子串"></span>
## [76、最小覆盖子串](#back)
```cpp
给你一个字符串 S、一个字符串 T，请在字符串 S 里面找出：包含 T 所有字母的最小子串。

输入: S = "ADOBECODEBANC", T = "ABC"
输出: "BANC"

说明：
如果 S 中不存这样的子串，则返回空字符串 ""。
如果 S 中存在这样的子串，我们保证它是唯一的答案。

class Solution {
public:
    string minWindow(string s, string t) {
        // 双指针滑窗方法
        // 1、指针left, right; 指针right向右遍历直到出现满足条件的子串
        // 2、找到子串后，指针left再向右遍历，直到窗口中的内容不满足条件
        // 3、重复1,2；
        unordered_map<char, int> window;  // 滑窗中含有模式串中字符的情况
        unordered_map<char, int> needs;  // 记录模式串中的字符情况
        string res;
        int start = 0, minLen = INT_MAX;  // 为了截取子串
        int left = 0, right = 0;  // 双指针
        for (char t_ : t) needs[t_]++;
        int match = 0;  // 用于判断匹配的条件
        while (right < s.size()) {
            char s_ = s[right];
            if (needs.count(s_) != 0) {
                window[s_]++;
                if (window[s_] == needs[s_]) match++;
            }
            right++;
            while (match == needs.size()) {  // 当滑窗中的内容符合条件
                if (right - left < minLen) {
                    start = left;
                    minLen = right - left;
                }
                char s1_ = s[left];
                if (needs.count(s1_) != 0) {
                    window[s1_]--;
                    if (window[s1_] < needs[s1_]) match--;
                }
                left++;
            }
        }
        return minLen == INT_MAX ? "" : s.substr(start, minLen);
    }
};
```

<span id="组合"></span>
## [77、组合](#back)
```cpp
给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。

输入: n = 4, k = 2
输出:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]

class Solution {
public:
    vector<vector<int>> ans;
    vector<int> tmp;
    void find(int a, int n, int k) {
        if (k <= 0) {
            ans.push_back(tmp);
            return ;
        }
        for (int i = a; i <= n; i++) {
            tmp.push_back(i);
            find(i + 1, n, k - 1);
            tmp.pop_back();
        }
    }

    vector<vector<int>> combine(int n, int k) {
    	// 回溯法（递归方式的dfs）
        find(1, n, k);
        return ans;
    }
};
```

<span id="子集"></span>
## [78、子集](#back)
```cpp
给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
说明：解集不能包含重复的子集。

输入: nums = [1,2,3]
输出:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]

class Solution {
public:
    vector<vector<int>> res;
    vector<int> vec;

    void find(int step, int num, vector<int> nums) {
        if (num <= nums.size()) res.push_back(vec);
        for (int i = step; i < nums.size(); i++) {
            vec.push_back(nums[i]);
            find(i + 1, num + 1, nums);
            vec.pop_back();
        }
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        find(0, 0, nums);
        return res;
    }
};
```

<span id="单词搜索"></span>
## [79、单词搜索](#back)
```cpp
给定一个二维网格和一个单词，找出该单词是否存在于网格中。
单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

给定 word = "ABCCED", 返回 true.
给定 word = "SEE", 返回 true.
给定 word = "ABCB", 返回 false.

class Solution {
public:
    int dir[4][4] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    bool dfs(int x, int y, int index, vector<vector<char>>& board, string &word, vector<vector<bool>>& flag) {
        if (index == word.size() - 1) return word[index] == board[x][y];
        if (word[index] == board[x][y]) {
            flag[x][y] = true;
            for (int i = 0; i < 4; i++) {
                int new_x = x + dir[i][0];
                int new_y = y + dir[i][1];
                if (new_x >= 0 && new_x < board.size() && new_y >= 0 && new_y < board[0].size() && !flag[new_x][new_y])
                    if(dfs(new_x, new_y, index + 1, board, word, flag))
                        return true;
            }
            flag[x][y] = false;
        }
        return false;
    }

    bool exist(vector<vector<char>>& board, string word) {
        int rows = board.size();
        int cols = board[0].size();
        vector<vector<bool>> flag(rows, vector<bool> (cols, false));

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if(dfs(i, j, 0, board, word, flag))
                    return true;
            }
        }
        return false;
    }
};
```

<span id="删除排序数组中的重复项2"></span>
## [80、删除排序数组中的重复项 II](#back)
```cpp
给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素最多出现两次，返回移除后数组的新长度。
不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。

给定 nums = [1,1,1,2,2,3],
函数应返回新长度 length = 5, 并且原数组的前五个元素被修改为 1, 1, 2, 2, 3 。
你不需要考虑数组中超出新长度后面的元素。

给定 nums = [0,0,1,1,1,1,2,3,3],
函数应返回新长度 length = 7, 并且原数组的前五个元素被修改为 0, 0, 1, 1, 2, 3, 3 。
你不需要考虑数组中超出新长度后面的元素。

class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        if (nums.size() <= 1) return nums.size();
        int i = 1, j = 1, count_ = 1;
        while(j < nums.size()) {
            if (nums[j - 1] == nums[j]) count_ += 1;
            else count_ = 1;
            if (count_ <= 2) {
                nums[i] = nums[j];
                i++;
            }
            j++;
        }
        return i;
    }
};
```

<span id="搜索旋转排序数组2"></span>
## [81、搜索旋转排序数组 II](#back)
```cpp
假设按照升序排序的数组在预先未知的某个点上进行了旋转。
( 例如，数组 [0,0,1,2,2,5,6] 可能变为 [2,5,6,0,0,1,2] )。
编写一个函数来判断给定的目标值是否存在于数组中。若存在返回 true，否则返回 false。

输入: nums = [2,5,6,0,0,1,2], target = 0
输出: true

输入: nums = [2,5,6,0,0,1,2], target = 3
输出: false

进阶:
这是 搜索旋转排序数组 的延伸题目，本题中的 nums  可能包含重复元素。
这会影响到程序的时间复杂度吗？会有怎样的影响，为什么？

class Solution {
public:
    bool search(vector<int>& nums, int target) {
    	// 二分查找
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = left + right >> 1;
            if (nums[mid] == target) return true;
            if (nums[left] == nums[mid]) {
                left++;
                continue;
            }
            if (nums[left] < nums[mid]) {  // 左半边有序
                if (target < nums[mid] && nums[left] <= target) right = mid - 1;
                else left = mid + 1;

            }
            else {  // 右半边有序
                if (nums[mid] < target && target <= nums[right]) left = mid + 1;
                else right = mid - 1;
            }
        }
        return false;
    }
};
```

<span id="删除排序链表中的重复元素2"></span>
## [82、删除排序链表中的重复元素 II](#back)
```cpp
给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中 没有重复出现 的数字。

输入: 1->2->3->3->4->4->5
输出: 1->2->5

输入: 1->1->1->2->3
输出: 2->3

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
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode* pDummy = new ListNode(0);
        pDummy -> next = head;
        ListNode* p = pDummy;
        while(p -> next) {
            ListNode* tmp = p -> next;
            while (tmp && p -> next ->val == tmp -> val) 
                tmp = tmp -> next;
            if (p -> next -> next == tmp) 
                p = p -> next;
            else p -> next = tmp;
        }
        return pDummy -> next;
    }
};
```

<span id="删除排序链表中的重复元素"></span>
## [83、删除排序链表中的重复元素](#back)
```cpp
给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。

输入: 1->1->2
输出: 1->2

输入: 1->1->2->3->3
输出: 1->2->3

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
    ListNode* deleteDuplicates(ListNode* head) {
        // // 方法1
        // ListNode* pDummy = new ListNode(0);
        // pDummy -> next = head;
        // ListNode* p = pDummy;
        // while(p -> next) {
        //     ListNode* tmp = p -> next;
        //     ListNode* tmp_ = tmp;
        //     while (tmp && p -> next ->val == tmp -> val) {
        //         tmp_ = tmp;
        //         tmp = tmp -> next;
        //     }
        //     if (p -> next -> next == tmp) 
        //         p = p -> next;
        //     else p -> next = tmp_;
        // }
        // return pDummy -> next;
        
        // 方法2
        ListNode* p = head;
        while (p != NULL && p -> next != NULL) {
            if (p -> next -> val == p -> val) {
                p -> next = p -> next -> next;
            }
            else p = p -> next;
        }
        return head;
    }
};
```

<span id="柱状图中最大的矩形"></span>
## [84、柱状图中最大的矩形](#back)
```
给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
求在该柱状图中，能够勾勒出来的矩形的最大面积。
```

<div align=center><img src=https://github.com/FangChao1086/LeetCode_Solutions/blob/master/依赖文件/leetcode84_柱状图1.png></div>

```
以上是柱状图的示例，其中每个柱子的宽度为 1，给定的高度为 [2,1,5,6,2,3]。
```

<div align=center><img src=https://github.com/FangChao1086/LeetCode_Solutions/blob/master/依赖文件/leetcode84_柱状图2.png></div>

```cpp
图中阴影部分为所能勾勒出的最大矩形面积，其面积为 10 个单位。

输入: [2,1,5,6,2,3]
输出: 10

class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        // 分治法
        // 找到最小高度的索引值 min_index，用其得到使用此索引的最大面积 min_index* (end - start + 1)
        // 在索引左边（不包括索引 min_index ）再继续找最小高度的索引，计算左边的最大面积
        // 同样在右边也找到一个最大面积
        //  3 者的最大面积即为所求
        return fenzhi(heights, 0, heights.size() - 1);
    }

    int fenzhi(vector<int>& heights, int start, int end) {
        if (start > end) return 0;
        int min_index = start;
        for (int i = start; i <= end; i++) 
            if (heights[min_index] > heights[i])
                min_index = i;
        int left_min = min_index - 1;
        int right_min = min_index + 1;
        // 找到左边比分界线大的数
        while (left_min > 0 && heights[left_min] <= heights[min_index]) left_min--;
        while (right_min < end && heights[right_min] <= heights[min_index]) right_min++;
        return max(heights[min_index] * (end - start + 1), max(fenzhi(heights, start, left_min), fenzhi(heights, right_min, end)));
    }
};
```

<span id="最大矩形"></span>
## [85、最大矩形](#back)
```cpp
给定一个仅包含 0 和 1 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。

输入:
[
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
输出: 6

class Solution {
public:
    void update(vector<vector<vector<int>>>& dp, int i, int j) {
        int line_min = dp[i][j][0];
        int rows  = dp[i][j][1];
        for (int count_ = 0; count_ < rows; count_++) {
            line_min = min(line_min, dp[i - count_][j][0]);
            dp[i][j][2] = max(dp[i][j][2], line_min * (count_ + 1));
        }
    }

    int maximalRectangle(vector<vector<char>>& matrix) {
        // 动态规划 参考：https://leetcode-cn.com/problems/maximal-rectangle/solution/geng-zhi-bai-yi-dian-de-dong-tai-gui-hua-by-vsym/
        if (matrix.size() == 0) return 0;
        int rows = matrix.size();
        int cols = matrix[0].size();
        vector<vector<vector<int>>> dp(rows, vector<vector<int>> (cols, {0, 0, 0}));
        int res = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == '0') ;
                else {
                    if (i == 0 && j == 0)
                        dp[i][j] = {1, 1, 1};
                    else if (i == 0)
                            dp[i][j] = {dp[i][j - 1][0] + 1, 1, dp[i][j - 1][2] + 1};
                    else if (j == 0) 
                            dp[i][j] = {1, dp[i - 1][j][1] + 1, dp[i - 1][j][2] + 1};
                    else {
                        dp[i][j][0] = dp[i][j - 1][0] + 1;
                        dp[i][j][1] = dp[i - 1][j][1] + 1;
                        update(dp, i, j);
                    }
                    res = max(res, dp[i][j][2]);
                }
            }
        }
        return res;
    }
};
```

<span id="分隔链表"></span>
## [86、分隔链表](#back)
```cpp
给定一个链表和一个特定值 x，对链表进行分隔，使得所有小于 x 的节点都在大于或等于 x 的节点之前。
你应当保留两个分区中每个节点的初始相对位置。

输入: head = 1->4->3->2->5->2, x = 3
输出: 1->2->2->4->3->5

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
    ListNode* partition(ListNode* head, int x) {
        // 双指针
        // first 存值小于 x 的节点
        // last 存值大于等于 x 的节点
        ListNode* first = new ListNode(0);
        ListNode* first_head = first;
        ListNode* last = new ListNode(0);
        ListNode* last_head = last;
        while (head != NULL) {
            if (head -> val < x) {
                first -> next = head;
                first = first -> next;
            }
            else {
                last -> next = head;
                last = last -> next;
            }
            head = head -> next;
        }
        last -> next = NULL;
        first -> next = last_head -> next;
        return first_head -> next;
    }
};
```

<span id="扰乱字符串"></span>
## [87、扰乱字符串](#back)
```cpp
给定一个字符串 s1，我们可以把它递归地分割成两个非空子字符串，从而将其表示为二叉树。

下图是字符串 s1 = "great" 的一种可能的表示形式。
    great
   /    \
  gr    eat
 / \    /  \
g   r  e   at
           / \
          a   t
在扰乱这个字符串的过程中，我们可以挑选任何一个非叶节点，然后交换它的两个子节点。

例如，如果我们挑选非叶节点 "gr" ，交换它的两个子节点，将会产生扰乱字符串 "rgeat" 。
    rgeat
   /    \
  rg    eat
 / \    /  \
r   g  e   at
           / \
          a   t
我们将 "rgeat” 称作 "great" 的一个扰乱字符串。

同样地，如果我们继续交换节点 "eat" 和 "at" 的子节点，将会产生另一个新的扰乱字符串 "rgtae" 。
    rgtae
   /    \
  rg    tae
 / \    /  \
r   g  ta  e
       / \
      t   a
我们将 "rgtae” 称作 "great" 的一个扰乱字符串。

给出两个长度相等的字符串 s1 和 s2，判断 s2 是否是 s1 的扰乱字符串。

输入: s1 = "great", s2 = "rgeat"
输出: true

输入: s1 = "abcde", s2 = "caebd"
输出: false

class Solution {
public:
    bool isScramble(string s1, string s2) {
        // 动态规划
        // 参考链接：https://leetcode-cn.com/problems/scramble-string/solution/c-dong-tai-gui-hua-by-da-li-wang-36/
        // dp[len][i][j] s1 中以 i 开始长度为 len 的字符串 与 s2 中以 j 开始长度为 len的字符串是否是干扰字符串
        if (s1.size() != s2.size()) return false;
        if (s1.empty()) return true;
        int N = s1.size();
        vector<vector<vector<bool>>> dp(N + 1, vector<vector<bool>> (N, vector<bool> (N, false)));
        for (int i = 0; i < N ; i++) {  // 初始化长度为 1 的字符，若此时 s1 与 s2 中的相等，则是一个干扰字符串
            for (int j = 0; j < N; j++) {
                dp[1][i][j] = s1[i] == s2[j];
            }
        }
        for (int len = 2; len <= N; len++) {
            for (int i = 0; i < N && i + len - 1 < N; i++) {
                for (int j = 0; j < N && j + len - 1 < N; j++) {
                    for (int k = 1; k < len; k++) {
                        if (dp[k][i][j] && dp[len - k][i + k][j + k]) {
                            dp[len][i][j] = true;
                            break;
                        }
                        if (dp[k][i][j + len - k] && dp[len - k][i + k][j]) {
                            dp[len][i][j] = true;
                            break;
                        }
                    }
                }
            }
        }
        return dp[N][0][0];
    }
};
```

<span id="合并两个有序数组"></span>
## [88、合并两个有序数组](#back)
```cpp
给定两个有序整数数组 nums1 和 nums2，将 nums2 合并到 nums1 中，使得 num1 成为一个有序数组。

说明:
初始化 nums1 和 nums2 的元素数量分别为 m 和 n。
你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。

输入:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3
输出: [1,2,2,3,5,6]

class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        if (n == 0) return ;
        if (m == 0) {
            nums1 = nums2;
            return ;
        }
        int len = m + n - 1;
        m--;n--;
        while (n >= 0 && m >= 0) {
            if (nums1[m] <= nums2[n]) {
                nums1[len--] = nums2[n--];
            }
            else {
                nums1[len--] = nums1[m--];
            }
        }
        if (m < 0) {
            while (n >= 0) {
                nums1[len--] = nums2[n--];
            }
        }
    }
};
```

<span id="格雷编码"></span>
## [89、格雷编码](#back)
```cpp
格雷编码是一个二进制数字系统，在该系统中，两个连续的数值仅有一个位数的差异。
给定一个代表编码总位数的非负整数 n，打印其格雷编码序列。格雷编码序列必须以 0 开头。

输入: 2
输出: [0,1,3,2]
解释:
00 - 0
01 - 1
11 - 3
10 - 2
对于给定的 n，其格雷编码序列并不唯一。
例如，[0,2,3,1] 也是一个有效的格雷编码序列。
00 - 0
10 - 2
11 - 3
01 - 1

输入: 0
输出: [0]
解释: 我们定义格雷编码序列必须以 0 开头。
     给定编码总位数为 n 的格雷编码序列，其长度为 2n。当 n = 0 时，长度为 20 = 1。
     因此，当 n = 0 时，其格雷编码序列为 [0]。

class Solution {
public:
    vector<int> grayCode(int n) {
        // 参考链接：https://leetcode-cn.com/problems/gray-code/solution/c-dong-tai-gui-hua-jian-ji-yi-dong-shi-jian-ji-kon/
        vector<int> result(1);
        result[0] = 0;
        for(int i = 1; i <= n; i++){
            int e = 1 << (i - 1);                           //i - 1位结果前增加一位1
            for(int j = result.size() - 1; j >= 0; j--){    // 镜像排列
                result.push_back(e + result[j]);
            }
        }
        return result;
    }
};
```

<span id="子集2"></span>
## [90、子集 II](#back)
```cpp
给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
说明：解集不能包含重复的子集。

输入: [1,2,2]
输出:
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]

class Solution {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        // 重复的数字在已经出现过的地方才能出现，在没有出现过的地方只能出现那个领头的数字
        vector<vector<int>> res = {{}};  // res.size() = 1;
        sort(nums.begin(), nums.end());
        int start = 0;
        for (int i = 0; i < nums.size(); i++) {
            start = (i && nums[i] == nums[i - 1]) ? start : 0;
            int len = res.size();
            for (int j = start; j < len; j++) {
                auto tmp = res[j];
                tmp.push_back(nums[i]);
                res.push_back(tmp);
            }
            start = len;
        }
        return res;
    }
};
```

<span id="解码方法"></span>
## [91、解码方法](#back)
```cpp
一条包含字母 A-Z 的消息通过以下方式进行了编码：
'A' -> 1
'B' -> 2
...
'Z' -> 26
给定一个只包含数字的非空字符串，请计算解码方法的总数。

输入: "12"
输出: 2
解释: 它可以解码为 "AB"（1 2）或者 "L"（12）。
示例 2:

输入: "226"
输出: 3
解释: 它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。

class Solution {
public:
    int numDecodings(string s) {
        // 动态规划
        // dp[i] : str[0...i] 的方法数
        if (s[0] == '0') return 0;
        int cur = 1, pre = 1;
        for (int i = 1; i < s.size(); i++) {
            int tmp = cur;
            if (s[i] == '0') {
                if (s[i - 1] == '1' || s[i - 1] == '2') cur = pre;
                else return 0;
            }
            else if (s[i - 1] == '1' || (s[i - 1] == '2' && s[i] >= '1' && s[i] <= '6'))
                cur += pre;
            pre = tmp;
        }
        return cur;
    }
};
```

<span id="反转链表2"></span>
## [92、反转链表 II](#back)
```cpp
反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。
说明:
1 ≤ m ≤ n ≤ 链表长度。

输入: 1->2->3->4->5->NULL, m = 2, n = 4
输出: 1->4->3->2->5->NULL

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
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        ListNode* pDummy  = new ListNode(0);
        pDummy -> next = head;
        auto left = pDummy, right = head;
        int cnt = m - 1; 
        while (cnt--) {
            left = right;
            right = right -> next;
        }
        auto savehead = left;
        auto savetail = right;
        left = right;
        right = right -> next;
        cnt = n - m;
        while (cnt--) {
            auto next_ = right -> next;
            right -> next = left;
            left = right;
            right = next_;
        }
        savehead -> next = left;
        savetail -> next = right;
        return pDummy -> next;
    }
};
```

<span id="复原IP地址"></span>
## [93、复原IP地址](#back)
```cpp
给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式。

输入: "25525511135"
输出: ["255.255.11.135", "255.255.111.35"]

class Solution {
public:
    vector<string> restoreIpAddresses(string s) {
        string ip;
        helper(s, 0, ip);
        return res;
    }

    void helper(string s, int n, string ip) {
        if (n == 4) {
            if (s.empty()) res.push_back(ip);
        }
        else {
            for (int k = 1; k < 4; k++) {
                if (s.size() < k) break;
                int val = stoi(s.substr(0, k));
                if (val > 255 ||  k != std::to_string(val).size()) continue;  // 010 这种错误
                helper(s.substr(k), n + 1, ip + s.substr(0, k) + (n == 3 ? "" : "."));
            }
        }
        return ;
    }

private:
    vector<string> res;
};
```

<span id="二叉树的中序遍历"></span>
## [94、二叉树的中序遍历](#back)
```cpp
给定一个二叉树，返回它的中序 遍历。

输入: [1,null,2,3]
   1
    \
     2
    /
   3

输出: [1,3,2]
进阶: 递归算法很简单，你可以通过迭代算法完成吗？

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        mid(root, res);
        return res;
    }

    // void mid(TreeNode* root, vector<int>& res) {
    //     if (root == NULL) return;
    //     mid(root -> left, res);
    //     res.push_back(root -> val);
    //     mid(root -> right, res);
    // }

    void mid(TreeNode* root, vector<int>& res) {
        // 方法2 非递归
        stack<TreeNode*> s;
        TreeNode* cur = root;
        while(!s.empty() || cur != NULL) {
            if (cur != NULL) {
                s.push(cur);
                cur = cur -> left;
            }
            else {
                cur = s.top();
                res.push_back(cur ->val);
                cur = cur -> right;
                s.pop();
            }
        }
    }
};
```

<span id="不同的二叉搜索树2"></span>
## [95、不同的二叉搜索树 II](#back)
```cpp
给定一个整数 n，生成所有由 1 ... n 为节点所组成的二叉搜索树。

输入: 3
输出:
[
  [1,null,3,2],
  [3,2,null,1],
  [3,1,null,null,2],
  [2,1,3],
  [1,null,2,null,3]
]
解释:
以上的输出对应以下 5 种不同结构的二叉搜索树：

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<TreeNode*> helper(int start, int end) {
        vector<TreeNode*> res;
        if (start > end) res.push_back(NULL);
        for (int i = start; i <= end; i++) {
            vector<TreeNode*> left = helper(start, i - 1);
            vector<TreeNode*> right = helper(i + 1, end);
            for (auto l : left) {
                for (auto r : right) {
                    TreeNode* root = new TreeNode(i);
                    root -> left = l;
                    root -> right = r;
                    res.push_back(root);
                }
            }
        }
        return res;
    }

    vector<TreeNode*> generateTrees(int n) {
        vector<TreeNode*> res;
        if (n == 0) return res;
        res = helper(1, n); 
        return res;
    }
};
```

<span id="不同的二叉搜索树"></span>
## [96、不同的二叉搜索树](#back)
```cpp
给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？

输入: 3
输出: 5
解释:
给定 n = 3, 一共有 5 种不同结构的二叉搜索树:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3

class Solution {
public:
    int numTrees(int n) {
        // 动态规划
        // 每一个节点 i 都是左边子树 1, 2, .. ,i - 1 和右边子树 i, i + 1, ..., n 这两种的乘积
        vector<int> dp(n + 1, 0);
        dp[0] = 1, dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }        
        return dp[n];
    }
};
```

<span id="交错字符串"></span>
## [97、交错字符串](#back)
```cpp
给定三个字符串 s1, s2, s3, 验证 s3 是否是由 s1 和 s2 交错组成的。

输入: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
输出: true

输入: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
输出: false

class Solution {
public:
    bool isInterleave(string s1, string s2, string s3) {
        // 动态规划
        int N1 = s1.size();
        int N2 = s2.size();
        int N3 = s3.size();
        if (N1 + N2 != N3) return false;
        vector<vector<bool>> dp(N1 + 1, vector<bool> (N2 + 1, false));
        dp[0][0] = true;
        for (int i = 0; i <= N1; i++) {
            for (int j = 0; j <= N2; j++) {
                if (i > 0 && s1[i - 1] == s3[i + j - 1])
                    dp[i][j] = dp[i][j] || dp[i - 1][j];
                if (j > 0 && s2[j - 1] == s3[i + j - 1])
                    dp[i][j] = dp[i][j] || dp[i][j - 1];
            }
        }
        return dp[N1][N2];
    }
};
```

<span id="验证二叉搜索树"></span>
## [98、验证二叉搜索树](#back)
```cpp
给定一个二叉树，判断其是否是一个有效的二叉搜索树。
假设一个二叉搜索树具有如下特征：
节点的左子树只包含小于当前节点的数。
节点的右子树只包含大于当前节点的数。
所有左子树和右子树自身必须也是二叉搜索树。

输入:
    2
   / \
  1   3
输出: true

输入:
    5
   / \
  1   4
     / \
    3   6
输出: false
解释: 输入为: [5,1,4,null,null,3,6]。
     根节点的值为 5 ，但是其右子节点值为 4 。

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool helper(TreeNode* root, long low, long high) {
        if (root == NULL) return true;
        long val = root -> val;
        if (val <= low || val >= high) return false;
        return helper(root -> left, low, val) && helper(root -> right, val, high);
    }

    bool isValidBST(TreeNode* root) {
        return helper(root, LONG_MIN, LONG_MAX);  // 上界下界
    }
};
```

<span id="恢复二叉搜索树"></span>
## [99、恢复二叉搜索树](#back)
```cpp
二叉搜索树中的两个节点被错误地交换。
请在不改变其结构的情况下，恢复这棵树。

输入: [1,3,null,null,2]

   1
  /
 3
  \
   2
输出: [3,1,null,null,2]

   3
  /
 1
  \
   2

输入: [3,1,4,null,null,2]

  3
 / \
1   4
   /
  2
输出: [2,1,4,null,null,3]

  2
 / \
1   4
   /
  3

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    void recoverTree(TreeNode* root) {
    	// 参考链接：https://leetcode-cn.com/problems/recover-binary-search-tree/solution/yin-yong-chang-shu-kong-jian-jie-jue-by-newbie-19/
        vector<TreeNode*> vec;
        TreeNode* pre = NULL;
        recoverTree(root, vec, pre);
        if (vec.size() == 2) {
            int tmp = vec[0] -> val;
            vec[0] -> val = vec[1] -> val;
            vec[1] -> val = tmp;
        }
        else {
            int tmp = vec[0] -> val;
            vec[0] -> val = vec[2] -> val;
            vec[2] -> val = tmp;
        }
        
    }

    void recoverTree(TreeNode* root, vector<TreeNode*>& vec, TreeNode*& pre) {
        if (!root) return ;
        recoverTree(root -> left, vec, pre);
        if (pre && vec.size() == 0) {
            if (root -> val < pre -> val) {
                vec.push_back(pre);
                vec.push_back(root);
            }
        }
        else if (vec.size() == 2 && (pre && root -> val < pre -> val))
            vec.push_back(root);
        pre = root;
        recoverTree(root -> right, vec, pre);
    }
};
```

<span id="相同的树"></span>
## [100、相同的树](#back)
```cpp
给定两个二叉树，编写一个函数来检验它们是否相同。
如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

输入:       1         1
          / \       / \
         2   3     2   3

        [1,2,3],   [1,2,3]

输出: true

输入:      1          1
          /           \
         2             2

        [1,2],     [1,null,2]

输出: false

输入:       1         1
          / \       / \
         2   1     1   2

        [1,2,1],   [1,1,2]

输出: false

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if (p == NULL && q == NULL) return true;
        else if (p == NULL || q == NULL ||p -> val != q -> val) return false;
        return isSameTree(p -> left, q -> left) && isSameTree(p -> right, q -> right);
    }
};
```

<span id="对称二叉树"></span>
## [101、对称二叉树](#back)
```cpp
给定一个二叉树，检查它是否是镜像对称的。
例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
    1
   / \
  2   2
 / \ / \
3  4 4  3
但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:
    1
   / \
  2   2
   \   \
   3    3
说明:
如果你可以运用递归和迭代两种方法解决这个问题，会很加分。

/**/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        // return isSymmetric(root, root);  // 常用递归，简单

        // 迭代
        // 每次取 que 中两个节点进行比较；将左右节点的子左右节点按相反（右左）方向压进 que 中；
        queue<TreeNode*> que;
        que.push(root);
        que.push(root);
        while (!que.empty()) {
            TreeNode* t1 = que.front();
            que.pop();
            TreeNode* t2 = que.front();
            que.pop();
            if (t1 == NULL && t2 == NULL) continue;
            if (t1 == NULL || t2 == NULL || t1 -> val != t2 -> val) return false;
            que.push(t1 -> left);
            que.push(t2 -> right);
            que.push(t1 -> right);
            que.push(t2 -> left);
        }
        return true;
    }
    
    // 递归辅助函数
    bool isSymmetric(TreeNode* root1, TreeNode* root2) {
        if (root1 == NULL && root2 == NULL) return true;
        if (root1 == NULL || root2 == NULL || root1 -> val != root2 -> val) return false;
        return isSymmetric(root1 -> left, root2 -> right) && isSymmetric(root1 -> right, root2 -> left);
    }
};
```

<span id="二叉树的层次遍历"></span>
## [102、二叉树的层次遍历](#back)
```cpp
给定一个二叉树，返回其按层次遍历的节点值。 （即逐层地，从左到右访问所有节点）。

给定二叉树: [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
返回其层次遍历结果：
[
  [3],
  [9,20],
  [15,7]
]

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if (root == NULL) return res;
        queue<TreeNode*> que;
        que.push(root);
        while (!que.empty()) {
            int len_T = que.size();
            vector<int> vec;
            for (int i = 0; i < len_T; i++) {
                TreeNode* tmp = que.front();
                que.pop();
                vec.push_back(tmp -> val);
                if (tmp -> left) que.push(tmp -> left);
                if (tmp -> right) que.push(tmp -> right);
            }
            res.push_back(vec);
        } 
        return res;
    }
};
```

<span id="二叉树的锯齿形层次遍历"></span>
## [103、二叉树的锯齿形层次遍历](#back)
```cpp
给定一个二叉树，返回其节点值的锯齿形层次遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

给定二叉树 [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
返回锯齿形层次遍历如下：
[
  [3],
  [20,9],
  [15,7]
]

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if (root == NULL) return res;
        queue<TreeNode*> que;
        que.push(root);
        bool flag = false;
        while (!que.empty()) {
            int len_Q = que.size();
            vector<int> vec;
            for (int i = 0; i < len_Q; i++) {
                TreeNode* tmp = que.front();
                que.pop();
                vec.push_back(tmp -> val);
                if (tmp -> left) que.push(tmp -> left);
                if (tmp -> right) que.push(tmp -> right);
            }
            if (flag) reverse(vec.begin(), vec.end());
            res.push_back(vec);
            flag = !flag;
        }
        return res;
    }
};
```

<span id="二叉树的最大深度"></span>
## [104、二叉树的最大深度](#back)
```cpp
给定一个二叉树，找出其最大深度。
二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
说明: 叶子节点是指没有子节点的节点。

给定二叉树 [3,9,20,null,null,15,7]，
    3
   / \
  9  20
    /  \
   15   7
返回它的最大深度 3 

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (root == NULL) return 0;
        int left = maxDepth(root -> left);
        int right = maxDepth(root -> right);
        return max(left + 1, right + 1);
    }
};
```

<span id="从前序与中序遍历序列构造二叉树"></span>
## [105、从前序与中序遍历序列构造二叉树](#back)
```cpp
根据一棵树的前序遍历与中序遍历构造二叉树。
你可以假设树中没有重复的元素。

前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
返回如下的二叉树：
    3
   / \
  9  20
    /  \
   15   7

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        if(preorder.size() == 0 || preorder.size() != inorder.size()) return NULL;
        int index_i = 0;
        for (int i = 0; i < inorder.size(); i++) {
            if (inorder[i] == preorder[0]) {
                index_i = i;
                break;
            }
        }
        TreeNode* root = new TreeNode(preorder[0]);
        vector<int> pre_left, pre_right, in_left, in_right;
        for (int i = 0; i < index_i; i++) {
            pre_left.push_back(preorder[i + 1]);
            in_left.push_back(inorder[i]);
        }
        for (int i = index_i + 1; i < preorder.size(); i++) {
            pre_right.push_back(preorder[i]);
            in_right.push_back(inorder[i]);
        }
        root -> left = buildTree(pre_left, in_left);
        root -> right = buildTree(pre_right, in_right);
        return root;
    }
};
```

<span id="从中序与后序遍历序列构造二叉树"></span>
## [106、从中序与后序遍历序列构造二叉树](#back)
```cpp
根据一棵树的中序遍历与后序遍历构造二叉树。
你可以假设树中没有重复的元素。

中序遍历 inorder = [9,3,15,20,7]
后序遍历 postorder = [9,15,7,20,3]
返回如下的二叉树：
    3
   / \
  9  20
    /  \
   15   7

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        if (inorder.size() == 0 || inorder.size() != postorder.size()) return NULL;
        int index_i = 0; 
        for (int i = inorder.size() - 1; i >= 0; i--) {
            if (inorder[i] == postorder[postorder.size() - 1]) {
                index_i = i;
                break;
            }
        }
        TreeNode* root = new TreeNode(postorder[postorder.size() - 1]);
        vector<int> in_left, in_right, post_left, post_right;
        for (int i = 0; i < index_i; i++) {
            in_left.push_back(inorder[i]);
            post_left.push_back(postorder[i]);
        }
        for (int i = index_i; i < inorder.size() - 1; i++) {
            in_right.push_back(inorder[i + 1]);
            post_right.push_back(postorder[i]);
        }
        root ->left = buildTree(in_left, post_left);
        root -> right = buildTree(in_right, post_right);
        return root;
    }
};
```

<span id="二叉树的层次遍历2"></span>
## [107、二叉树的层次遍历 II](#back)
```cpp
给定一个二叉树，返回其节点值自底向上的层次遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）

给定二叉树 [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
返回其自底向上的层次遍历为：
[
  [15,7],
  [9,20],
  [3]
]

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrderBottom(TreeNode* root) {
        vector<vector<int>> res;
        if (root == NULL) return res;
        queue<TreeNode*> que;
        que.push(root);
        while (!que.empty()) {
            int len_q = que.size();
            vector<int> vec;
            for (int i = 0; i < len_q; i++) {
                TreeNode* tmp = que.front();
                que.pop();
                vec.push_back(tmp -> val);
                if (tmp -> left) que.push(tmp -> left);
                if (tmp -> right) que.push(tmp -> right);
            }
            res.push_back(vec);
        }
        reverse(res.begin(), res.end());
        return res;
    }
};
```

<span id="将有序数组转换为二叉搜索树"></span>
## [108、将有序数组转换为二叉搜索树](#back)
```cpp
将一个按照升序排列的有序数组，转换为一棵高度平衡二叉搜索树。
本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。

给定有序数组: [-10,-3,0,5,9],
一个可能的答案是：[0,-3,9,-10,null,5]，它可以表示下面这个高度平衡二叉搜索树：
      0
     / \
   -3   9
   /   /
 -10  5

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        if (nums.size() == 0) return NULL;
        if (nums.size() == 1) {
            return new TreeNode(nums[0]);
        }
        TreeNode* root = new TreeNode(nums[(nums.size()) / 2]);
        vector<int> nums1;
        vector<int> nums2;
        for (int i = 0; i < (nums.size()) / 2; i++) nums1.push_back(nums[i]);
        for (int i = (nums.size()) / 2 + 1 ; i < nums.size(); i++) nums2.push_back(nums[i]);
        root -> left = sortedArrayToBST(nums1);
        root -> right = sortedArrayToBST(nums2);
        return root;
    }
};
```

<span id="有序链表转换二叉搜索树"></span>
## [109、有序链表转换二叉搜索树](#back)
```cpp
给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。
本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。

给定的有序链表： [-10, -3, 0, 5, 9],
一个可能的答案是：[0, -3, 9, -10, null, 5], 它可以表示下面这个高度平衡二叉搜索树：
      0
     / \
   -3   9
   /   /
 -10  5

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* sortedListToBST(ListNode* head) {
        if (head == NULL) return NULL;
        if (head -> next == NULL) return new TreeNode(head -> val);
        ListNode* L1 = head;
        int index_i = 0;
        for (; L1 != NULL; index_i++) 
            L1 = L1 -> next;
        L1 = head;
        for (int i = 1 ; i < index_i / 2; i++) {
            L1 = L1 -> next;
        }
        TreeNode* root = new TreeNode(L1 -> next -> val);
        ListNode* L_last = L1 -> next -> next;
        L1 -> next = NULL;
        root -> left = sortedListToBST(head);  // 此时的 head 只包含前半部分，如 [-10,-3,0,5,9] 中的 [-10, -3]
        root -> right = sortedListToBST(L_last);
        return root;        
    }
};
```

<span id="平衡二叉树"></span>
## [110、平衡二叉树](#back)
```cpp
给定一个二叉树，判断它是否是高度平衡的二叉树。
本题中，一棵高度平衡二叉树定义为：
一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过1。


给定二叉树 [3,9,20,null,null,15,7]
    3
   / \
  9  20
    /  \
   15   7
返回 true 。

给定二叉树 [1,2,2,3,3,null,null,4,4]
       1
      / \
     2   2
    / \
   3   3
  / \
 4   4
返回 false 。

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isBalanced(TreeNode* root) {
        if (root == NULL) return true;
        int left = getDepth(root -> left);
        int right = getDepth(root -> right);
        if (abs(left - right) > 1) return false;
        return isBalanced(root -> left) && isBalanced(root -> right); 
    }

    int getDepth(TreeNode* root) {
        if (root == NULL) return 0;
        int left = getDepth(root -> left);
        int right = getDepth(root -> right);
        return max(left, right) + 1;
    }
};
```

<span id="二叉树的最小深度"></span>
## [111、二叉树的最小深度](#back)
```cpp
给定一个二叉树，找出其最小深度。
最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
说明: 叶子节点是指没有子节点的节点。

给定二叉树 [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
返回它的最小深度  2.

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int minDepth(TreeNode* root) {
        if (root == NULL) return 0;
        int left = minDepth(root -> left);
        int right = minDepth(root -> right);
        if (root -> left == NULL || root -> right == NULL)  // 非叶子节点
            return left == 0 ? right + 1 : left + 1;
        else 
            return min(left, right) + 1;
    }
};
```

<span id="路径总和"></span>
## [112、路径总和](#back)
```cpp
给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。
说明: 叶子节点是指没有子节点的节点。

给定如下二叉树，以及目标和 sum = 22，
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \      \
        7    2      1
返回 true, 因为存在目标和为 22 的根节点到叶子节点的路径 5->4->11->2。

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool hasPathSum(TreeNode* root, int sum) {
        if (root == NULL) return false;
        if (root -> left == NULL && root -> right == NULL) {
            return sum - (root -> val) == 0;
        }
        return hasPathSum(root -> left, sum - (root -> val)) || hasPathSum(root -> right, sum - (root -> val));
    }
};
```

<span id="路径总和2"></span>
## [113、路径总和 II](#back)
```cpp
给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。
说明: 叶子节点是指没有子节点的节点。

给定如下二叉树，以及目标和 sum = 22，
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1
返回:
[
   [5,4,11,2],
   [5,8,4,5]
]

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> res;
    vector<int> vec;

    void pathSum_helper(TreeNode* root, int sum) {
        if (root == NULL) return ;
        vec.push_back(root -> val);
        if (root -> left == nullptr && root -> right == nullptr && sum - (root -> val) == 0) {
            res.push_back(vec);
        }
        pathSum(root -> left, sum - (root -> val));
        pathSum(root -> right, sum -(root -> val));
        vec.pop_back();
    }

    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        pathSum_helper(root, sum);
        return res;        
    }
};
```

<span id="二叉树展开为链表"></span>
## [114、二叉树展开为链表](#back)
```cpp
给定一个二叉树，原地将它展开为链表。

例如，给定二叉树
    1
   / \
  2   5
 / \   \
3   4   6
将其展开为：
1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    void flatten(TreeNode* root) {
        while (root != nullptr) {
            if (root -> left != nullptr) {
                auto left_root = root -> left;
                while (left_root -> right != nullptr) left_root = left_root -> right;
                left_root -> right = root -> right;
                root -> right = root -> left;
                root -> left = nullptr;
            }
            root = root -> right;
        }
    }
};
```

<span id="不同的子序列"></span>
## [115、不同的子序列](#back)
```cpp
给定一个字符串 S 和一个字符串 T，计算在 S 的子序列中 T 出现的个数。
一个字符串的一个子序列是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）

输入: S = "rabbbit", T = "rabbit"
输出: 3
解释:
如下图所示, 有 3 种可以从 S 中得到 "rabbit" 的方案。
(上箭头符号 ^ 表示选取的字母)
rabbbit
^^^^ ^^
rabbbit
^^ ^^^^
rabbbit
^^^ ^^^

输入: S = "babgbag", T = "bag"
输出: 5
解释:
如下图所示, 有 5 种可以从 S 中得到 "bag" 的方案。 
(上箭头符号 ^ 表示选取的字母)
babgbag
^^ ^
babgbag
^^    ^
babgbag
^    ^^
babgbag
  ^  ^^
babgbag
    ^^^

class Solution {
public:
    int numDistinct(string s, string t) {
        // 动态规划
        // dp[i][j] : S[:j] 中有 T[:i] 子序列的个数
        vector<vector<double>> dp(t.size() + 1, vector<double> (s.size() + 1, 0));
        for (int j = 0; j < s.size() + 1; j++) 
            dp[0][j] = 1;
        for (int i = 1; i < t.size() + 1; i++) {
            for (int j = 1; j < s.size() + 1; j++) {
                if (s[j - 1] == t[i - 1]) 
                    dp[i][j] = dp[i - 1][j - 1] + dp[i][j - 1];
                else 
                    dp[i][j] = dp[i][j - 1];
            }
        }
        return dp[t.size()][s.size()];
    }
};
```

<span id="填充每个节点的下一个右侧节点指针"></span>
## [116、填充每个节点的下一个右侧节点指针](#back)
<div align=center><img src="https://github.com/FangChao1086/LeetCode_Solutions/blob/master/依赖文件/116_填充每个节点的下一个右侧节点指针.png"></div>

```cpp
给定一个完美二叉树，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
初始状态下，所有 next 指针都被设置为 NULL。

/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* left;
    Node* right;
    Node* next;

    Node() : val(0), left(NULL), right(NULL), next(NULL) {}

    Node(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}

    Node(int _val, Node* _left, Node* _right, Node* _next)
        : val(_val), left(_left), right(_right), next(_next) {}
};
*/
class Solution {
public:
    Node* connect(Node* root) {
        // 层序遍历，每层最后一个节点指向 NULL, 其他指向当前层的下一个节点
        if (root == NULL) return root;
        queue<Node*> que;
        que.push(root);
        while (!que.empty()) {
            int len_q = que.size();
            for(int i = 1; i <= len_q; i++) {
                Node* tmp = que.front();
                que.pop();
                if (i == len_q) tmp -> next = NULL;
                else tmp -> next = que.front();
                if (tmp -> left) que.push(tmp -> left);
                if (tmp -> right) que.push(tmp -> right);
            }
        }
        return root;
    }
};
```

<span id="填充每个节点的下一个右侧节点指针2"></span>
## [117、填充每个节点的下一个右侧节点指针 II](#back)
```cpp
给定一个二叉树
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
初始状态下，所有 next 指针都被设置为 NULL。

/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* left;
    Node* right;
    Node* next;

    Node() : val(0), left(NULL), right(NULL), next(NULL) {}

    Node(int _val) : val(_val), left(NULL), right(NULL), next(NULL) {}

    Node(int _val, Node* _left, Node* _right, Node* _next)
        : val(_val), left(_left), right(_right), next(_next) {}
};
*/
class Solution {
public:
    Node* connect(Node* root) {
        // 层序遍历，每层最后一个节点指向 NULL, 其他指向当前层的下一个节点
	// 普通二叉树，解法可以用 116 题解法
        if (root == NULL) return root;
        queue<Node*> que;
        que.push(root);
        while (!que.empty()) {
            int len_q = que.size();
            for(int i = 1; i <= len_q; i++) {
                Node* tmp = que.front();
                que.pop();
                if (i == len_q) tmp -> next = NULL;
                else tmp -> next = que.front();
                if (tmp -> left) que.push(tmp -> left);
                if (tmp -> right) que.push(tmp -> right);
            }
        }
        return root;
    }
};
```

<span id="杨辉三角"></span>
## [118、杨辉三角](#back)

<div align=center><img src="https://github.com/FangChao1086/LeetCode_Solutions/blob/master/依赖文件/LeetCode118_杨辉三角.gif"></div>

```cpp
给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。
在杨辉三角中，每个数是它左上方和右上方的数的和。

输入: 5
输出:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]

class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> res(numRows);
        if (numRows == 0) return res;
        res[0].push_back(1);
        for(int i = 1; i < numRows; i++) {
            res[i].push_back(1);
            for (int j = 1; j < i; j++) {
                res[i].push_back(res[i - 1][j - 1] + res[i - 1][j]);
            }
            res[i].push_back(1);
        }
        return res;
    }
};
```

<span id="杨辉三角2"></span>
## [119、杨辉三角 II](#back)
```cpp
给定一个非负索引 k，其中 k ≤ 33，返回杨辉三角的第 k 行。
在杨辉三角中，每个数是它左上方和右上方的数的和。

输入: 3
输出: [1,3,3,1]

class Solution {
public:
    vector<int> getRow(int rowIndex) {
        // if (rowIndex == 0) return {1};
        // vector<vector<int>> res(rowIndex + 1);
        // res[0].push_back(1);
        // for (int i = 1; i < rowIndex + 1; i++) {
        //     res[i].push_back(1);
        //     for (int j = 1; j < i; j++) {
        //         res[i].push_back(res[i - 1][j - 1] + res[i - 1][j]);
        //     }
        //     res[i].push_back(1);
        // }
        // return res[rowIndex];

        // 更优解
        vector<int> result;
        for (int i = 0 ; i <= rowIndex; i++) {
            result.push_back(1); 
            for (int j = i - 1; j > 0; j--) {
                result[j] += result[j - 1];
            }
        }
        return result;
    }
};
```

<span id="三角形最小路径和"></span>
## [120、三角形最小路径和](#back)
```cpp
给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。

例如，给定三角形：
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。
说明：
如果你可以只使用 O(n) 的额外空间（n 为三角形的总行数）来解决这个问题，那么你的算法会很加分。

class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        vector<vector<int>> dp(triangle.size());
        dp[0].push_back(triangle[0][0]);
        for (int i = 1; i < triangle.size(); i++) {
            dp[i].push_back(dp[i - 1][0] + triangle[i][0]);
            for(int j = 1; j < i; j++) {
                dp[i].push_back(min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle[i][j]);
            }
            dp[i].push_back(dp[i - 1][i - 1] + triangle[i][i]);
        }
        int min_ = dp[dp.size() - 1][0];
        for (int i = 1; i < dp.size(); i++) {
            min_ = min(dp[dp.size() - 1][i], min_);
        }
        return min_;

    }
};
```

<span id="买卖股票的最佳时机"></span>
## [121、买卖股票的最佳时机(easy)](#back)
```cpp
给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法来计算你所能获取的最大利润。
注意你不能在买入股票前卖出股票。

输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。

输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.size() == 0) return 0;
        int min_ = prices[0], max_ = prices[0], max_all = 0;
        for (int i = 1;i < prices.size(); i++) {
            if (prices[i] < min_) {
                min_ = prices[i];
                max_ = prices[i];
            }
            if (prices[i] > max_) {
                max_ = prices[i];
            }
            max_all = max(max_ - min_, max_all);
        }
        return max_all;


    }
};
```

<span id="字典序排数"></span>
## [386、字典序排数](#back)
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
