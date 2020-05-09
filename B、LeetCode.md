<span id="back"></span>
# LeetCode_CPP版题目及答案
||||||
|-|-|-|-|-|
|[1、两数之和](#两数之和)|[2、两数相加](#两数相加) |[3、无重复字符的最长子串(medium)](#无重复字符的最长子串)| [4、寻找两个有序数组的中位数](#寻找两个有序数组的中位数)  |[5、最长回文子串](#最长回文子串) | 
|[6、Z字形变换](#Z字形变换)  |[7、整数反转](#整数反转)  |[8、字符串转换整数(atoi)](#字符串转换整数（atoi）)  |[9、回文数](#回文数)  |[10、正则表达式匹配](#正则表达式匹配)  |
|[11、盛最多水的容器(medium)](#盛最多水的容器)  |[12、整数转罗马数字](#整数转罗马数字)  |[13、罗马数字转整数](#罗马数字转整数)  |[14、最长公共前缀](#最长公共前缀)  |[15、三数之和](#三数之和)  |
|[16、最接近的三数之和](#最接近的三数之和)  |[17、电话号码的字母组合](#电话号码的字母组合)  |[18、四数之和](#四数之和)  |[19、删除链表的倒数第N个节点](#删除链表的倒数第N个节点)  |[20、有效的括号](#有效的括号)  |
|[21、合并两个有序链表(easy)](#合并两个有序链表)  |[22、括号生成(medium)](#括号生成)  |[23、合并K个排序链表(hard)](#合并K个排序链表)  |[24、两两交换链表中的节点](#两两交换链表中的节点)  |[25、K 个一组翻转链表](#K个一组翻转链表)  
|[26、删除排序数组中的重复项](#删除排序数组中的重复项)  |[27、移除元素](#移除元素)  |[28、实现str()](#实现str())  |[29、两数相除](#两数相除)  |[30、串联所有单词的子串](#串联所有单词的子串)  
|[31、下一个排列](#下一个排列)  |[32、最长有效括号](#最长有效括号)  |[33、搜索旋转排序数组(medium)](#搜索旋转排序数组)  |[34、排序数组查找元素的第一和最后一个位置](#排序数组查找元素的第一和最后一个位置)  |[35、搜索插入位置](#搜索插入位置)  
|[36、有效的数独](#有效的数独)  |[37、解数独](#解数独)  |[38、报数](#报数)  |[39、组合总和](#组合总和)  |[40、组合总和 II](#组合总和2)  
|[41、缺失的第一个正数](#缺失的第一个正数)  |[42、接雨水(hard)](#接雨水)  |[43、字符串相乘](#字符串相乘)  |[44、通配符匹配](#通配符匹配)  |[45、跳跃游戏 II](#跳跃游戏2)  
|[46、全排列(medium)](#全排列)  |[47、全排列 II](#全排列2)  |[48、旋转图像](#旋转图像)  |[49、 字母异位词分组](#字母异位词分组)  |[50、Pow(x, n)](#Pow)  
|[51、N皇后](#N皇后)  |[52、N皇后 II](#N皇后2)  |[53、最大子序和](#最大子序和)  |[54、螺旋矩阵](#螺旋矩阵)  |[55、跳跃游戏(medium)](#跳跃游戏)  
|[56、合并区间(medium)](#合并区间)  |[57、插入区间](#插入区间)  |[58、最后一个单词的长度](#最后一个单词的长度)  |[59、螺旋矩阵 II](#螺旋矩阵2)  |[60、第k个排列](#第k个排列)  
|[61、旋转链表](#旋转链表)  |[62、不同路径](#不同路径)  |[63、不同路径 II](#不同路径2)  |[64、最小路径和](#最小路径和)  |[65、有效数字](#有效数字)  
|[66、加一](#加一)  |[67、二进制求和](#二进制求和)  |[68、文本左右对齐](#文本左右对齐)  |[69、X的平方根(easy)](#X的平方根)  |[70、爬楼梯](#爬楼梯)  
|[71、简化路径](#简化路径)  |[72、编辑距离(hard)](#编辑距离)  |[73、矩阵置零](#矩阵置零)  |[74、搜索二维矩阵](#搜索二维矩阵)  |[75、颜色分类](#颜色分类)  
|[76、最小覆盖子串](#最小覆盖子串)  |[77、组合](#组合)  |[78、子集](#子集)  |[79、单词搜索](#单词搜索)  |[80、删除排序数组中的重复项 II](#删除排序数组中的重复项2)  
|[81、搜索旋转排序数组 II](#搜索旋转排序数组2)  |[82、删除排序链表中的重复元素 II](#删除排序链表中的重复元素2)  |[83、删除排序链表中的重复元素](#删除排序链表中的重复元素)  |[84、柱状图中最大的矩形](#柱状图中最大的矩形)  |[85、最大矩形](#最大矩形)  
|[86、分隔链表](#分隔链表)  |[87、扰乱字符串](#扰乱字符串)  |[88、合并两个有序数组](#合并两个有序数组)  |[89、格雷编码](#格雷编码)  |[90、子集 II](#子集2)  
|[91、解码方法](#解码方法)  |[92、反转链表 II](#反转链表2)  |[93、复原IP地址](#复原IP地址)  |[94、二叉树的中序遍历](#二叉树的中序遍历)  |[95、不同的二叉搜索树 II](#不同的二叉搜索树2)  
|[96、不同的二叉搜索树](#不同的二叉搜索树)  |[97、交错字符串](#交错字符串)  |[98、验证二叉搜索树(medium)](#验证二叉搜索树)  |[99、恢复二叉搜索树](#恢复二叉搜索树)  |[100、相同的树](#相同的树)  
|[101、对称二叉树](#对称二叉树)  |[102、二叉树的层次遍历](#二叉树的层次遍历)  |[103、二叉树的锯齿形层次遍历](#二叉树的锯齿形层次遍历)  |[104、二叉树的最大深度](#二叉树的最大深度)  |[105、从前序与中序遍历序列构造二叉树](#从前序与中序遍历序列构造二叉树)  
|[106、从中序与后序遍历序列构造二叉树](#从中序与后序遍历序列构造二叉树)  |[107、二叉树的层次遍历 II](#二叉树的层次遍历2)  |[108、将有序数组转换为二叉搜索树](#将有序数组转换为二叉搜索树)  |[109、有序链表转换二叉搜索树](#有序链表转换二叉搜索树)  |[110、平衡二叉树](#平衡二叉树)  
|[111、二叉树的最小深度](#二叉树的最小深度)  |[112、路径总和](#路径总和)  |[113、路径总和 II](#路径总和2)  |[114、二叉树展开为链表](#二叉树展开为链表)  |[115、不同的子序列](#不同的子序列)  
|[116、填充每个节点的下一个右侧节点指针](#填充每个节点的下一个右侧节点指针)  |[117、填充每个节点的下一个右侧节点指针 II](#填充每个节点的下一个右侧节点指针2)  |[118、杨辉三角](#杨辉三角)|[119、杨辉三角 II](#杨辉三角2)|[120、三角形最小路径和](#三角形最小路径和)|
|[121、买卖股票的最佳时机(easy)](#买卖股票的最佳时机)|[122、买卖股票的最佳时机 II(easy)](#买卖股票的最佳时机2)|[123、买卖股票的最佳时机 III(hard)](#买卖股票的最佳时机3)|[124、二叉树中的最大路径和(hard)](#二叉树中的最大路径和)|[125、验证字符串(easy)](#验证字符串)|
||[127、单词接龙(medium)](#单词接龙)|[128、最长连续序列(hard)](#最长连续序列)|[129、求根到叶子节点数字之和(medium)](#求根到叶子节点数字之和)|[130、被围绕的区域(medium)](#被围绕的区域)|
|[131、分割回文串(medium)](#分割回文串)|[132、分割回文串 II(hard)](#分割回文串2)|[133、克隆图(medium)](#克隆图)|[134、加油站(medium)](#加油站)|[135、分发糖果(hard)](#分发糖果)|
|[136、只出现一次的数字(easy)](#只出现一次的数字)|[137、只出现一次的数字 II(medium)](#只出现一次的数字2)|[138、复制带随机指针的链表(medium)](#复制带随机指针的链表)|[139、单词拆分(medium)](#单词拆分)||
|[141、环形链表(easy)](#环形链表)|[142、环形链表 II(medium)](#环形链表2)|[143、重排链表(medium)](#重排链表)|[144、二叉树的前序遍历(medium)](#二叉树的前序遍历)|[145、二叉树的后序遍历(hard)](#二叉树的后序遍历)|
|[146、LRU缓存机制(medium)](#LRU缓存机制)|[147、对链表进行插入排序(medium)](#对链表进行插入排序)|[148、排序链表(medium)](#排序链表)|[149、直线上最多的点数(hard)](#直线上最多的点数)|[150、逆波兰表达式求值(medium)](#逆波兰表达式求值)|
|[151、翻转字符串里的单词(medium)](#翻转字符串里的单词)|[152、乘积最大子序列(medium)](#乘积最大子序列)|[153、寻找旋转排序数组中的最小值(medium)](#寻找旋转排序数组中的最小值)|[154、寻找旋转排序数组中的最小值 II(hard)](#寻找旋转排序数组中的最小值2)|[155、最小栈(easy)](#最小栈)|
|||||[160、相交链表(easy)](#相交链表)|
||[162、寻找峰值(medium)](#寻找峰值)||
||||[169、多数元素(easy)](#多数元素)||
||||[199、二叉树的右视图(medium)](#二叉树的右视图)|[200、岛屿数量(medium)](#岛屿数量)|
||[202、快乐数(easy)](#快乐数)||||
|[206、反转链表(easy)](#反转链表)||
|[221、最大正方形(medium)](#最大正方形)||
||||[289、生命游戏(medium)](#生命游戏)||
|||||[300、最长上升子序列(medium)](#最长上升子序列)|
||[322、零钱兑换(medium)](#零钱兑换)||||
|||||[355、设计推特(medium)](#设计推特)|
|||||[365、水壶问题(medium)](#水壶问题)|
||[386、字典序排数](#字典序排数)  |
||||[409、最长回文串(easy)](#最长回文串)||
|||||[445、两数相加 II](#两数相加2)|
|||||[460、LFU缓存(hard)](#LFU缓存)|
|[466、统计重复个数(hard)](#统计重复个数)||||
||[542、01 矩阵(medium)](#01矩阵)|[543、二叉树的直径(easy)](#二叉树的直径)|
||[572、另一个树的子树(easy)](#另一个树的子树)||||
|||||[695、岛屿的最大面积(medium)](#岛屿的最大面积)|
|||||[820、单词的压缩编码(medium)](#单词的压缩编码)|
|[836、矩形重叠(easy)](#矩形重叠)|||||
|[876、链表的中间结点(easy)](#链表的中间结点)||
||[887、鸡蛋掉落(hard)](#鸡蛋掉落)||||
||[892、三维形体的表面积(easy)](#三维形体的表面积)||||
||[912、排序数组(medium)](#排序数组)||[914、卡牌分组(easy)](#卡牌分组)||
|||||[945、使数组唯一的最小增量(medium)](#使数组唯一的最小增量)|
|||[983、最低票价(medium)](#最低票价)|||
||||[994、腐烂的橘子(easy)](#腐烂的橘子)||
||||[999、车的可用捕获量(easy)](#车的可用捕获量)||
|||[1013、将数组分成和相等的三个部分(easy)](#将数组分成和相等的三个部分)|||
|[1071、字符串的最大公因子(easy)](#字符串的最大公因子)||
|||||[1095、山脉数组中查找目标值(hard)](#山脉数组中查找目标值)|
|||[1103、分糖果 II(easy)](#分糖果2)|||
|[1111、有效括号的嵌套深度(medium)](#有效括号的嵌套深度)||
|||||[1160、拼写单词(easy)](#拼写单词)|
||[1162、地图分析(medium)](#地图分析)||
|||[1248、统计「优美子数组」(medium)](#统计优美子数组)|||
||||[1394、字符串压缩(easy)](#字符串压缩)||
|||[1418、旋转矩阵(medium)](#旋转矩阵)|||
|[1476、交点(hard)](#交点)||
|[1481、硬币(medium)](#硬币)||||
|[1496、按摩师(easy)](#按摩师)||
|[1531、机器人的运动范围(medium)](#机器人的运动范围)||
|||[1538、最小的k个数(easy)](#最小的k个数)|||
||||[1579、圆圈中最后剩下的数字(easy)](#圆圈中最后剩下的数字)||
|[1591、数组中的逆序对(hard)](#数组中的逆序对)||||
|||[1608、数组中数字出现的次数(medium)](#数组中数字出现的次数)|||

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
## [3、无重复字符的最长子串(medium)](#back)
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
## [11、盛最多水的容器(medium)](#back)
```cpp
给定n个非负整数a1，a2，...，an，每个数代表坐标中的一个点(i, ai)。
在坐标内画n条垂直线，垂直线i的两个端点分别为(i, ai)和(i, 0)。
找出其中的两条线，使得它们与x轴共同构成的容器可以容纳最多的水。
说明：你不能倾斜容器，且 n 的值至少为 2。

输入：[1,8,6,2,5,4,8,3,7]
输出：49

class Solution {
public:
    int maxArea(vector<int>& height) {
        int i = 0, j = height.size() - 1;
        int area = (j - i) * min(height[i], height[j]);
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
## [21、合并两个有序链表(easy)](#back)
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
## [22、括号生成(medium)](#back)
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
## [23、合并K个排序链表(hard)](#back)
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
## [33、搜索旋转排序数组(medium)](#back)
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
## [42、接雨水(hard)](#back)
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
## [46、全排列(medium)](#back)
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
## [55、跳跃游戏(medium)](#back)
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
## [56、合并区间(medium)](#back)
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
    	// 方法1
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
	
	// 方法2	
	// 排序 + 指针
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> res;
        for (int i = 0; i < intervals.size(); ) {
            int t = intervals[i][1];
            int j = i + 1;
            while (j < intervals.size() && intervals[j][0] <= t) {
                t = max(t, intervals[j][1]);
                j++;
            }
            res.push_back({intervals[i][0], t});
            i = j;
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
## [69、X的平方根(easy)](#back)
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
## [72、编辑距离(hard)](#back)
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
## [98、验证二叉搜索树(medium)](#back)
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
	// 方法1
        if (prices.size() == 0) return 0;
        int min_ = prices[0], max_ = prices[0], max_all = 0;
        for (int i = 1;i < prices.size(); i++) {
            if (prices[i] < min_) {  // 当出现最小时, min_ 和 max_ 重置
                min_ = prices[i];
                max_ = prices[i];
            }
            if (prices[i] > max_) {  // 当之后出现的值大于 max_ , 需要更新 max_
                max_ = prices[i];
            }
            max_all = max(max_ - min_, max_all);
        }
        return max_all;
		
	// 方法2
	// 最大收益 = 每个当前的卖出值 - 最小值
	// if(prices.size() == 0) return 0;
        // int min_ = prices[0], max_profit = 0;
        // for (int i = 1; i < prices.size(); i++) {
        //     max_profit = max(max_profit, prices[i] - min_);
        //     min_ = min(prices[i], min_);
        // }
        // return max_profit;
    }
};
```

<span id="买卖股票的最佳时机2"></span>
## [122、买卖股票的最佳时机 II(easy)](#back)
```cpp
给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

输入: [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 

输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 
注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。

输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.size() == 0) return 0;
        int buy_min = prices[0], sell_max = prices[0], profit = 0;
        for (int i = 1; i < prices.size(); i++) {
            if (buy_min > prices[i]) {
                buy_min = prices[i];
                sell_max = prices[i];
            }
            if (sell_max < prices[i]) {
                sell_max = prices[i];
                profit += (sell_max - buy_min);
                buy_min = prices[i];
            }
        }
        return profit;
    }
};
```

<span id="买卖股票的最佳时机3"></span>
## [123、买卖股票的最佳时机 III(hard)](#back)
```cpp
给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

输入: [3,3,5,0,0,3,1,4]
输出: 6
解释: 在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 
随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 

输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 
注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。   
因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。

输入: [7,6,4,3,1] 
输出: 0 
解释: 在这个情况下, 没有交易完成, 所以最大利润为 0。

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        // dp[i][k][0]: 从开始到 i 天，发生交易 K 次， 没有持有股票
        // dp[i][k][1]：从开始到 i 天，发生交易 K 次， 持有股票
        if(prices.size() <= 1) return 0;
        int n = prices.size(), maxK = 2;
        vector<vector<vector<int>>> dp(n, vector<vector<int>> (3, vector<int> (2, 0)));
        for (int i = 0; i < n ; i++) {
            for (int k = 1; k <= maxK; k++) {
                if (i == 0) {
                    dp[0][k][0] = 0;
                    dp[0][k][1] = -prices[0];
                    continue;
                }
                // 没有持有股票: 1、之前就没有2、之前有，卖掉了
                dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i]);
                // 持有股票：1、之前就有2、之前没有，新买的，发生交易
                dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i]);
            }
        }
        return dp[n - 1][2][0];
    }
};
```

<span id="二叉树中的最大路径和"></span>
## [124、二叉树中的最大路径和(hard)](#back)
```cpp
给定一个非空二叉树，返回其最大路径和。
本题中，路径被定义为一条从树中任意节点出发，达到任意节点的序列。该路径至少包含一个节点，且不一定经过根节点。

输入: [1,2,3]
       1
      / \
     2   3
输出: 6

输入: [-10,9,20,null,null,15,7]
   -10
   / \
  9  20
    /  \
   15   7
输出: 42

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
    int maxPathSum(TreeNode* root, int &val) {
        if(root == NULL) return 0;
        int left = maxPathSum(root -> left, val);
        int right = maxPathSum(root -> right, val);
        int lm = root -> val + max(0, left) + max(0, right);
        int ret = root -> val + max(0, max(left, right));
        val = max(val, max(lm, ret));
        return ret;
    }

    int maxPathSum(TreeNode* root) {
        // 参考：https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/solution/er-cha-shu-zhong-de-zui-da-lu-jing-he-by-ikaruga/
        int val = INT_MIN;
        maxPathSum(root, val);
        return val;
    }
};
```

<span id="验证字符串"></span>
## [125、验证字符串(easy)](#back)
```cpp
给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。
说明：本题中，我们将空字符串定义为有效的回文串。

输入: "A man, a plan, a canal: Panama"
输出: true

输入: "race a car"
输出: false

class Solution {
public:
    bool isPalindrome(string s) {
        if (s.size() == 0) return true;
        int left = 0, right = s.size() - 1;
        while (left < right) {
            if (!isalpha(s[left]) && !isdigit(s[left])) {
                left++;
                continue;
            }
            if (!isalpha(s[right]) && !isdigit(s[right])) {
                right--;
                continue;
            }
            if (tolower(s[left]) == tolower(s[right])) {
                left++;
                right--;
            }
            else return false;
        }
        return true;
    }
};
```

<span id="单词接龙"></span>
## [127、单词接龙(medium)](#back)
```cpp
给定两个单词（beginWord 和 endWord）和一个字典，找到从 beginWord 到 endWord 的最短转换序列的长度。
转换需遵循如下规则：
*每次转换只能改变一个字母。
*转换过程中的中间单词必须是字典中的单词。
说明:
*如果不存在这样的转换序列，返回 0。
*所有单词具有相同的长度。
*所有单词只由小写字母组成。
*字典中不存在重复的单词。
*你可以假设 beginWord 和 endWord 是非空的，且二者不相同。

输入:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]
输出: 5
解释: 一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog",
     返回它的长度 5。

输入:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]
输出: 0
解释: endWord "cog" 不在字典中，所以无法进行转换。

class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        // hit,1 -> hot,2 -> dot,3 -> log,4 -> log,5
        //                -> lot,3
        unordered_set<string> s;
        for (auto &i : wordList) s.insert(i);
        queue<pair<string, int>> q;
        q.push({beginWord, 1});  // 加入beginword
        string tmp;  // 每个节点的字符
        int step;  // 到达该节点的step
        while (!q.empty()) {
            if(q.front().first == endWord)
                return q.front().second;
            tmp = q.front().first;
            step = q.front().second;
            q.pop();
            // 寻找写一个单词
            char ch;
            for (int i = 0; i < tmp.size(); i++) {
                ch = tmp[i];
                for (char c = 'a'; c <= 'z'; c++) {
                    if (ch == c) continue;
                    tmp[i] = c;
                    if (s.find(tmp) != s.end()) {  // 如果在 s 中找的到
                        q.push({tmp, step + 1});
                        s.erase(tmp);  // 删除该节点
                    }
                    tmp[i] = ch;  // 复原
                }
            }
        }
        return 0;
    }
};
```

<span id="最长连续序列"></span>
## [128、最长连续序列(hard)](#back)
```cpp
给定一个未排序的整数数组，找出最长连续序列的长度。
要求算法的时间复杂度为 O(n)。

输入: [100, 4, 200, 1, 3, 2]
输出: 4
解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。

class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        // unorder_set 底层是hash_set，查找删除复杂度为 O(1)； 遍历时间复杂度 O(n), 空间复杂度为 O(1);
        if (nums.size() <= 1) return nums.size();
        unordered_set<int> s(nums.begin(), nums.end());  // 去重
        int res = 1;
        for (auto num : s) {
            if (s.count(num - 1) != 0) continue;  // num 为于子序列中部
            int len = 1;  // 当 num 位于子序列最左边，往下进行，统计序列长度
            while (s.count(num + 1) != 0) {  // num 位于子序列的最左边，统计连续的情况
                len++;
                num++;
            }
            res = max(res, len);
        }
        return res;
    }
};
```

<span id="求根到叶子节点数字之和"></span>
## [129、求根到叶子节点数字之和(medium)](#back)
```cpp
给定一个二叉树，它的每个结点都存放一个 0-9 的数字，每条从根到叶子节点的路径都代表一个数字。
例如，从根到叶子节点路径 1->2->3 代表数字 123。
计算从根到叶子节点生成的所有数字之和。
说明: 叶子节点是指没有子节点的节点。

输入: [1,2,3]
    1
   / \
  2   3
输出: 25
解释:
从根到叶子节点路径 1->2 代表数字 12.
从根到叶子节点路径 1->3 代表数字 13.
因此，数字总和 = 12 + 13 = 25.

输入: [4,9,0,5,1]
    4
   / \
  9   0
 / \
5   1
输出: 1026
解释:
从根到叶子节点路径 4->9->5 代表数字 495.
从根到叶子节点路径 4->9->1 代表数字 491.
从根到叶子节点路径 4->0 代表数字 40.
因此，数字总和 = 495 + 491 + 40 = 1026.

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
    void sumNumberDFS(TreeNode* root, int path, int& all_path) {
        if (!root) return ;
        path = path * 10 + root -> val;
        if (root -> left == NULL && root -> right == NULL) {
            all_path += path;
            return ;
        }
        sumNumberDFS(root -> left, path, all_path);
        sumNumberDFS(root -> right, path, all_path);
    }

    int sumNumbers(TreeNode* root) {
        int path = 0, all_path = 0;
        sumNumberDFS(root, path, all_path);
        return all_path;
    } 
};
```

<span id="被围绕的区域"></span>
## [130、被围绕的区域(medium)](#back)
```cpp
给定一个二维的矩阵，包含 'X' 和 'O'（字母 O）。
找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。

X X X X
X O O X
X X O X
X O X X
运行你的函数后，矩阵变为：
X X X X
X X X X
X X X X
X O X X
解释:
被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。

class Solution {
public:
    void solve(vector<vector<char>>& board) {
        int rows = board.size();
        if (rows == 0) return ;
        int cols = board[0].size();
        for (int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                bool edge = i == 0 || i == rows - 1 || j == 0 || j == cols - 1;
                if (edge && board[i][j] == 'O') {
                    dfs(board, i , j);
                }
            }
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (board[i][j] == '#') {
                    board[i][j] = 'O';
                }
                else board[i][j] = 'X';
            }
        }
    }

    void dfs(vector<vector<char>>& board, int i, int j) {
        if (i < 0 || i >= board.size() || j < 0 || j >= board[0].size() || board[i][j] == 'X' || board[i][j] == '#')
            return ;
        board[i][j] = '#';
        dfs(board, i - 1, j);
        dfs(board, i + 1, j);
        dfs(board, i, j - 1);
        dfs(board, i, j + 1);
    }
};
```

<span id="分割回文串"></span>
## [131、分割回文串(medium)](#back)
<div align=center><img src="https://github.com/FangChao1086/LeetCode_Solutions/blob/master/依赖文件/LeetCode131_分割字符串.png"></div>

```cpp
给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。
返回 s 所有可能的分割方案。

输入: "aab"
输出:
[
  ["aa","b"],
  ["a","a","b"]
]

class Solution {
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string>> res;
        vector<string> cur;
        dfs(s, cur, res);
        return res;
    }

    bool isPalindrome(string s) {
        return s == string(s.rbegin(), s.rend());
    }

    void dfs(string s, vector<string>& cur, vector<vector<string>>& res) {
        if (s == "") {
            res.push_back(cur);
            return ;
        }
        for (int i = 1; i <= s.size(); i++) {
            string tmp = s.substr(0, i);
            if (isPalindrome(tmp)) {
                cur.push_back(tmp);
                dfs(s.substr(i, s.size() - 1), cur, res);
                cur.pop_back();
            }
        }
    }
};
```

<span id="分割回文串2"></span>
## [132、分割回文串 II(hard)](#back)
```cpp
给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。
返回符合要求的最少分割次数。

输入: "aab"
输出: 1
解释: 进行一次分割就可将 s 分割成 ["aa","b"] 这样两个回文子串。

class Solution {
public:
    int minCut(string s) {
        // 动态规划
        // dp[i]: s[0 : i] 是回文需要的分割次数
        // dp[i] = min(dp[i], dp[j] + 1) if dp[j + 1 : i] 是回文
        int len = s.size();
        int dp[len];
        // 初始化
        for (int i = 0; i < len; i++)
            dp[i] = i;
        // 记录子串 s[a : b]是否是回文，一开始初始化为false
        vector<vector<bool>> checkPalindrome(len, vector<bool> (len, false));
        for (int right = 0; right < len; right++) {
            for (int left = 0; left <= right; left++) {
                if (s[left] == s[right] && (right - left <= 2 || checkPalindrome[left + 1][right - 1]))
                    checkPalindrome[left][right] = true;
            }
        }
        // 状态转移
        for (int i = 0; i < len; i++) {
            if (checkPalindrome[0][i]) {
                dp[i] = 0;
                continue;
            }
            for (int j = 0; j < i; j++) {
                if(checkPalindrome[j + 1][i]) {
                    dp[i] = min(dp[i], dp[j] + 1);
                }
            }
        }
        return dp[len - 1];
    }
};
```

<span id="克隆图"></span>
## [133、克隆图(medium)](#back)
<div align=center><img src="https://github.com/FangChao1086/LeetCode_Solutions/blob/master/依赖文件/LeetCode133_克隆图.png"></div>

```cpp
给你无向 连通 图中一个节点的引用，请你返回该图的 深拷贝（克隆）。
图中的每个节点都包含它的值 val（int） 和其邻居的列表（list[Node]）。
class Node {
    public int val;
    public List<Node> neighbors;
}
 
测试用例格式：
简单起见，每个节点的值都和它的索引相同。
例如，第一个节点值为 1，第二个节点值为 2，以此类推。
该图在测试用例中使用邻接列表表示。
邻接列表是用于表示有限图的无序列表的集合。每个列表都描述了图中节点的邻居集。
给定节点将始终是图中的第一个节点（值为 1）。你必须将给定节点的拷贝作为对克隆图的引用返回。

输入：adjList = [[2,4],[1,3],[2,4],[1,3]]
输出：[[2,4],[1,3],[2,4],[1,3]]
解释：
图中有 4 个节点。
节点 1 的值是 1，它有两个邻居：节点 2 和 4 。
节点 2 的值是 2，它有两个邻居：节点 1 和 3 。
节点 3 的值是 3，它有两个邻居：节点 2 和 4 。
节点 4 的值是 4，它有两个邻居：节点 1 和 3 。

输入：adjList = [[]]
输出：[[]]
解释：输入包含一个空列表。该图仅仅只有一个值为 1 的节点，它没有任何邻居。

输入：adjList = []
输出：[]
解释：这个图是空的，它不含任何节点。

输入：adjList = [[2],[1]]
输出：[[2],[1]]

提示：
节点数介于 1 到 100 之间。
每个节点值都是唯一的。
无向图是一个简单图，这意味着图中没有重复的边，也没有自环。
由于图是无向的，如果节点 p 是节点 q 的邻居，那么节点 q 也必须是节点 p 的邻居。
图是连通图，你可以从给定节点访问到所有节点。

/*
// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> neighbors;
    
    Node() {
        val = 0;
        neighbors = vector<Node*>();
    }
    
    Node(int _val) {
        val = _val;
        neighbors = vector<Node*>();
    }
    
    Node(int _val, vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};
*/
class Solution {
public:
    Node* cloneGraph(Node* node) {
        // BFS + map
        // 1、根据原来节点的节点数，和节点值创建新的节点
        // 2、根据原来节点的对应关系，连接新的对应节点
        if (!node) return NULL;
        queue<Node*> que;
        que.push(node);
        map<Node*, Node*> mp;
        while (!que.empty()) {
            Node* temp = que.front();
            que.pop();
            // 新节点创建
            Node* p = new Node(temp -> val, {});
            mp.insert({temp, p});  // 将新节点 p 与旧节点 temp 之间形成映射
            for (Node* neighborsNode : temp -> neighbors) {
                if (mp.find(neighborsNode) == mp.end()) {
                    que.push(neighborsNode);
                }
            }
        }
        // 遍历所有节点 完成边的连接
        map<Node*, Node*>::iterator iter;
        for (iter = mp.begin(); iter != mp.end(); iter++) {
            for (Node* neighborsNode : iter -> first ->neighbors){
                iter -> second ->neighbors.push_back(mp.find(neighborsNode) -> second);
            }
        }
        return mp.find(node) -> second;
    }
};
```

<span id="加油站"></span>
## [134、加油站(medium)](#back)
```cpp
在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。
你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。
如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。
说明: 
如果题目有解，该答案即为唯一答案。
输入数组均为非空数组，且长度相同。
输入数组中的元素均为非负数。

输入: 
gas  = [1,2,3,4,5]
cost = [3,4,5,1,2]
输出: 3
解释:
从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
因此，3 可为起始索引。

输入: 
gas  = [2,3,4]
cost = [3,4,3]
输出: -1
解释:
你不能从 0 号或 1 号加油站出发，因为没有足够的汽油可以让你行驶到下一个加油站。
我们从 2 号加油站出发，可以获得 4 升汽油。 此时油箱有 = 0 + 4 = 4 升汽油
开往 0 号加油站，此时油箱有 4 - 3 + 2 = 3 升汽油
开往 1 号加油站，此时油箱有 3 - 3 + 3 = 3 升汽油
你无法返回 2 号加油站，因为返程需要消耗 4 升汽油，但是你的油箱只有 3 升汽油。
因此，无论怎样，你都不可能绕环路行驶一周。

class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int len = gas.size();
        if (len == 0) return -1;
        int cur = 0, total = 0, start_position = 0;
        for (int i = 0; i < len; i++) {
            total += gas[i] - cost[i];
            cur += gas[i] - cost[i];
            if (cur < 0) {
                start_position = i + 1;
                cur = 0;
            }
        }
        return total >= 0 ? start_position : -1;
    }
};
```

<span id="分发糖果"></span>
## [135、分发糖果(hard)](#back)
```cpp
老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。
你需要按照以下要求，帮助老师给这些孩子分发糖果：
每个孩子至少分配到 1 个糖果。
相邻的孩子中，评分高的孩子必须获得更多的糖果。
那么这样下来，老师至少需要准备多少颗糖果呢？

输入: [1,0,2]
输出: 5
解释: 你可以分别给这三个孩子分发 2、1、2 颗糖果。

输入: [1,2,2]
输出: 4
解释: 你可以分别给这三个孩子分发 1、2、1 颗糖果。
     第三个孩子只得到 1 颗糖果，这已满足上述两个条件。

class Solution {
public:
    int candy(vector<int>& ratings) {
        // dp[i] : 第 i 个孩子应该获得的糖果数；
        // 前后各扫描一次
        int len = ratings.size();
        if (len <= 1) return ratings.size();
        vector<int> dp(len, 1);
        for (int i = 1; i < len; i++) {
            if (ratings[i] > ratings[i - 1])
                dp[i] = dp[i - 1] + 1;
        }
        for (int i = len - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1])
                dp[i] = max(dp[i], dp[i + 1] + 1);
        }
        int all_ = 0;
        for (int i = 0; i < len; i++) {
            all_ += dp[i];
        }
        return all_;
    }
};
```

<span id="只出现一次的数字"></span>
## [136、只出现一次的数字(easy)](#back)
```cpp
给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
说明：
你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

输入: [2,2,1]
输出: 1

输入: [4,1,2,1,2]
输出: 4

class Solution {
public:
    int singleNumber(vector<int>& nums) {
        // 异或
        // a 异或 0 = a
        // a 异或 a = 0  
        if (nums.size() == 0) return 0;
        int result = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            result ^= nums[i];
        }
        return result;
    }
};
```

<span id="只出现一次的数字2"></span>
## [137、只出现一次的数字 II](#back)
```cpp
给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。
说明：你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

输入: [2,2,3,2]
输出: 3

输入: [0,1,0,1,0,1,99]
输出: 99

class Solution {
public:
    int singleNumber(vector<int>& nums) {
        // 各个数字和的二进制每位中1的个数是 3 的倍数或者0；
        int res = 0;
        for (int i = 0; i < 32; i++) {
            int tmp = 0;
            for (int j = 0; j < nums.size(); j++) {
                tmp += (nums[j] >> i) & 1;
            }
            res ^= (tmp % 3) << i;
        }
        return res;
    }
};
```

<span id="复制带随机指针的链表"></span>
## [138、复制带随机指针的链表(medium)](#back)
```cpp
给定一个链表，每个节点包含一个额外增加的随机指针，该指针可以指向链表中的任何节点或空节点。
要求返回这个链表的 深拷贝。 
我们用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：
val：一个表示 Node.val 的整数。
random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。

输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]

输入：head = [[1,1],[2,1]]
输出：[[1,1],[2,1]]

输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]

输入：head = []
输出：[]
解释：给定的链表为空（空指针），因此返回 null。

/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
*/
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (head == NULL) return head;
        Node* p = head;
        while (p != nullptr) {
            Node* new_ = new Node(p -> val);
            new_ -> next = p -> next;
            p -> next = new_;
            p = new_ -> next;
        }
        p = head;
        while (p != nullptr) {
            Node* copy_next = p -> next;
            if (p -> random) {
                copy_next -> random = p -> random -> next;
            }
            p = copy_next -> next;
        }
        p = head;
        Node* copy_ = p -> next;
        while (p -> next != nullptr) {
            Node* tmp = p -> next;
            p -> next = tmp -> next;
            p = tmp;
        }
        return copy_;
    }
};
```

<span id="单词拆分"></span>
## [139、单词拆分](#back)
```cpp
给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。
说明：
拆分时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。

输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。

输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。

输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false

class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        // 动态规划
        vector<bool> dp(s.size() + 1, false);
        dp[0] = true;
        for (int i = 0; i <= s.size(); i++) {
            for (auto word : wordDict) {
                int word_size = word.size();
                if (i - word_size >= 0) {
                    // s 中 i - word_size 开始，长度为 word_size 的子串与 word 相比较，相同则返回 0
                    int cur = s.compare(i - word_size, word_size, word);
                    if (cur == 0 && dp[i - word_size]) 
                        dp[i] = true;
                }
            }
        }
        return dp[s.size()];
    }
};
```

<span id="环形链表"></span>
## [141、环形链表(easy)](#back)
```cpp
给定一个链表，判断链表中是否有环。
为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。
如果 pos 是 -1，则在该链表中没有环。

3 -> 2 -> 0 -> -4
     ^          | 
     | -------- V
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。

1 - 2
^   |
|---V
输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。

输入：head = [1], pos = -1
输出：false
解释：链表中没有环。

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
    bool hasCycle(ListNode *head) {
    	// 快慢指针
        if (head == NULL || head -> next == NULL) return false;
        ListNode* p1 = head;
        ListNode* p2 = head;
        while (p2 != NULL && p2 -> next != NULL) {
            p1 = p1 -> next;
            p2 = p2 -> next -> next;
            if (p1 == p2) return true;
        }
        return false;
    }
};
```

<span id="环形链表2"></span>
## [142、环形链表 II(medium)](#back)
```cpp
给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 
如果 pos 是 -1，则在该链表中没有环。
说明：不允许修改给定的链表。

3 -> 2 -> 0 -> -4
     ^          | 
     | -------- V
输入：head = [3,2,0,-4], pos = 1
输出：tail connects to node index 1
解释：链表中有一个环，其尾部连接到第二个节点。


1 - 2
^   |
|---V
输入：head = [1,2], pos = 0
输出：tail connects to node index 0
解释：链表中有一个环，其尾部连接到第一个节点。

输入：head = [1], pos = -1
输出：no cycle
解释：链表中没有环。

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
    ListNode *detectCycle(ListNode *head) {
        if(head == NULL || head -> next == NULL) return NULL;
        ListNode* p1 = head;  // 慢指针
        ListNode* p2 = head;  // 快指针
        while (p2 != NULL && p2 -> next != NULL) {
            p1 = p1 -> next;
            p2 = p2 -> next -> next;
            if (p1 == p2) {
                p2 = head;
                while(p1 != p2) {
                    p1 = p1 -> next;
                    p2 = p2 -> next;
                } 
                return p1;
            }
        }
        return NULL;
    }
};
```

<span id="重排链表"></span>
## [143、重排链表(medium)](#back)
```cpp
给定一个单链表 L：L0→L1→…→Ln-1→Ln ，
将其重新排列后变为： L0→Ln→L1→Ln-1→L2→Ln-2→…
你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

给定链表 1->2->3->4, 重新排列为 1->4->2->3.

给定链表 1->2->3->4->5, 重新排列为 1->5->2->4->3.

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
    ListNode* reverse(ListNode* head) {
        ListNode* p_head = head;
        ListNode* p_pre = NULL;
        while (p_head) {
            ListNode* tmp = p_head -> next;
            p_head -> next = p_pre;
            p_pre = p_head;
            p_head = tmp;
        }
        return p_pre;
    }

    void reorderList(ListNode* head) {
        // 找中间节点
        // 中间节点反转
        // 合并
        if (head == NULL || head -> next == NULL) return ;
        ListNode* p_slow = head;
        ListNode* p_fast = head;
        while (p_fast != NULL &&  p_fast -> next != NULL) {
            p_slow = p_slow -> next;
            p_fast = p_fast -> next ->next;
        }
        ListNode* p_mid = p_slow -> next;  // 中间节点之后的节点
        p_slow -> next = NULL;  // 左边 head -...-p_slow - NULL;  p_mid(p_slow -> next) ...
        p_mid = reverse(p_mid);
        ListNode* left = head;
        while (left -> next != NULL && p_mid != NULL) {
            // 保存下一个节点
            ListNode* leftTemp = left -> next;
            ListNode* rightTemp = p_mid -> next;

            // 左 1->2->3 右 5->4
            // 左 1->5->2->3
            left -> next = p_mid;
            p_mid -> next = leftTemp;
            
            // 左 2->3 右 4
            left = leftTemp;
            p_mid = rightTemp;
        }
    }
};
```

<span id="二叉树的前序遍历"></span>
## [144、二叉树的前序遍历(medium)](#back)
```cpp
给定一个二叉树，返回它的 前序 遍历。

输入: [1,null,2,3]  
   1
    \
     2
    /
   3 
输出: [1,2,3]

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
    void recursive(TreeNode* root, vector<int>& res) {
        if (root == NULL) return ;
        res.push_back(root -> val);
        recursive(root -> left, res);
        recursive(root -> right, res);
    }

    void iteration(TreeNode* root, vector<int>& res) {
        stack<TreeNode*> stk;
        stk.push(root);
        TreeNode* cur = NULL;
        while (!stk.empty()) {
            cur = stk.top();
            stk.pop();
            res.push_back(cur -> val);
            if (cur -> right) stk.push(cur -> right);
            if (cur -> left) stk.push(cur -> left);
        }
    }

    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res;
        if (root == NULL) return res;
        // recursive(root, res);  // 递归
        iteration(root, res);  // 迭代
        return res;
    }
};
```

<span id="二叉树的后序遍历"></span>
## [145、二叉树的后序遍历(hard)](#back)
```cpp
给定一个二叉树，返回它的 后序 遍历。

输入: [1,null,2,3]  
   1
    \
     2
    /
   3 
输出: [3,2,1]

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
    void recursive(TreeNode* root, vector<int>& res) {
        if(root == NULL) return ;
        recursive(root -> left, res);
        recursive(root -> right, res);
        res.push_back(root -> val);
    }

    void iteration(TreeNode* root, vector<int>& res) {
        stack<TreeNode*> stk1, stk2;
        stk1.push(root);
        while (!stk1.empty()) {
            TreeNode* tmp = stk1.top();
            stk1.pop();
            stk2.push(tmp);
            if (tmp -> left) stk1.push(tmp -> left);
            if (tmp -> right) stk1.push(tmp -> right);
        }
        while(!stk2.empty()) {
            res.push_back(stk2.top() -> val);
            stk2.pop();
        }
    }

    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> res;
        if (root == NULL) return res;
        // recursive(root, res);  // 递归
        iteration(root, res);  // 迭代
        return res;
    }
};
```

<span id="LRU缓存机制"></span>
## [146、LRU缓存机制(medium)](#back)
```cpp
运用你所掌握的数据结构，设计和实现一个  LRU (最近最少使用) 缓存机制。它应该支持以下操作： 获取数据 get 和 写入数据 put 。
获取数据 get(key) - 如果密钥 (key) 存在于缓存中，则获取密钥的值（总是正数），否则返回 -1。
写入数据 put(key, value) - 如果密钥不存在，则写入其数据值。当缓存容量达到上限时，它应该在写入新数据之前删除最近最少使用的数据值，从而为新的数据值留出空间。
进阶:
你是否可以在 O(1) 时间复杂度内完成这两种操作？

LRUCache cache = new LRUCache( 2 /* 缓存容量 */ );
cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // 返回  1
cache.put(3, 3);    // 该操作会使得密钥 2 作废
cache.get(2);       // 返回 -1 (未找到)
cache.put(4, 4);    // 该操作会使得密钥 1 作废
cache.get(1);       // 返回 -1 (未找到)
cache.get(3);       // 返回  3
cache.get(4);       // 返回  4

class LRUCache {
private:
    int cap;
    // 双链表：存储 (key, value) 元组
    list<pair<int , int>> cache;
    // 哈希表：key 映射到 (key, value) 在 cache 中的位置
    unordered_map<int, list<pair<int, int>>::iterator> un_mp;
public:
    LRUCache(int capacity) {
        this -> cap = capacity;
    }
    
    int get(int key) {
        auto it = un_mp.find(key);
        if (it == un_mp.end()) return -1;  // 访问的 key 不存在
        // 将 (key, value) 放到队头
        pair<int, int> kv = *un_mp[key];
        cache.erase(un_mp[key]);
        cache.push_front(kv);
        // 更新 (key, value) 在cache中的位置
        un_mp[key] = cache.begin();
        return kv.second;
    }
    
    void put(int key, int value) {
        auto it = un_mp.find(key);
        if (it == un_mp.end()) {  // 如果 key 不存在
            if (cache.size() == cap) {  // cache 已满
                auto last_pair = cache.back();
                int last_key = last_pair.first;
                un_mp.erase(last_key);
                cache.pop_back();
            }
            // 没满
            cache.push_front(make_pair(key, value));
            un_mp[key] = cache.begin();
        }
        else {
            // 如果 key 存在，更改value并换到队头
            cache.erase(un_mp[key]);
            cache.push_front(make_pair(key, value));
            un_mp[key] = cache.begin();
        }
    }
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
```

<span id="对链表进行插入排序"></span>
## [147、对链表进行插入排序(medium)](#back)

<div align=center><img src="https://github.com/FangChao1086/LeetCode_Solutions/blob/master/依赖文件/LeetCode147_对链表进行插入排序.gif"></div>

```cpp
对链表进行插入排序。

插入排序的动画演示如上。从第一个元素开始，该链表可以被认为已经部分排序（用黑色表示）。
每次迭代时，从输入数据中移除一个元素（用红色表示），并原地将其插入到已排好序的链表中。
插入排序算法：
插入排序是迭代的，每次只移动一个元素，直到所有元素可以形成一个有序的输出列表。
每次迭代中，插入排序只从输入数据中移除一个待排序的元素，找到它在序列中适当的位置，并将其插入。
重复直到所有输入数据插入完为止。

输入: 4->2->1->3
输出: 1->2->3->4

输入: -1->5->3->4->0
输出: -1->0->3->4->5

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
    ListNode* insertionSortList(ListNode* head) {
        if (head == nullptr || head -> next == nullptr) return head;
        ListNode* pDummy = new ListNode(0);
        pDummy -> next = head;
        ListNode* p_pre = head;
        ListNode* node = head -> next;
        while (node) {
            if (node -> val < p_pre -> val) {
                ListNode* temp = pDummy;
                while (temp -> next ->val < node -> val) {
                    temp = temp -> next;
                }
                p_pre -> next = node -> next;
                node -> next = temp -> next;
                temp -> next = node;
                node = p_pre -> next;
            }
            else {
                p_pre = p_pre -> next;
                node = node -> next;
            }
        }
        return pDummy -> next;
    }
};
```

<span id="排序链表"></span>
## [148、排序链表(medium)](#back)
```cpp
在 O(nlogn) 时间复杂度和常数级空间复杂度下，对链表进行排序。

输入: 4->2->1->3
输出: 1->2->3->4

输入: -1->5->3->4->0
输出: -1->0->3->4->5

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
    ListNode* sortList(ListNode* head) {
        // 归并排序
        if (head == nullptr || head -> next == nullptr) return head;
        ListNode* p_fast = head -> next;  // 注意这里的 p_fast 不是 head, 而是 head -> next
        ListNode* p_slow = head;
        while (p_fast != nullptr && p_fast -> next != nullptr) {
            p_slow = p_slow -> next;
            p_fast = p_fast -> next;
        }
        ListNode* tmp = p_slow -> next;
        p_slow -> next = nullptr;
        ListNode* left = sortList(head);  // 左边归并后正序链表
        ListNode* right = sortList(tmp);  // 右边归并后正序链表
        ListNode* p_dummy = new ListNode(0);
        ListNode* p_new = p_dummy;
        while(left != nullptr && right != nullptr) {
            if (left -> val < right -> val) {
                p_new -> next = left;
                left = left -> next;
            }
            else {
                p_new -> next = right;
                right = right -> next;
            }
            p_new = p_new -> next;
        }
        p_new -> next = left != nullptr ? left : right;
        return p_dummy -> next;
    }
};
```

<span id="直线上最多的点数"></span>
## [149、直线上最多的点数(hard)](#back)
```cpp
给定一个二维平面，平面上有 n 个点，求最多有多少个点在同一条直线上。

输入: [[1,1],[2,2],[3,3]]
输出: 3
解释:
^
|
|        o
|     o
|  o  
+------------->
0  1  2  3  4

输入: [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
输出: 4
解释:
^
|
|  o
|     o        o
|        o
|  o        o
+------------------->
0  1  2  3  4  5  6

class Solution {
public:
    int maxPoints(vector<vector<int>>& points) {
        // 暴力解
        int size_ = points.size();
        if (size_ <= 2) return size_;
        int num_oneline = 0, max_num = 0;
        for (int i = 0; i < size_; i++) {
            int x1 = points[i][0];
            int y1 = points[i][1];
            for (int j = i + 1; j < size_; j++) {
                if(points[j][0] == x1 && points[j][1] == y1) continue; 
                int x2 = points[j][0];
                int y2 = points[j][1];
                num_oneline = 0;
                for(int z = 0; z < size_; z++) {
                    if(points[z][0] == x1 && points[z][1] == y1)  num_oneline++;
                    else if(points[z][0] == x2 && points[z][1] == y2)  num_oneline++;
                    else if (long(points[z][1] - y1) * (x2 - x1) == long(points[z][0] - x1) * (y2 - y1)) num_oneline++;
                }
                max_num = max(max_num, num_oneline);
            }
        }
        if(max_num == 0) return points.size();  // 解决全是一个点的问题
        return max_num;
    }
};
```

<span id="逆波兰表达式求值"></span>
## [150、逆波兰表达式求值(medium)](#back)
```cpp
根据逆波兰表示法，求表达式的值。
有效的运算符包括 +, -, *, / 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。
说明：
整数除法只保留整数部分。
给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。

输入: ["2", "1", "+", "3", "*"]
输出: 9
解释: ((2 + 1) * 3) = 9

输入: ["4", "13", "5", "/", "+"]
输出: 6
解释: (4 + (13 / 5)) = 6

输入: ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
输出: 22
解释: 
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22

class Solution {
public:
    int evalRPN(vector<string>& tokens) {
        stack<int> stk;
        int len_t = tokens.size(); 
        for (int i = 0; i < len_t; i++) {
            if (tokens[i] == "+") {
                int a = stk.top();
                stk.pop();
                int b = stk.top();
                stk.pop();
                stk.push(b + a);
            }
            else if (tokens[i] == "-") {
                int a = stk.top();
                stk.pop();
                int b = stk.top();
                stk.pop();
                stk.push(b - a);
            }
            else if (tokens[i] == "*") {
                int a = stk.top();
                stk.pop();
                int b = stk.top();
                stk.pop();
                stk.push(b * a);
            }
            else if (tokens[i] == "/") {
                int a = stk.top();
                stk.pop();
                int b = stk.top();
                stk.pop();
                stk.push(b / a);
            }
            else {
                int tmp = atoi(tokens[i].c_str());
                stk.push(tmp);
            }
        }
        return stk.top();
    }
};
```

<span id="翻转字符串里的单词"></span>
## [151、翻转字符串里的单词(medium)](#back)
```cpp
给定一个字符串，逐个翻转字符串中的每个单词。

输入: "the sky is blue"
输出: "blue is sky the"

输入: "  hello world!  "
输出: "world! hello"
解释: 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。

输入: "a good   example"
输出: "example good a"
解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
 
说明：
无空格字符构成一个单词。
输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。

class Solution {
public:
    void reverse(string& s, int left, int right) {
        while(left < right) {
            swap(s[left], s[right]);
            left++;
            right--;
        }
    }

    string reverseWords(string s) {
        int l_= 0, r_ = s.size() - 1;
        while(l_< s.size() && s[l_] == ' ') l_++;
        while(r_ >= 0 && s[r_] == ' ') r_--;
        if (l_ > r_) return "";
        s = s.substr(l_, r_ - l_ + 1);  // 去除首尾空格
        reverse(s, 0, s.size() - 1);
        s += ' ';
        int index = 0;
        string s_new;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == ' ') {
                reverse(s, index, i - 1);
                s_new += s.substr(index, i - index) + ' ';
                while(s[i] == ' '){  // 当中间分隔出现多个空格，反转后只需要一个空格
                    i++;
                }
                index = i;
            }
        }
        return s_new.substr(0, s_new.size() - 1);
    }
};
```

<span id="乘积最大子序列"></span>
## [152、乘积最大子序列(medium)](#back)
```cpp
给定一个整数数组 nums ，找出一个序列中乘积最大的连续子序列（该序列至少包含一个数）。

输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。

输入: [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。

class Solution {
public:
    int maxProduct(vector<int>& nums) {
        // 动态规划
        // i_max : 当前位置乘积的最大值
	// i_min : 为解决数字中含有的负数
        int max_ = INT_MIN, i_max = 1, i_min = 1;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] < 0) {
                int tmp = i_min;
                i_min = i_max;
                i_max = tmp;
            }
            i_max = max(nums[i] * i_max, nums[i]);
            i_min = min(nums[i] * i_min, nums[i]);
            max_ = max(max_, i_max);
        }
        return max_;
    }
};
```

<span id="寻找旋转排序数组中的最小值"></span>
## [153、寻找旋转排序数组中的最小值(medium)](#back)
```cpp
假设按照升序排序的数组在预先未知的某个点上进行了旋转。
( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
请找出其中最小的元素。
你可以假设数组中不存在重复元素。

输入: [3,4,5,1,2]
输出: 1

输入: [4,5,6,7,0,1,2]
输出: 0

class Solution {
public:
    int findMin(vector<int>& nums) {
        // 二分查找
        int len_ = nums.size();
        int left = 0, mid = 0, right = len_ - 1;
        while (left < right) {
            mid = (left + right) >> 1;
            if (nums[mid] > nums[right]) left = mid + 1;
            else right = mid;
        }
        return nums[left];
    }
};
```

<span id="寻找旋转排序数组中的最小值2"></span>
## [154、寻找旋转排序数组中的最小值 II(hard)](#back)
```cpp
假设按照升序排序的数组在预先未知的某个点上进行了旋转。
( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
请找出其中最小的元素。
注意数组中可能存在重复的元素。

输入: [1,3,5]
输出: 1

输入: [2,2,2,0,1]
输出: 0

class Solution {
public:
    int findMin(vector<int>& nums) {
        // 二分查找
        int len_ = nums.size();
        int left = 0, right = len_ - 1;
        while(left < right) {
            if (nums[left] < nums[right]) return nums[left];  // 优化，可要可不要 
            int mid = (left + right) >> 1;
            if (nums[mid] > nums[right]) left = mid + 1;
            else if (nums[mid] < nums[right]) right = mid;
            else {
                if (nums[mid] == nums[left]) right--;  // 可能在左或者右
                else right = mid;  // 在左边
            }
        }
        return nums[left];
    }
};
```

<span id="最小栈"></span>
## [155、最小栈(easy)](#back)
```cpp
设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。
push(x) -- 将元素 x 推入栈中。
pop() -- 删除栈顶的元素。
top() -- 获取栈顶元素。
getMin() -- 检索栈中的最小元素。

MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.

class MinStack {
public:
    /** initialize your data structure here. */
    MinStack() {

    }
    
    void push(int x) {
        stk.push(x);
        if(m_stk.empty() || m_stk.top() > x) {
            m_stk.push(x);
        }
        else m_stk.push(m_stk.top());

    }
    
    void pop() {
        stk.pop();
        m_stk.pop();
    }
    
    int top() {
        return stk.top();

    }
    
    int getMin() {
        return m_stk.top();
    }
private:
    stack<int> stk;
    stack<int> m_stk;
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */
```

<span id="相交链表"></span>
## [160、相交链表(easy)](#back)
```cpp
编写一个程序，找到两个单链表相交的起始节点。

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
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        // 双指针
        ListNode* p_headA = headA;
        ListNode* p_headB = headB;
        while (p_headA != p_headB) {
            p_headA = p_headA == nullptr ? headB : p_headA -> next;
            p_headB = p_headB == nullptr ? headA : p_headB -> next;
        } 
        return p_headA;
    }
};
```

<span id="寻找峰值"></span>
## [162、寻找峰值(medium)](#back)
```cpp
峰值元素是指其值大于左右相邻值的元素。
给定一个输入数组 nums，其中 nums[i] ≠ nums[i+1]，找到峰值元素并返回其索引。
数组可能包含多个峰值，在这种情况下，返回任何一个峰值所在位置即可。
你可以假设 nums[-1] = nums[n] = -∞。

输入: nums = [1,2,3,1]
输出: 2
解释: 3 是峰值元素，你的函数应该返回其索引 2。

输入: nums = [1,2,1,3,5,6,4]
输出: 1 或 5 
解释: 你的函数可以返回索引 1，其峰值元素为 2；
     或者返回索引 5， 其峰值元素为 6。
说明:
你的解法应该是 O(logN) 时间复杂度的。

class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        // 二分查找
        int len_ = nums.size();
        int left = 0, right = len_ - 1;
        while (left < right) {
            int mid = (left + right) >> 1;
            if (nums[mid] > nums[mid + 1]) right = mid;
            else left = mid + 1;
        }
        return left;
    }
};
```

<span id="多数元素"></span>
## [169、多数元素(easy)](#back)
```cpp
给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
你可以假设数组是非空的，并且给定的数组总是存在多数元素。

输入: [3,2,3]
输出: 3

输入: [2,2,1,1,1,2,2]
输出: 2

class Solution {
public:
    int majorityElement(vector<int>& nums) {
    	// 排序后最中间的数即是所求值
        sort(nums.begin(), nums.end());
        return nums[nums.size() / 2];
    }
};
```

<span id="二叉树的右视图"></span>
## [199、二叉树的右视图(medium)](#back)
```cpp
给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

输入: [1,2,3,null,5,null,4]
输出: [1, 3, 4]
解释:
   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---

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
    vector<int> rightSideView(TreeNode* root) {
        vector<int> res;
        if(root == nullptr) return res;
        queue<TreeNode*> que;
        que.push(root);
        while (!que.empty()) {
            int size_ = que.size();
            for (int i = 0; i < size_; i++) {
                TreeNode* tmp = que.front();
                que.pop();
                if (i == size_ - 1) {
                    res.push_back(tmp -> val);
                }
                if (tmp -> left) {
                    que.push(tmp -> left);
                }
                if (tmp -> right) {
                    que.push(tmp -> right);
                }
            }
        }
        return res;
    }
};
```

<span id="岛屿数量"></span>
## [200、岛屿数量(medium)](#back)
```cpp
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
此外，你可以假设该网格的四条边均被水包围。

输入:
11110
11010
11000
00000
输出: 1

输入:
11000
11000
00100
00011
输出: 3
解释: 每座岛屿只能由水平和/或竖直方向上相邻的陆地连接而成。

class Solution {
public:
    int count_ = 0;
    int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    void find_islands(vector<vector<char>>& grid, vector<vector<int>>& flag, int i, int j) {
        if (i >= 0 && i < grid.size() && j >= 0 && j < grid[0].size() && grid[i][j] == '1' && flag[i][j] == 0) {
            flag[i][j] = 1;
            count_ += 1;
            for (int z = 0; z < 4; z++) {
                int new_i = i + dirs[z][0];
                int new_j = j + dirs[z][1];
                if (new_i >= 0 && new_i < grid.size() && new_j >= 0 && new_j < grid[0].size() && grid[new_i][new_j] == '1' && flag[new_i][new_j] == 0) {
                    find_islands(grid, flag, new_i, new_j);
                    count_--;
                }
            }
        }
    }

    int numIslands(vector<vector<char>>& grid) {
        if (grid.size() == 0) return 0;
        int rows = grid.size();
        int cols = grid[0].size();
        vector<vector<int>> flag(rows, vector<int> (cols, 0));
        for (int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if (grid[i][j] == '1' && flag[i][j] == 0) {
                    find_islands(grid, flag, i, j);
                }
            }
        }
        return count_;
    }
};
```

<span id="快乐数"></span>
## [202、快乐数(easy)](#back)
```cpp
编写一个算法来判断一个数 n 是不是快乐数。

「快乐数」定义为：
对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，
然后重复这个过程直到这个数变为 1，也可能是无限循环但始终变不到 1。
如果可以变为 1，那么这个数就是快乐数。
如果 n 是快乐数就返回 True ；不是，则返回 False 。

输入：19
输出：true
解释：
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1

class Solution {
public:
    int changeNum(int n) {
        int sum = 0;
        while (n) {
            sum += pow((n % 10), 2);
            n = n / 10;
        }
        return sum;
    }

    bool isHappy(int n) {
        // 双指针
        int p_slow = n;
        int p_fast = changeNum(n);
        while ((p_fast != 1) && (p_slow != p_fast)) {
            p_slow = changeNum(p_slow);
            p_fast = changeNum(changeNum(p_fast));
        }
        return p_fast == 1;
    }
};
```

<span id="反转链表"></span>
## [206、反转链表(easy)](#back)
```cpp
反转一个单链表。

输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
进阶:
你可以迭代或递归地反转链表。你能否用两种方法解决这道题？

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
    ListNode* recur_pre = nullptr;  // 递归使用
    ListNode* reverseList(ListNode* head) {
        // // 非递归（较多使用）
        // ListNode* cur = head;
        // ListNode* pre = nullptr;
        // while (cur != nullptr) {
        //     ListNode* tmp = cur -> next;
        //     cur -> next = pre;
        //     pre = cur;
        //     cur = tmp; 
        // }
        // return pre;

        // 递归
        if (head == nullptr) return recur_pre;
        ListNode* tmp = head -> next;
        head -> next = recur_pre;
        recur_pre = head;
        head = tmp;
        return reverseList(head);
    }
};
```

<span id="最大正方形"></span>
## [221、最大正方形(medium)](#back)
```cpp
在一个由 0 和 1 组成的二维矩阵内，找到只包含 1 的最大正方形，并返回其面积。

输入: 
1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
输出: 4

class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        // 动态规划
        // dp[i][j]：以 i, j 结尾的正方形边长最大值(位置 i, j)处的值为1
        // 当 matrix[i][j] 不是 1，dp[i][j] == 0
        if (matrix.size() == 0 || matrix[0].size() == 0) {
            return 0;
        }
        int rows = matrix.size(), cols = matrix[0].size();
        int max_ = 0;
        vector<vector<int>> dp(rows, vector<int> (cols, 0));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) dp[i][j] = 1;
                    else dp[i][j] = min(min(dp[i - 1][j - 1], dp[i][j - 1]), dp[i - 1][j]) + 1;
                    max_ = max(max_, dp[i][j]);
                }
            }
        }
        return pow(max_, 2);
    }
};
```

<span id="生命游戏"></span>
## [289、生命游戏(medium)](#back)
```cpp
根据 百度百科 ，生命游戏，简称为生命，是英国数学家约翰·何顿·康威在 1970 年发明的细胞自动机。
给定一个包含 m × n 个格子的面板，每一个格子都可以看成是一个细胞。
每个细胞都具有一个初始状态：1 即为活细胞（live），或 0 即为死细胞（dead）。

每个细胞与其八个相邻位置（水平，垂直，对角线）的细胞都遵循以下四条生存定律：
如果活细胞周围八个位置的活细胞数少于两个，则该位置活细胞死亡；
如果活细胞周围八个位置有两个或三个活细胞，则该位置活细胞仍然存活；
如果活细胞周围八个位置有超过三个活细胞，则该位置活细胞死亡；
如果死细胞周围正好有三个活细胞，则该位置死细胞复活；

根据当前状态，写一个函数来计算面板上所有细胞的下一个（一次更新后的）状态。
下一个状态是通过将上述规则同时应用于当前状态下的每个细胞所形成的，其中细胞的出生和死亡是同时发生的。

输入： 
[
  [0,1,0],
  [0,0,1],
  [1,1,1],
  [0,0,0]
]
输出：
[
  [0,0,0],
  [1,0,1],
  [0,1,1],
  [0,1,0]
]

class Solution {
public:
    void gameOfLife(vector<vector<int>>& board) {
        int rows = board.size();
        int cols = board[0].size();
        int dir[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int count_ = 0;
                for (int z = 0; z < 8; z++) {
                    int new_x = i + dir[z][0];
                    int new_y = j + dir[z][1];
                    if (new_x >= 0 && new_x < rows && new_y >= 0 && new_y < cols && abs(board[new_x][new_y]) == 1) {
                        count_++;
                    }
                }
                if (board[i][j] == 1 && (count_ < 2 || count_ >3)) {
                    board[i][j] = -1;  // 活细胞变成死细胞
                }
                if (board[i][j] == 0 && (count_ == 3)) {
                    board[i][j] = 2;  // 死细胞变成活细胞
                }
            }
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (board[i][j] <= 0) {
                    board[i][j] = 0;
                }
                else board[i][j] = 1;
            }
        }
    }
};
```

<span id="最长上升子序列"></span>
## [300、最长上升子序列(medium)](#back)
```cpp
给定一个无序的整数数组，找到其中最长上升子序列的长度。

输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
说明:
可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。
你算法的时间复杂度应该为 O(n2) 。
进阶: 你能将算法的时间复杂度降低到 O(n log n) 吗?

class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        // 动态规划
        // dp[i] = 以 i 位置中的数结尾时上升子序列的长度
        if(nums.size() <= 1) return nums.size();
        vector<int> dp(nums.size(), 1);
        for (int i = 1; i < nums.size(); i++) {
            int max_ = dp[i];
            for (int j = 0; j < i; j++) {
                if(nums[j] < nums[i] && max_ <= dp[j]) {
                    max_ = dp[j];
                    dp[i] = dp[j] + 1;
                }
            }
        }
        int dp_len = dp[0];
        for(auto a : dp) {
            dp_len = max(a, dp_len);
        }
        return dp_len;
    }
};
```

<span id="零钱兑换"></span>
## [322、零钱兑换](#back)
```cpp
给定不同面额的硬币 coins 和一个总金额 amount。
编写一个函数来计算可以凑成总金额所需的最少的硬币个数。
如果没有任何一种硬币组合能组成总金额，返回 -1。

输入: coins = [1, 2, 5], amount = 11
输出: 3 
解释: 11 = 5 + 5 + 1

输入: coins = [2], amount = 3
输出: -1

class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        // dp[i] : 组成金额 i 所需要的最少硬币数目
        // dp[i] = min ( dp[i - coins[j]] ) + 1; coins[j] 为硬币数组中的单个硬币元素
        int max_ = amount + 1;
        vector<int> dp(amount + 1,  max_); 
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.size(); j++) {
                if (coins[j] <= i)
                    dp[i] = min(dp[i], dp[i - coins[j]] + 1);
            }
        }
        return dp[amount] == max_ ? -1 : dp[amount];
    }
};
```

<span id="设计推特"></span>
## [355、设计推特(medium)](#back)
```cpp
设计一个简化版的推特(Twitter)，可以让用户实现发送推文，
关注/取消关注其他用户，能够看见关注人（包括自己）的最近十条推文。
你的设计需要支持以下的几个功能：
postTweet(userId, tweetId): 创建一条新的推文
getNewsFeed(userId): 检索最近的十条推文。
每个推文都必须是由此用户关注的人或者是用户自己发出的。
推文必须按照时间顺序由最近的开始排序。
follow(followerId, followeeId): 关注一个用户
unfollow(followerId, followeeId): 取消关注一个用户

Twitter twitter = new Twitter();

// 用户1发送了一条新推文 (用户id = 1, 推文id = 5).
twitter.postTweet(1, 5);
// 用户1的获取推文应当返回一个列表，其中包含一个id为5的推文.
twitter.getNewsFeed(1);
// 用户1关注了用户2.
twitter.follow(1, 2);
// 用户2发送了一个新推文 (推文id = 6).
twitter.postTweet(2, 6);
// 用户1的获取推文应当返回一个列表，其中包含两个推文，id分别为 -> [6, 5].
// 推文id6应当在推文id5之前，因为它是在5之后发送的.
twitter.getNewsFeed(1);
// 用户1取消关注了用户2.
twitter.unfollow(1, 2);
// 用户1的获取推文应当返回一个列表，其中包含一个id为5的推文.
// 因为用户1已经不再关注用户2.
twitter.getNewsFeed(1);

class Twitter {
    // 哈希表 + 链表
    struct Node {
        // 存储关注人的 Id
        unordered_set<int> followee;
        // 用链表存储 tweetId
        list<int> tweet;
    };
    // getNewsFeed 检索的推文的上限以及 tweetId 的时间戳
    int recentMax, time;
    // tweetId 对应发送的时间
    unordered_map<int, int> tweetTime;
    // 每个用户存储的信息
    unordered_map<int, Node> user;

public:
    /** Initialize your data structure here. */
    Twitter() {
        time = 0;
        recentMax = 10;
        user.clear();
    }

    void init(int userId) {
        user[userId].followee.clear();
        user[userId].tweet.clear();
    }
    
    /** Compose a new tweet. */
    void postTweet(int userId, int tweetId) {
        if (user.find(userId) == user.end()) {
            init(userId);
        }
        // 达到限制，去除链表末尾元素
        if (user[userId].tweet.size() == recentMax) {
            user[userId].tweet.pop_back();
        }
        user[userId].tweet.push_front(tweetId);
        tweetTime[tweetId] = ++time;
    }
    
    /** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
    vector<int> getNewsFeed(int userId) {
        vector<int> ans;
        ans.clear();
        for (list<int>::iterator it = user[userId].tweet.begin(); it != user[userId].tweet.end();++it) {
            ans.emplace_back(*it);
        }
        for (auto followeeId: user[userId].followee) {
            if (followeeId == userId) continue; // 自己关注自己的情况
            vector<int> res;
            res.clear();
            list<int>::iterator it = user[followeeId].tweet.begin();
            int i = 0;
            while (i < ans.size() && it != user[followeeId].tweet.end()) {
                if (tweetTime[(*it)] > tweetTime[ans[i]]) {
                    res.emplace_back(*it);
                    ++it;
                }
                else {
                    res.emplace_back(ans[i]);
                    ++i;
                }
                if (res.size() == recentMax) break;
            }
            for (; i < ans.size() && res.size() < recentMax; ++i) res.emplace_back(ans[i]);
            for (; it != user[followeeId].tweet.end() && res.size() < recentMax; ++it) res.emplace_back(*it);
            ans.assign(res.begin(), res.end());
        }
        return ans;
    }
    
    /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
    void follow(int followerId, int followeeId) {
        if (user.find(followerId) == user.end()) {
            init(followerId);
        }
        if (user.find(followeeId) == user.end()) {
            init(followeeId);
        }
        user[followerId].followee.insert(followeeId);
    }
    
    /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
    void unfollow(int followerId, int followeeId) {
        user[followerId].followee.erase(followeeId);
    }
};

/**
 * Your Twitter object will be instantiated and called as such:
 * Twitter* obj = new Twitter();
 * obj->postTweet(userId,tweetId);
 * vector<int> param_2 = obj->getNewsFeed(userId);
 * obj->follow(followerId,followeeId);
 * obj->unfollow(followerId,followeeId);
 */
```

<span id="水壶问题"></span>
## [365、水壶问题(medium)](#back)
```cpp
有两个容量分别为 x升 和 y升 的水壶以及无限多的水。
请判断能否通过使用这两个水壶，从而可以得到恰好 z升 的水？
如果可以，最后请用以上水壶中的一或两个来盛放取得的 z升 水。
你允许：
装满任意一个水壶
清空任意一个水壶
从一个水壶向另外一个水壶倒水，直到装满或者倒空

输入: x = 3, y = 5, z = 4
输出: True

输入: x = 2, y = 6, z = 5
输出: False

class Solution {
public:
    // 最大公约数
    int gcd(int x, int y) {
        int z = y;
        while (x % y != 0) {
            z = x % y;
            x = y;
            y = z;
        }
        return z;
    }

    bool canMeasureWater(int x, int y, int z) {
        if (x + y < z) return false;
        if (x == 0 || y == 0) return z == 0 || x + y == z;
        return z % gcd(x, y) == 0;
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

<span id="最长回文串"></span>
## [409、最长回文串(easy)](#back)
```cpp
给定一个包含大写字母和小写字母的字符串，找到通过这些字母构造成的最长的回文串。
在构造过程中，请注意区分大小写。比如 "Aa" 不能当做一个回文字符串。
注意:
假设字符串的长度不会超过 1010。

输入:"abccccdd"
输出:7
解释:我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。

class Solution {
public:
    int longestPalindrome(string s) {
        unordered_map<char, int> mp_;
        for (int i = 0; i < s.size(); i++) {
            mp_[s[i]]++;
        }
        int all_ = 0;
        for (int i = 0; i < mp_.size(); i++) {
            if(mp_[i] / 2 != 0) {
                all_ += mp_[i] / 2 * 2;
            }
        }
        return all_ == s.size() ? all_ : (all_ + 1);
    }
};
```

<span id="两数相加2"></span>
## [445、两数相加 II](#back)
```cpp
给你两个 非空 链表来代表两个非负整数。
数字最高位位于链表开始位置。它们的每个节点只存储一位数字。
将这两数相加会返回一个新的链表。
你可以假设除了数字 0 之外，这两个数字都不会以零开头。
进阶：
如果输入链表不能修改该如何处理？换句话说，你不能对列表中的节点进行翻转。

输入：(7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 8 -> 0 -> 7

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
        stack<int> s1, s2;
        while (l1) {
            s1.push(l1 -> val);
            l1 = l1 -> next;
        }
        while (l2) {
            s2.push(l2 -> val);
            l2 = l2 -> next;
        }
        int flag = 0;
        ListNode* ans = nullptr;
        while (!s1.empty() || !s2.empty() || flag != 0) {
            int a = s1.empty() ? 0 : s1.top();
            int b = s2.empty() ? 0 : s2.top();
            if (!s1.empty()) s1.pop();
            if (!s2.empty()) s2.pop();
            int cur = a + b + flag;
            flag = cur / 10;
            cur %= 10;
            auto cur_node = new ListNode(cur);
            cur_node -> next = ans;
            ans = cur_node;
        }
        return ans;
    }
};
```

<span id="LFU缓存"></span>
## [460、LFU缓存(hard)](#back)
```cpp
设计并实现最不经常使用（LFU）缓存的数据结构。它应该支持以下操作：get 和 put。
get(key) - 如果键存在于缓存中，则获取键的值（总是正数），否则返回 -1。
put(key, value) - 如果键不存在，请设置或插入值。
当缓存达到其容量时，它应该在插入新项目之前，使最不经常使用的项目无效。
在此问题中，当存在平局（即两个或更多个键具有相同使用频率）时，最近最少使用的键将被去除。

LFUCache cache = new LFUCache( 2 /* capacity (缓存容量) */ );
cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // 返回 1
cache.put(3, 3);    // 去除 key 2
cache.get(2);       // 返回 -1 (未找到key 2)
cache.get(3);       // 返回 3
cache.put(4, 4);    // 去除 key 1
cache.get(1);       // 返回 -1 (未找到 key 1)
cache.get(3);       // 返回 3
cache.get(4);       // 返回 4

struct Node {
    int key, val, freq;
    Node(int _key, int _val, int _freq): key(_key), val(_val), freq(_freq){};
};

class LFUCache {
    // 双哈希表
    int minfreq, capacity;
    unordered_map<int, list<Node>::iterator> key_table;
    unordered_map<int, list<Node>> freq_table;

public:
    LFUCache(int _capacity) {
        minfreq = 0;
        capacity = _capacity;
        key_table.clear();
        freq_table.clear();
    }
    
    int get(int key) {
        if (capacity == 0) return -1;  // 缓存的阈值
        auto it = key_table.find(key);
        if (it == key_table.end()) return -1;  // 哈希表中没有找到节点
        list<Node>:: iterator node = it -> second;
        int val = node -> val, freq = node -> freq;
        freq_table[freq].erase(node);
        if (freq_table[freq].size() == 0) {
            freq_table.erase(freq);
            if (minfreq == freq) minfreq += 1;
        }
        // 插入到 freq + 1 中
        freq_table[freq + 1].push_front(Node(key, val, freq + 1));
        key_table[key] = freq_table[freq + 1].begin();
        return val;
    }
    
    void put(int key, int value) {
        if (capacity == 0) return ;
        auto it = key_table.find(key);
        if (it == key_table.end()) {  // put 的 key 不在缓存中
            // 缓存已满， put前需要删除
            if (key_table.size() == capacity) {
                auto it2 = freq_table[minfreq].back();
                key_table.erase(it2.key);
                freq_table[minfreq].pop_back();
                if (freq_table[minfreq].size() == 0) 
                    freq_table.erase(minfreq); 
            }
            freq_table[1].push_front(Node(key, value, 1));
            key_table[key] = freq_table[1].begin();
            minfreq = 1;
        }
        else {  // put 的 key 在缓存中
            list<Node>::iterator node = it -> second;
            int freq = node -> freq;
            freq_table[freq].erase(node);
            if (freq_table[freq].size() == 0) {
                freq_table.erase(freq);
                if (minfreq == freq) minfreq += 1;
            }
            freq_table[freq + 1].push_front(Node(key, value, freq + 1));
            key_table[key] = freq_table[freq + 1].begin();
        }
    }
};

/**
 * Your LFUCache object will be instantiated and called as such:
 * LFUCache* obj = new LFUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
```

<span id="统计重复个数"></span>
## [466、统计重复个数(hard)](#back)
```cpp
由 n 个连接的字符串 s 组成字符串 S，记作 S = [s,n]。例如，["abc",3]=“abcabcabc”。
如果我们可以从 s2 中删除某些字符使其变为 s1，则称字符串 s1 可以从字符串 s2 获得。
例如，根据定义，"abc" 可以从 “abdbec” 获得，但不能从 “acbbe” 获得。
现在给你两个非空字符串 s1 和 s2（每个最多 100 个字符长）
和两个整数 0 ≤ n1 ≤ 106 和 1 ≤ n2 ≤ 106。
现在考虑字符串 S1 和 S2，其中 S1=[s1,n1] 、S2=[s2,n2] 。
请你找出一个可以满足使[S2,M] 从 S1 获得的最大整数 M 。

输入：
s1 ="acb",n1 = 4
s2 ="ab",n2 = 2
输出：
2

class Solution {
public:
    int getMaxRepetitions(string s1, int n1, string s2, int n2) {
        // 参考链接：https://leetcode-cn.com/problems/count-the-repetitions/solution/tong-ji-zhong-fu-ge-shu-by-leetcode-solution/
        if (n1 == 0) return 0;
        int s1cnt = 0, s2cnt = 0, index = 0;
        unordered_map<int, pair<int, int>> recall;
        pair<int, int> pre_loop, in_loop;
        while (true) {
            ++s1cnt;
            for (char ch : s1) {
                if (ch == s2[index]) {
                    index += 1;
                    if (index == s2.size()) {
                        ++s2cnt;
                        index = 0;
                    }
                }
            }
            if (s1cnt == n1) {
                return s2cnt / n2;
            }
            if (recall.count(index)) {
                auto [s1cnt_prime, s2cnt_prime] = recall[index];
                pre_loop = {s1cnt_prime, s2cnt_prime};
                in_loop = {s1cnt - s1cnt_prime, s2cnt - s2cnt_prime};
                break;
            }
            else {
                recall[index] = {s1cnt, s2cnt};
            }
        }
        int ans = pre_loop.second + (n1 - pre_loop.first) / in_loop.first * in_loop.second;
        int rest = (n1 - pre_loop.first) % in_loop.first;
        for (int i = 0; i < rest; i++) {
            for (char ch : s1) {
                if (ch == s2[index]) {
                    ++index;
                    if (index == s2.size()) {
                        ++ans;
                        index = 0;
                    }
                }
            }
        }
        return ans / n2;
    }
};
```

<span id="01矩阵"></span>
## [542、01 矩阵(medium)](#back)
```cpp
给定一个由 0 和 1 组成的矩阵，找出每个元素到最近的 0 的距离。
两个相邻元素间的距离为 1 。

输入:
0 0 0
0 1 0
1 1 1
输出:
0 0 0
0 1 0
1 2 1

class Solution {
public:
    vector<vector<int>> updateMatrix(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        queue<pair<int, int>> que;
        vector<vector<int>> dis(m, vector<int> (n));  // 存储距离结果
        vector<vector<int>> flag(m, vector<int> (n));  // 用于确认当前点是否已经被处理
        int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        // 初始化队列，将位置为 0 的加入队列中
        for (int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    que.push({i, j});
                    flag[i][j] = 1;  // 表示当前位置已经被处理
                }
            }
        }

        // 广度优先搜索
        while (!que.empty()) {
            auto [o_i, o_j] = que.front();
            que.pop();
            for (int i = 0; i < 4; i++) {
                int new_i = o_i + dirs[i][0];
                int new_j = o_j + dirs[i][1];
                if (new_i >= 0 && new_i < m && new_j >= 0 && new_j < n && flag[new_i][new_j] != 1) {
                    dis[new_i][new_j] += dis[o_i][o_j] + 1;
                    que.push({new_i, new_j});
                    flag[new_i][new_j] = 1;
                }
            }
        }
        return dis;
    }
};
```

<span id="二叉树的直径"></span>
## [543、二叉树的直径(easy)](#back)
```cpp
给定一棵二叉树，你需要计算它的直径长度。
一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过根结点。

给定二叉树
          1
         / \
        2   3
       / \     
      4   5    
返回 3, 它的长度是路径 [4,2,1,3] 或者 [5,2,1,3]。
注意：两结点之间的路径长度是以它们之间边的数目表示。

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
    int res;

    int dfs(TreeNode* root) {
        if (root == nullptr) return 0;
        int left = dfs(root -> left);
        int right = dfs(root -> right);
        res = max(res, left + right);
        return max(left, right) + 1;
    }

    int diameterOfBinaryTree(TreeNode* root) {
        dfs(root);
        return res;
    }
};
```

<span id="另一个树的子树"></span>
## [572、另一个树的子树(easy)](#back)
```cpp
给定两个非空二叉树 s 和 t，检验 s 中是否包含和 t 具有相同结构和节点值的子树。
s 的一个子树包括 s 的一个节点和这个节点的所有子孙。s 也可以看做它自身的一棵子树。

给定的树 s:
     3
    / \
   4   5
  / \
 1   2
给定的树 t：
   4 
  / \
 1   2
返回 true，因为 t 与 s 的一个子树拥有相同的结构和节点值。

给定的树 s：
     3
    / \
   4   5
  / \
 1   2
    /
   0
给定的树 t：
   4
  / \
 1   2
返回 false。

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
    bool sameTree(TreeNode* pRoot1, TreeNode* pRoot2) {
        if (pRoot1 == nullptr && pRoot2 == nullptr) return true;
        if (pRoot1 == nullptr || pRoot2 == nullptr) return false;
        if (pRoot1 -> val == pRoot2 -> val) {
            return sameTree(pRoot1 -> left, pRoot2 -> left) && sameTree(pRoot1 -> right, pRoot2 -> right);
        }
        else return false;
    }

    bool isSubtree(TreeNode* s, TreeNode* t) {
        if (s == nullptr || t == nullptr) return false;
        return sameTree(s, t) || isSubtree(s -> left, t) || isSubtree(s -> right, t);
    }
};
```

<span id="岛屿的最大面积"></span>
## [695、岛屿的最大面积(medium)](#back)
```cpp
给定一个包含了一些 0 和 1的非空二维数组 grid , 一个 岛屿 是由四个方向 (水平或垂直) 的 1 (代表土地) 构成的组合。
你可以假设二维矩阵的四个边缘都被水包围着。
找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为0。)

[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
对于上面这个给定矩阵应返回 6。注意答案不应该是11，因为岛屿只能包含水平或垂直的四个方向的‘1’。

[[0,0,0,0,0,0,0,0]]
对于上面这个给定的矩阵, 返回 0。

注意: 给定的矩阵grid 的长度和宽度都不超过 50。

class Solution {
public:
    int dir[4][4] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    int dfs(vector<vector<int>>& grid, int x, int y) {
        if (x < 0 || x >= grid.size() || y < 0 || y >= grid[0].size() || grid[x][y] == 0)
            return 0;
        grid[x][y] = 0;
        int ans = 1;
        for(int i = 0; i < 4; i++) {
            int new_x = x + dir[i][0];
            int new_y = y + dir[i][1];
            ans += dfs(grid, new_x, new_y);
        }
        return ans;
    }

    int maxAreaOfIsland(vector<vector<int>>& grid) {
        // 动态规划
        int max_ = 0;
        for(int i = 0; i < grid.size(); i++) {
            for(int j = 0; j < grid[0].size(); j++) {
                max_ = max(max_, dfs(grid, i, j));
            }
        }
        return max_;
    }
};
```

<span id="单词的压缩编码"></span>
## [820、单词的压缩编码(medium)](#back)
```cpp
给定一个单词列表，我们将这个列表编码成一个索引字符串 S 与一个索引列表 A。
例如，如果这个列表是 ["time", "me", "bell"]，
我们就可以将其表示为 S = "time#bell#" 和 indexes = [0, 2, 5]。
对于每一个索引，我们可以通过从字符串 S 中索引的位置开始读取字符串，直到 "#" 结束，来恢复我们之前的单词列表。
那么成功对给定单词列表进行编码的最小字符串长度是多少呢？

输入: words = ["time", "me", "bell"]
输出: 10
说明: S = "time#bell#" ， indexes = [0, 2, 5] 。
 
1 <= words.length <= 2000
1 <= words[i].length <= 7
每个单词都是小写字母 。

// 辅助；字典树
class TrieNode {
    TrieNode* children[26];
public:
    int count;
    TrieNode(){
        for (int i = 0 ; i < 26; i++) children[i] = nullptr;
        count = 0;
    }
    TrieNode* get(char c) {
        if (children[c - 'a'] == nullptr) {
            children[c - 'a'] = new TrieNode();
            count++;
        }
        return children[c - 'a'];
    }
};


class Solution {
public:
    int minimumLengthEncoding(vector<string>& words) {
        // 参考：https://leetcode-cn.com/problems/short-encoding-of-words/solution/dan-ci-de-ya-suo-bian-ma-by-leetcode-solution/
        // // 方法1；存储后缀; 时间复杂度高
        // unordered_set<string> s(words.begin(), words.end());
        // for (auto w1 : words) {
        //     for (int k =1; k < w1.size(); k++) {
        //         s.erase(w1.substr(k));
        //     }
        // }
        // int ans = 0;
        // for (auto s1 : s) {
        //     ans += s1.size() + 1;
        // }
        // return ans;

        // 方法2； 字典树；时间复杂度低
        TrieNode* trie = new TrieNode();
        unordered_map<TrieNode*, int> node;
        for (int i = 0; i < words.size(); i++) {
            string w = words[i];
            TrieNode* cur = trie;
            for (int j = w.size() - 1; j >= 0; j--) {
                cur = cur -> get(w[j]);
            }
            node[cur] = i;
        }
        int ans = 0;
        for (auto [n, idx] : node) {
            if (n -> count == 0) {
                ans += words[idx].size() + 1;
            }
        }
        return ans;
    }
};
```

<span id="矩形重叠"></span>
## [836、矩形重叠(easy)](#back)
```cpp
矩形以列表 [x1, y1, x2, y2] 的形式表示，其中 (x1, y1) 为左下角的坐标，(x2, y2) 是右上角的坐标。
如果相交的面积为正，则称两矩形重叠。需要明确的是，只在角或边接触的两个矩形不构成重叠。
给出两个矩形，判断它们是否重叠并返回结果。

输入：rec1 = [0,0,2,2], rec2 = [1,1,3,3]
输出：true

输入：rec1 = [0,0,1,1], rec2 = [1,0,2,1]
输出：false

class Solution {
public:
    bool isRectangleOverlap(vector<int>& rec1, vector<int>& rec2) {
        // 检查区域
        // 重叠时即有重叠部分也是矩形
        // 两个矩形在 x 、y 轴上的投影有交集
        return (min(rec1[2], rec2[2]) > max(rec1[0], rec2[0])) && (min(rec1[3], rec2[3]) > max(rec1[1], rec2[1]));
    }
};
```

<span id="链表的中间结点"></span>
## [876、链表的中间结点(easy)](#back)
```cpp
给定一个带有头结点 head 的非空单链表，返回链表的中间结点。
如果有两个中间结点，则返回第二个中间结点。

输入：[1,2,3,4,5]
输出：此列表中的结点 3 (序列化形式：[3,4,5])
返回的结点值为 3 。 (测评系统对该结点序列化表述是 [3,4,5])。
注意，我们返回了一个 ListNode 类型的对象 ans，这样：
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, 以及 ans.next.next.next = NULL.

输入：[1,2,3,4,5,6]
输出：此列表中的结点 4 (序列化形式：[4,5,6])
由于该列表有两个中间结点，值分别为 3 和 4，我们返回第二个结点。

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
    ListNode* middleNode(ListNode* head) {
        ListNode* p_slow = head;
        ListNode* p_fast = head;
        while(p_fast != nullptr && p_fast -> next != nullptr) {
            p_slow = p_slow -> next;
            p_fast = p_fast -> next -> next;
        }
        return p_slow;
    }
};
```

<span id="鸡蛋掉落"></span>
## [887、鸡蛋掉落(hard)](#back)
```cpp
你将获得 K 个鸡蛋，并可以使用一栋从 1 到 N  共有 N 层楼的建筑。
每个蛋的功能都是一样的，如果一个蛋碎了，你就不能再把它掉下去。
你知道存在楼层 F ，满足 0 <= F <= N 任何从高于 F 的楼层落下的鸡蛋都会碎，
从 F 楼层或比它低的楼层落下的鸡蛋都不会破。
每次移动，你可以取一个鸡蛋（如果你有完整的鸡蛋）并把它从任一楼层 X 扔下（满足 1 <= X <= N）。
你的目标是确切地知道 F 的值是多少。
无论 F 的初始值如何，你确定 F 的值的最小移动次数是多少？

输入：K = 1, N = 2
输出：2
解释：
鸡蛋从 1 楼掉落。如果它碎了，我们肯定知道 F = 0 。
否则，鸡蛋从 2 楼掉落。如果它碎了，我们肯定知道 F = 1 。
如果它没碎，那么我们肯定知道 F = 2 。
因此，在最坏的情况下我们需要移动 2 次以确定 F 是多少。

输入：K = 2, N = 6
输出：3

输入：K = 3, N = 14
输出：4

class Solution {
public:
    int superEggDrop(int K, int N) {
        // 参考链接：https://leetcode-cn.com/problems/super-egg-drop/solution/ji-dan-diao-luo-by-leetcode-solution/  方法3：数学法
        if (N == 1) return 1;
        vector<vector<int>> dp(N + 1, vector<int> (K + 1, 0));
        for (int i = 1; i <= K; i++) dp[1][i] = 1;
        int ans = -1;
        for (int i = 2; i <= N; i++) {
            for (int j = 1; j <= K; j++) {
                dp[i][j] = 1 + dp[i - 1][j - 1] + dp[i - 1][j];
            }
            if (dp[i][K] >= N) {
                ans = i;
                break;
            }
        }
        return ans;
    }
};
```

<span id="三维形体的表面积"></span>
## [892、三维形体的表面积(easy)](#back)
```cpp
在 N * N 的网格上，我们放置一些 1 * 1 * 1  的立方体。
每个值 v = grid[i][j] 表示 v 个正方体叠放在对应单元格 (i, j) 上。
请你返回最终形体的表面积。 

输入：[[2]]
输出：10

输入：[[1,2],[3,4]]
输出：34
解释：位置(0，0）竖向放置 1 个单位的长方体，位置（0，1）竖向放置 2 个单位的长方体
     位置(1，0）竖向放置 3 个单位的长方体，位置（1，1）竖向放置 4 个单位的长方体

输入：[[1,0],[0,2]]
输出：16

输入：[[1,1,1],[1,0,1],[1,1,1]]
输出：32

输入：[[2,2,2],[2,1,2],[2,2,2]]
输出：46

class Solution {
public:
    int surfaceArea(vector<vector<int>>& grid) {
        int len_ = grid.size(), area_ = 0;
        for (int i = 0; i < len_; i++) {
            for (int j = 0; j < len_; j++) {
                if (grid[i][j] > 0) {
                    area_ += (grid[i][j] << 2) + 2;  // 加上当前位置处图形的表面积
                    area_ -= i > 0 ? min(grid[i][j], grid[i - 1][j]) << 1: 0;  // 减去 i 重合面
                    area_ -= j > 0 ? min(grid[i][j], grid[i][j - 1]) << 1: 0;  // 减去 j 重合面
                }
            }
        }
        return area_;
    }
};
```

<span id="排序数组"></span>
## [912、排序数组(medium)](#back)
```cpp
给你一个整数数组 nums，请你将该数组升序排列。

输入：nums = [5,2,3,1]
输出：[1,2,3,5]

输入：nums = [5,1,1,2,0,0]
输出：[0,0,1,1,2,5]

class Solution {
public:
    int partion(vector<int>& nums, int l, int r) {
        int pivot = nums[r];
        int i = l - 1;
        for (int j = l; j < r; j++) {
            if (nums[j] <= pivot) {
                i = i + 1;
                swap(nums[i], nums[j]);
            }
        }
        swap(nums[i + 1], nums[r]);
        return i + 1;
    }

    int randomized_partition(vector<int>& nums, int l, int r) {
        int i = rand() % (r - l + 1) + l;
        swap(nums[r], nums[i]);
        return partion(nums, l, r);
    }

    void randomized_quicksort(vector<int>& nums, int l ,int r) {
        if (l < r) {
            int pos = randomized_partition(nums, l, r);
            randomized_quicksort(nums, l ,pos - 1);
            randomized_quicksort(nums, pos + 1, r);
        }
    }

    vector<int> sortArray(vector<int>& nums) {
        // 快排
        srand((unsigned)time(NULL));  // 随机种子
        randomized_quicksort(nums, 0, nums.size() - 1);
        return nums;
    }
};
```

<span id="卡牌分组"></span>
## [914、卡牌分组(easy)](#back)
```cpp
给定一副牌，每张牌上都写着一个整数。
此时，你需要选定一个数字 X，使我们可以将整副牌按下述规则分成 1 组或更多组：
每组都有 X 张牌。
组内所有的牌上都写着相同的整数。
仅当你可选的 X >= 2 时返回 true。

输入：[1,2,3,4,4,3,2,1]
输出：true
解释：可行的分组是 [1,1]，[2,2]，[3,3]，[4,4]

输入：[1,1,1,2,2,2,3,3]
输出：false
解释：没有满足要求的分组。

输入：[1]
输出：false
解释：没有满足要求的分组。

输入：[1,1]
输出：true
解释：可行的分组是 [1,1]

输入：[1,1,2,2,2,2]
输出：true
解释：可行的分组是 [1,1]，[2,2]，[2,2]

提示：
1 <= deck.length <= 10000
0 <= deck[i] < 10000

class Solution {
public:
    bool hasGroupsSizeX(vector<int>& deck) {
        map<int, int> mp_;
        for (auto d : deck) 
            mp_[d]++;
        int a = mp_[deck[0]];
        for (auto d1 : deck) {
            if (a < 2) return false;
            a = gcd(a, mp_[d1]);  // 最大公约数
        }
        return true;
    }
};
```

<span id="使数组唯一的最小增量"></span>
## [945、使数组唯一的最小增量(medium)](#back)
```cpp
给定整数数组 A，每次 move 操作将会选择任意 A[i]，并将其递增 1。
返回使 A 中的每个值都是唯一的最少操作次数。

输入：[1,2,2]
输出：1
解释：经过一次 move 操作，数组将变为 [1, 2, 3]。

输入：[3,2,1,2,1,7]
输出：6
解释：经过 6 次 move 操作，数组将变为 [3, 4, 1, 2, 5, 7]。
可以看出 5 次或 5 次以下的 move 操作是不能让数组的每个值唯一的。

class Solution {
public:
    int minIncrementForUnique(vector<int>& A) {
        sort(A.begin(), A.end());
        int all_ = 0;
        for(int i = 1; i < A.size(); i++) {
            if (A[i] < A[i - 1]) {
                all_ += A[i - 1] + 1 - A[i];
                A[i] = A[i - 1] + 1;
            }
            else if(A[i] == A[i - 1]) {
                A[i] += 1;
                all_ += 1;
            }
        }
        return all_;
    }
};
```

<span id="最低票价"></span>
## [983、最低票价(medium)](#back)
```cpp
在一个火车旅行很受欢迎的国度，你提前一年计划了一些火车旅行。
在接下来的一年里，你要旅行的日子将以一个名为 days 的数组给出。
每一项是一个从 1 到 365 的整数。

火车票有三种不同的销售方式：
一张为期一天的通行证售价为 costs[0] 美元；
一张为期七天的通行证售价为 costs[1] 美元；
一张为期三十天的通行证售价为 costs[2] 美元。
通行证允许数天无限制的旅行。 
例如，如果我们在第 2 天获得一张为期 7 天的通行证，
那么我们可以连着旅行 7 天：第 2 天、第 3 天、第 4 天、第 5 天、第 6 天、第 7 天和第 8 天。
返回你想要完成在给定的列表 days 中列出的每一天的旅行所需要的最低消费。

输入：days = [1,4,6,7,8,20], costs = [2,7,15]
输出：11
解释： 
例如，这里有一种购买通行证的方法，可以让你完成你的旅行计划：
在第 1 天，你花了 costs[0] = $2 买了一张为期 1 天的通行证，它将在第 1 天生效。
在第 3 天，你花了 costs[1] = $7 买了一张为期 7 天的通行证，它将在第 3, 4, ..., 9 天生效。
在第 20 天，你花了 costs[0] = $2 买了一张为期 1 天的通行证，它将在第 20 天生效。
你总共花了 $11，并完成了你计划的每一天旅行。

输入：days = [1,2,3,4,5,6,7,8,9,10,30,31], costs = [2,7,15]
输出：17
解释：
例如，这里有一种购买通行证的方法，可以让你完成你的旅行计划： 
在第 1 天，你花了 costs[2] = $15 买了一张为期 30 天的通行证，它将在第 1, 2, ..., 30 天生效。
在第 31 天，你花了 costs[0] = $2 买了一张为期 1 天的通行证，它将在第 31 天生效。 
你总共花了 $17，并完成了你计划的每一天旅行。

class Solution {
public:
    vector<int> days, costs;
    vector<int> memo;
    int duration[3] = {1, 7, 30};

    int mincostTickets(vector<int>& days, vector<int>& costs) {
        // dp[i]：表示能够完成第 days[i] 天到最后的旅行计划的最小花费
        this -> days = days;
        this -> costs =  costs;
        memo.assign(days.size(), -1);
        return dp(0);        
    }

    int dp(int i) {
        if (i >= days.size()) return 0;
        if (memo[i] != -1) return memo[i];
        memo[i] = INT_MAX;
        int j = i;
        for (int k = 0; k < 3; k++) {
            while (j < days.size() && days[j] < days[i] + duration[k]) {
                j++;
            }
            memo[i] = min(memo[i], dp(j) + costs[k]);
        }
        return memo[i];
    }
};
```

<span id="腐烂的橘子"></span>
## [994、腐烂的橘子(easy)](#back)

<div align=center><img src="https://github.com/FangChao1086/LeetCode_Solutions/blob/master/依赖文件/LeetCode994_腐烂的橘子.png"></div>

```cpp
在给定的网格中，每个单元格可以有以下三个值之一：
值 0 代表空单元格；
值 1 代表新鲜橘子；
值 2 代表腐烂的橘子。
每分钟，任何与腐烂的橘子（在 4 个正方向上）相邻的新鲜橘子都会腐烂。
返回直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1。

输入：[[2,1,1],[1,1,0],[0,1,1]]
输出：4

输入：[[2,1,1],[0,1,1],[1,0,1]]
输出：-1
解释：左下角的橘子（第 2 行， 第 0 列）永远不会腐烂，因为腐烂只会发生在 4 个正向上。

输入：[[0,2]]
输出：0
解释：因为 0 分钟时已经没有新鲜橘子了，所以答案就是 0 。
 
提示：
1 <= grid.length <= 10
1 <= grid[0].length <= 10
grid[i][j] 仅为 0、1 或 2

class Solution {
public:
    int orangesRotting(vector<vector<int>>& grid) {
        // bfs
        int min_ = 0, count_one = 0; 
        queue<pair<int, int>> que;
        int rows = grid.size();
        int cols = grid[0].size();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 1) count_one++;
                else if(grid[i][j] == 2)  que.push({i, j});
            }
        }
        vector<pair<int, int>> dirs = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
        while (!que.empty()) {
            int que_size = que.size();
            bool rotten = 0;  // 记录是否发生变质
            for (int i = 0; i < que_size; i++) {
                auto tmp = que.front();
                que.pop();
                for(auto dir : dirs) {
                    int new_i = dir.first + tmp.first;
                    int new_j = dir.second + tmp.second;
                    if (new_i >= 0 && new_i < rows && new_j >= 0 && new_j < cols && grid[new_i][new_j] == 1) {
                        grid[new_i][new_j] = 2;
                        que.push({new_i, new_j});   
                        rotten = true;
                        count_one--;
                    }
                }
            }
            if(rotten) min_++;
        }
        return count_one > 0 ? -1 : min_;
    }
};
```

<span id="车的可用捕获量"></span>
## [999、车的可用捕获量(easy)](#back)
```cpp
在一个 8 x 8 的棋盘上，有一个白色车（rook）。
也可能有空方块，白色的象（bishop）和黑色的卒（pawn）。
它们分别以字符 “R”，“.”，“B” 和 “p” 给出。
大写字符表示白棋，小写字符表示黑棋。

车按国际象棋中的规则移动：它选择四个基本方向中的一个（北，东，西和南），
然后朝那个方向移动，直到它选择停止、到达棋盘的边缘或移动到同一方格来捕获该方格上颜色相反的卒。
另外，车不能与其他友方（白色）象进入同一个方格。
返回车能够在一次移动中捕获到的卒的数量。
注意：向左、右、上、下每个反向最多只能捕获一个

输入：
[[".",".",".",".",".",".",".","."],
[".",".",".","p",".",".",".","."],
[".",".",".","R",".",".",".","p"],
[".",".",".",".",".",".",".","."],
[".",".",".",".",".",".",".","."],
[".",".",".","p",".",".",".","."],
[".",".",".",".",".",".",".","."],
[".",".",".",".",".",".",".","."]]
输出：3
解释：在本例中，车能够捕获所有的卒。

输入：
[[".",".",".",".",".",".",".","."],
[".","p","p","p","p","p",".","."],
[".","p","p","B","p","p",".","."],
[".","p","B","R","B","p",".","."],
[".","p","p","B","p","p",".","."],
[".","p","p","p","p","p",".","."],
[".",".",".",".",".",".",".","."],
[".",".",".",".",".",".",".","."]]
输出：0
解释：象阻止了车捕获任何卒。

输入：
[[".",".",".",".",".",".",".","."],
[".",".",".","p",".",".",".","."],
[".",".",".","p",".",".",".","."],
["p","p",".","R",".","p","B","."],
[".",".",".",".",".",".",".","."],
[".",".",".","B",".",".",".","."],
[".",".",".","p",".",".",".","."],
[".",".",".",".",".",".",".","."]]
输出：3
解释：车可以捕获位置 b5，d6 和 f5 的卒。
 
提示：
board.length == board[i].length == 8
board[i][j] 可以是 'R'，'.'，'B' 或 'p'
只有一个格子上存在 board[i][j] == 'R'

class Solution {
public:
    int numRookCaptures(vector<vector<char>>& board) {
        // 棋盘大小固定 8 * 8
        int dir[4][4] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int r_i = 0, r_j = 0;
        int count_ = 0;
        for (int i = 0; i < 8; i++) {
            for (int j =0; j < 8; j++) {
                if (board[i][j] == 'R') {
                    r_i = i;
                    r_j = j;
                    break;
                }
            }
        }
        for (int i = 0; i < 4; i++) {
            for (int step = 1; ;step++) {
                int new_r_i = r_i + step * dir[i][0];
                int new_r_j = r_j + step * dir[i][1];
                if (new_r_i < 0 || new_r_i >= 8 || new_r_j < 0 || new_r_j >= 8 || board[new_r_i][new_r_j] == 'B') {
                    break;
                }
                else if (board[new_r_i][new_r_j] == 'p') {
                    count_++;
                    break;
                }
            }
        }
        return count_;
    }
};
```

<span id="将数组分成和相等的三个部分"></span>
## [1013、将数组分成和相等的三个部分(easy)](#back)
```cpp
给你一个整数数组 A，只有可以将其划分为三个和相等的非空部分时才返回 true，否则返回 false。
形式上，如果可以找出索引 i+1 < j 且满足 
(A[0] + A[1] + ... + A[i] == A[i+1] + A[i+2] + ... + A[j-1] == A[j] + A[j-1] + ... + A[A.length - 1]) 
就可以将数组三等分。

输出：[0,2,1,-6,6,-7,9,1,2,0,1]
输出：true
解释：0 + 2 + 1 = -6 + 6 - 7 + 9 + 1 = 2 + 0 + 1

输入：[0,2,1,-6,6,7,9,-1,2,0,1]
输出：false

输入：[3,3,6,5,-2,2,5,1,-9,4]
输出：true
解释：3 + 3 = 6 = 5 - 2 + 2 + 5 + 1 - 9 + 4
 
提示：
3 <= A.length <= 50000
-10^4 <= A[i] <= 10^4

class Solution {
public:
    bool canThreePartsEqualSum(vector<int>& A) {
        // 计算和sum
        // 查找子数组等于 sum / 3 的个数， 找到 3 个返回 true;
        int sum = accumulate(A.begin(), A.end(), 0);
        if (sum % 3 != 0) return false;
        int sum_one = 0, count_ = 0;
        for (int i = 0; i < A.size(); i++) {
            sum_one += A[i]; 
            if (sum_one == sum / 3) {
                sum_one = 0;
                count_++;
            }
            if(count_ == 3) return true;
        }
        return false;
    }
};
```

<span id="字符串的最大公因子"></span>
## [1071、字符串的最大公因子(easy)](#back)
```cpp
对于字符串 S 和 T，只有在 S = T + ... + T（T 与自身连接 1 次或多次）时，我们才认定 “T 能除尽 S”。
返回最长字符串 X，要求满足 X 能除尽 str1 且 X 能除尽 str2。

输入：str1 = "ABCABC", str2 = "ABC"
输出："ABC"

输入：str1 = "ABABAB", str2 = "ABAB"
输出："AB"

输入：str1 = "LEET", str2 = "CODE"
输出：""

class Solution {
public:
    // 求最大公约数， 辗转相除法
    int gcd(int x, int y) {
        int z = y;
        while (x % y != 0) {
            z = x % y;
            x = y;
            y = z;
        }
        return z;
    }

    string gcdOfStrings(string str1, string str2) {
        // 数学方法
        // 当 str1 + str2 == str2 + str1 时必然存在，不相等时则不存在
        if (str1 + str2 != str2 + str1) return "";
        return str1.substr(0, gcd(str1.size(), str2.size()));
    }
};
```

<span id="山脉数组中查找目标值"></span>
## [1095、山脉数组中查找目标值(hard)](#back)
```cpp
给你一个 山脉数组 mountainArr，请你返回能够使得 mountainArr.get(index) 等于 target 最小的下标 index 值。
如果不存在这样的下标 index，就请返回 -1。

如果数组 A 是一个山脉数组的话，那它满足如下条件：
首先，A.length >= 3
其次，在 0 < i < A.length - 1 条件下，存在 i 使得：
A[0] < A[1] < ... A[i-1] < A[i]
A[i] > A[i+1] > ... > A[A.length - 1] 
你不能直接访问该山脉数组，必须通过 MountainArray 接口来获取数据：
MountainArray.get(k) - 会返回数组中索引为k 的元素（下标从 0 开始）
MountainArray.length() - 会返回该数组的长度
 
注意：
对 MountainArray.get 发起超过 100 次调用的提交将被视为错误答案。
此外，任何试图规避判题系统的解决方案都将会导致比赛资格被取消。 

输入：array = [1,2,3,4,5,3,1], target = 3
输出：2
解释：3 在数组中出现了两次，下标分别为 2 和 5，我们返回最小的下标 2。

输入：array = [0,1,2,4,2,1], target = 3
输出：-1
解释：3 在数组中没有出现，返回 -1。

/**
 * // This is the MountainArray's API interface.
 * // You should not implement it, or speculate about its implementation
 * class MountainArray {
 *   public:
 *     int get(int index);
 *     int length();
 * };
 */

class Solution {
public:
    int binary_search(MountainArray &mountainArr, int target, int left, int right, int reverse) {
        target *= reverse;  // 全部变为升序查找
        while (left <= right) {
            int mid = (left + right) >> 1;
            int cur = reverse * (mountainArr.get(mid));
            if (cur == target) {
                return mid;
            }
            else if (cur < target) {
                left = mid + 1;
            }
            else right = mid - 1;
        }
        return -1;
    }


    int findInMountainArray(int target, MountainArray &mountainArr) {
        int left = 0, right = mountainArr.length() - 1;
        // 利用二分查找找到峰值
        while (left < right) {
            int mid = (left + right) >> 1;
            if (mountainArr.get(mid) < mountainArr.get(mid + 1)) {
                left = mid + 1;
            }
            else right = mid;
        }
        int peak = left;
        int index = binary_search(mountainArr, target, 0, peak, 1);  // 左边升序, 其中最后的 1 表示升序
        if (index != -1) {
            return index;
        }
        else return binary_search(mountainArr, target, peak + 1, mountainArr.length() - 1, -1);  // 右边降序, 其中最后的 -1 表示降序
    }
};
```

<span id="分糖果2"></span>
## [1103、分糖果 II(easy)](#back)
```cpp
排排坐，分糖果。
我们买了一些糖果 candies，打算把它们分给排好队的 n = num_people 个小朋友。
给第一个小朋友 1 颗糖果，第二个小朋友 2 颗，依此类推，直到给最后一个小朋友 n 颗糖果。
然后，我们再回到队伍的起点，给第一个小朋友 n + 1 颗糖果，第二个小朋友 n + 2 颗，依此类推，直到给最后一个小朋友 2 * n 颗糖果。
重复上述过程（每次都比上一次多给出一颗糖果，当到达队伍终点后再次从队伍起点开始），直到我们分完所有的糖果。注意，就算我们手中的剩下糖果数不够（不比前一次发出的糖果多），这些糖果也会全部发给当前的小朋友。
返回一个长度为 num_people、元素之和为 candies 的数组，以表示糖果的最终分发情况（即 ans[i] 表示第 i 个小朋友分到的糖果数）。

输入：candies = 7, num_people = 4
输出：[1,2,3,1]
解释：
第一次，ans[0] += 1，数组变为 [1,0,0,0]。
第二次，ans[1] += 2，数组变为 [1,2,0,0]。
第三次，ans[2] += 3，数组变为 [1,2,3,0]。
第四次，ans[3] += 1（因为此时只剩下 1 颗糖果），最终数组变为 [1,2,3,1]。

输入：candies = 10, num_people = 3
输出：[5,2,3]
解释：
第一次，ans[0] += 1，数组变为 [1,0,0]。
第二次，ans[1] += 2，数组变为 [1,2,0]。
第三次，ans[2] += 3，数组变为 [1,2,3]。
第四次，ans[0] += 4，最终数组变为 [5,2,3]。
 
提示：
1 <= candies <= 10^9
1 <= num_people <= 1000

class Solution {
public:
    vector<int> distributeCandies(int candies, int num_people) {
        vector<int> res(num_people, 0);
        int i = 0;
        while(candies > 0) {
            res[(i + num_people ) % num_people] += candies >=  i + 1 ? i + 1 : candies;
            candies -= i + 1;
            i++;
        }
        return res;
    }
};
```

<span id="有效括号的嵌套深度"></span>
## [1111、有效括号的嵌套深度(medium)](#back)
```cpp
给你一个「有效括号字符串」 seq，
请你将其分成两个不相交的有效括号字符串，A 和 B，
并使这两个字符串的深度最小。

不相交：每个 seq[i] 只能分给 A 和 B 二者中的一个，不能既属于 A 也属于 B 。
A 或 B 中的元素在原字符串中可以不连续。
max(depth(A), depth(B)) 的可能取值最小。
answer[i] = 0，seq[i] 分给 A 。
answer[i] = 1，seq[i] 分给 B 。

输入：seq = "(()())"
输出：[0,1,1,1,1,0]

输入：seq = "()(())()"
输出：[0,0,0,1,1,0,1,1]

例如：""，"()()"，和 "()(()())" 都是有效括号字符串，
嵌套深度分别为 0，1，2，而 ")(" 和 "(()" 都不是有效括号字符串。

class Solution {
public:
    vector<int> maxDepthAfterSplit(string seq) {
        vector<int> ans;
        int d = 0;
        for (auto s : seq) {
            if (s == '(') {
                ans.push_back(d % 2);
                d++;
            }
            else {
                d--;
                ans.push_back(d % 2);
            }
        }
        return ans;
    }
};
```

<span id="拼写单词"></span>
## [1160、拼写单词(easy)](#back)
```cpp
给你一份『词汇表』（字符串数组） words 和一张『字母表』（字符串） chars。
假如你可以用 chars 中的『字母』（字符）拼写出 words 中的某个『单词』（字符串），那么我们就认为你掌握了这个单词。
注意：每次拼写时，chars 中的每个字母都只能用一次。
返回词汇表 words 中你掌握的所有单词的长度之和。

输入：words = ["cat","bt","hat","tree"], chars = "atach"
输出：6
解释： 
可以形成字符串 "cat" 和 "hat"，所以答案是 3 + 3 = 6。

输入：words = ["hello","world","leetcode"], chars = "welldonehoneyr"
输出：10
解释：
可以形成字符串 "hello" 和 "world"，所以答案是 5 + 5 = 10。
 
提示：
1 <= words.length <= 1000
1 <= words[i].length, chars.length <= 100
所有字符串中都仅包含小写英文字母

class Solution {
public:
    int countCharacters(vector<string>& words, string chars) {
        int len_ = words.size();
        int all_ = 0;
        unordered_map<char, int> mp_;
        for (auto a : chars) {
            mp_[a]++;
        }        
        for (int i = 0; i < len_; i++) {
            int flag = 1;
            int word_size = words[i].size();
            unordered_map<char, int> mp_word;
            for (auto b : words[i]) {
                mp_word[b]++;
            }
            for (auto b : words[i]) {
                if (mp_word[b] > mp_[b]) {
                    flag = 0;
                    break;
                }
            }
            if(flag) {
                all_ += word_size;
            }
        }
        return all_;
    }
};
```

<span id="地图分析"></span>
## [1162、地图分析(medium)](#back)
```cpp
你现在手里有一份大小为 N x N 的『地图』（网格） grid，上面的每个『区域』（单元格）都用 0 和 1 标记好了。
其中 0 代表海洋，1 代表陆地，你知道距离陆地区域(所有陆地)最远的海洋区域是是哪一个吗？
请返回该海洋区域到离它最近的陆地区域的距离。
我们这里说的距离是『曼哈顿距离』（ Manhattan Distance）：
(x0, y0) 和 (x1, y1) 这两个区域之间的距离是 |x0 - x1| + |y0 - y1| 。
如果我们的地图上只有陆地或者海洋，请返回 -1。

输入：[[1,0,1],
      [0,0,0],
      [1,0,1]]
输出：2
解释： 
海洋区域 (1, 1) 和所有陆地区域之间的距离都达到最大，最大距离为 2。

输入：[[1,0,0],
      [0,0,0],
      [0,0,0]]
输出：4
解释： 
海洋区域 (2, 2) 和所有陆地区域之间的距离都达到最大，最大距离为 4。

提示：
1 <= grid.length == grid[0].length <= 100
grid[i][j] 不是 0 就是 1

class Solution {
public:
    int maxDistance(vector<vector<int>>& grid) {
        // 这里距离最远指的是该海洋距离所有的陆地的曼哈顿距离最远，而非某一个陆地
        int dir[4][4] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        queue<pair<int, int>> que;
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == 0) continue;
                que.push({i, j});  // 存陆地
            }
        }
        if (que.size() == 0 || que.size() == grid.size() * grid[0].size())  // 如果全是陆地或者全是海洋
            return -1;
        int ans = 0;
        while (!que.empty()) {
            auto q = que.front();
            que.pop();
            ans = grid[q.first][q.second];
            for (auto d : dir) {
                int new_x = q.first + d[0];
                int new_y = q.second + d[1];
                if (new_x < 0 || new_x >= grid.size() || new_y < 0 || new_y >= grid[0].size() || grid[new_x][new_y] != 0) 
                    continue;
                grid[new_x][new_y] = ans + 1;
                que.push({new_x, new_y});
            }
        }
        return ans - 1;
    }
};
```

<span id="统计优美子数组"></span>
## [1248、统计「优美子数组」(medium)](#back)
```cpp
给你一个整数数组 nums 和一个整数 k。
如果某个 连续 子数组中恰好有 k 个奇数数字，我们就认为这个子数组是「优美子数组」。
请返回这个数组中「优美子数组」的数目。

输入：nums = [1,1,2,1,1], k = 3
输出：2
解释：包含 3 个奇数的子数组是 [1,1,2,1] 和 [1,2,1,1] 。

输入：nums = [2,4,6], k = 1
输出：0
解释：数列中不包含任何奇数，所以不存在优美子数组。

输入：nums = [2,2,2,1,2,2,1,2,2,2], k = 2
输出：16

class Solution {
public:
    int numberOfSubarrays(vector<int>& nums, int k) {
        // 参考链接：https://leetcode-cn.com/problems/count-number-of-nice-subarrays/solution/tong-ji-you-mei-zi-shu-zu-by-leetcode-solution/
        int n = nums.size();
        vector<int> res(n + 1, 0);
        res[0] = 1;
        int ans = 0, pre_i = 0;
        for (int i = 0; i < n; i++) {
            pre_i += nums[i] & 1;
            ans += pre_i >= k ? res[pre_i - k] : 0;
            res[pre_i] += 1;
        }
        return ans;
    }
};
```

<span id="字符串压缩"></span>
## [1394、字符串压缩(easy)](#back)
```cpp
字符串压缩。利用字符重复出现的次数，编写一种方法，实现基本的字符串压缩功能。
比如，字符串aabcccccaaa会变为a2b1c5a3。
若“压缩”后的字符串没有变短，则返回原先的字符串。
你可以假设字符串中只包含大小写英文字母（a至z）。

 输入："aabcccccaaa"
 输出："a2b1c5a3"

 输入："abbccd"
 输出："abbccd"
 解释："abbccd"压缩后为"a1b2c2d1"，比原字符串长度更长。

class Solution {
public:
    string compressString(string S) {
        char c_ = S[0];
        int count_ = 1;
        string new_s;
        for (int i = 1; i < S.size(); i++) {
            if (c_ == S[i]) count_++;
            else {
                new_s += c_ + to_string(count_);  
                c_ = S[i];
                count_ = 1;
            }
        }
        new_s += c_ + to_string(count_);
        return new_s.size() < S.size() ? new_s : S;
    }
};
```

<span id="旋转矩阵"></span>
## [1418、旋转矩阵(medium)](#back)
```cpp
给你一幅由 N × N 矩阵表示的图像，其中每个像素的大小为 4 字节。
请你设计一种算法，将图像旋转 90 度。
不占用额外内存空间能否做到？

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
        int N = matrix.size();
        int left = 0, right = N - 1, top = 0, bottom = N - 1;
        while (left <= right && top <= bottom) {
            if (top <= bottom) {
                for (int i = top; i < bottom; i++) {
                    swap(matrix[i][left], matrix[bottom][i]);
                }
                int bottom_ = bottom;
                for (int i = left; i < right; i++) {
                    swap(matrix[bottom][i], matrix[bottom_--][right]);
                }
                for (int i = bottom; i > top; i--) {
                    swap(matrix[i][right], matrix[top][i]);
                }
            }
            top++,bottom--,left++,right--;
        }
    }
};
```

<span id="交点"></span>
## [1476、交点(hard)](#back)
```cpp
给定两条线段（表示为起点start = {X1, Y1}和终点end = {X2, Y2}），如果它们有交点，请计算其交点，没有交点则返回空值。

要求浮点型误差不超过10^-6。若有多个交点（线段重叠）则返回 X 值最小的点，X 坐标相同则返回 Y 值最小的点。

输入：
line1 = {0, 0}, {1, 0}
line2 = {1, 1}, {0, -1}
输出： {0.5, 0}

输入：
line1 = {0, 0}, {3, 3}
line2 = {1, 1}, {2, 2}
输出： {1, 1}

输入：
line1 = {0, 0}, {1, 1}
line2 = {1, 0}, {2, 1}
输出： {}，两条线段没有交点

class Solution {
public:
    // 判断 (xk, yk) 是否在「线段」(x1, y1)~(x2, y2) 上
    // 这里的前提是 (xk, yk) 一定在「直线」(x1, y1)~(x2, y2) 上
    bool inside(int x1, int y1, int x2, int y2, int xk, int yk) {
        // 若与 x 轴平行，只需要判断 x 的部分
        // 若与 y 轴平行，只需要判断 y 的部分
        // 若为普通线段，则都要判断
        return (x1 == x2 || (min(x1, x2) <= xk && xk <= max(x1, x2))) && (y1 == y2 || (min(y1, y2) <= yk && yk <= max(y1, y2)));
    }

    void update(vector<double>& ans, double xk, double yk) {
        // 将一个交点与当前 ans 中的结果进行比较
        // 若更优则替换
        if (!ans.size() || xk < ans[0] || (xk == ans[0] && yk < ans[1])) {
            ans = {xk, yk};
        }
    }

    vector<double> intersection(vector<int>& start1, vector<int>& end1, vector<int>& start2, vector<int>& end2) {
        // 参考链接：https://leetcode-cn.com/problems/intersection-lcci/solution/jiao-dian-by-leetcode-solution/
        int x1 = start1[0], y1 = start1[1];
        int x2 = end1[0], y2 = end1[1];
        int x3 = start2[0], y3 = start2[1];
        int x4 = end2[0], y4 = end2[1];

        vector<double> ans;
        // 判断 (x1, y1)~(x2, y2) 和 (x3, y3)~(x4, y3) 是否平行
        if ((y4 - y3) * (x2 - x1) == (y2 - y1) * (x4 - x3)) {
            // 若平行，则判断 (x3, y3) 是否在「直线」(x1, y1)~(x2, y2) 上
            if ((y2 - y1) * (x3 - x1) == (y3 - y1) * (x2 - x1)) {
                // 判断 (x3, y3) 是否在「线段」(x1, y1)~(x2, y2) 上
                if (inside(x1, y1, x2, y2, x3, y3)) {
                    update(ans, (double)x3, (double)y3);
                }
                // 判断 (x4, y4) 是否在「线段」(x1, y1)~(x2, y2) 上
                if (inside(x1, y1, x2, y2, x4, y4)) {
                    update(ans, (double)x4, (double)y4);
                }
                // 判断 (x1, y1) 是否在「线段」(x3, y3)~(x4, y4) 上
                if (inside(x3, y3, x4, y4, x1, y1)) {
                    update(ans, (double)x1, (double)y1);
                }
                // 判断 (x2, y2) 是否在「线段」(x3, y3)~(x4, y4) 上
                if (inside(x3, y3, x4, y4, x2, y2)) {
                    update(ans, (double)x2, (double)y2);
                }
            }
            // 在平行时，其余的所有情况都不会有交点
        }
        else {
            // 联立方程得到 t1 和 t2 的值
            double t1 = (double)(x3 * (y4 - y3) + y1 * (x4 - x3) - y3 * (x4 - x3) - x1 * (y4 - y3)) / ((x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1));
            double t2 = (double)(x1 * (y2 - y1) + y3 * (x2 - x1) - y1 * (x2 - x1) - x3 * (y2 - y1)) / ((x4 - x3) * (y2 - y1) - (x2 - x1) * (y4 - y3));
            // 判断 t1 和 t2 是否均在 [0, 1] 之间
            if (t1 >= 0.0 && t1 <= 1.0 && t2 >= 0.0 && t2 <= 1.0) {
                ans = {x1 + t1 * (x2 - x1), y1 + t1 * (y2 - y1)};
            }
        }
        return ans;
    }
};
```

<span id="硬币"></span>
## [1481、硬币(medium)](#back)
```cpp
硬币。给定数量不限的硬币，币值为25分、10分、5分和1分，编写代码计算n分有几种表示法。
(结果可能会很大，你需要将结果模上1000000007)

输入: n = 5
输出：2
解释: 有两种方式可以凑成总金额:
5=5
5=1+1+1+1+1

输入: n = 10
输出：4
解释: 有四种方式可以凑成总金额:
10=10
10=5+5
10=5+1+1+1+1+1
10=1+1+1+1+1+1+1+1+1+1

class Solution {
public:
    int waysToChange(int n) {
        // 动态规划
        // dp[i][j]：使用前 i 种硬币凑成和为 j 的方法数
        int vec[4] = {1, 5, 10, 25};
        vector<vector<int>> dp(4, vector<int> (n + 1, 0));
        for (int i = 0; i < 4; i++) {
            dp[i][0] = 1;
        }
        for (int j = 0; j < n + 1; j++) {
            if (j % vec[0] == 0) {
                dp[0][j] = 1;
            }
        }
        for(int i = 1; i < 4; i++) {
            for (int j = 1; j < n + 1; j++) {
                if (j - vec[i] >= 0) {
                    dp[i][j] = (dp[i][j - vec[i]] + dp[i - 1][j]) % 1000000007;
                }
                else {
                    dp[i][j] = dp[i - 1][j]  % 1000000007;
                }
            }
        }
        return dp[3][n];
    }
};
```

<span id="按摩师"></span>
## [1496、按摩师(easy)](#back)
```cpp
一个有名的按摩师会收到源源不断的预约请求，每个预约都可以选择接或不接。
在每次预约服务之间要有休息时间，因此她不能接受相邻的预约。
给定一个预约请求序列，替按摩师找到最优的预约集合（总预约时间最长），返回总的分钟数。
注意：本题相对原题稍作改动,可以相隔2个预约

输入： [1,2,3,1]
输出： 4
解释： 选择 1 号预约和 3 号预约，总时长 = 1 + 3 = 4。

输入： [2,7,9,3,1]
输出： 12
解释： 选择 1 号预约、 3 号预约和 5 号预约，总时长 = 2 + 9 + 1 = 12。

输入： [2,1,4,5,3,1,1,3]
输出： 12
解释： 选择 1 号预约、 3 号预约、 5 号预约和 8 号预约，总时长 = 2 + 4 + 3 + 3 = 12。

class Solution {
public:
    int massage(vector<int>& nums) {
        // 动态规划
        // dp[i][0]: 第 i 次不预约，前 i 中最大预约时间 
        // 转移方程：dp[i][0] = max(dp[i - 1][0], dp[i - 1][1])
        // dp[i][1]: 第 i 次预约，前 i 中最大预约时间 
        // 转移方程：dp[i][1] = dp[i - 1][0] + nums[i]
        if (nums.size() <= 0) return 0;
        int dp0 = 0, dp1 = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            int ndp0 = max(dp0, dp1);  // 计算dp[i][0]
            int ndp1 = dp0 + nums[i];  // 计算dp[i][1]
            dp0 = ndp0;  // dp[i][0]更新dp0
            dp1 = ndp1;  // dp[i][1]更新dp1
        }
        return max(dp0, dp1);
    }
};
```

<span id="机器人的运动范围"></span>
## [1531、机器人的运动范围(medium)](#back)
```cpp
地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。
一个机器人从坐标 [0, 0] 的格子开始移动，
它每次可以向左、右、上、下移动一格（不能移动到方格外），
也不能进入行坐标和列坐标的数位之和大于k的格子。
例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。
但它不能进入方格 [35, 38]，因为3+5+3+8=19。
请问该机器人能够到达多少个格子？

输入：m = 2, n = 3, k = 1
输出：3

输入：m = 3, n = 1, k = 0
输出：1

class Solution {
public:
    // 判断机器人是否可以进入位置 (a, b)
    bool torf(int a, int b, int k) {
        int count_ = 0;
        while (a) {
            count_ += a % 10;
            a = a / 10;
        }
        while (b) {
            count_ += b % 10;
            b = b / 10;
        }
        return count_ <= k ? true : false;
    }

    int sum = 0;
    int dir[4][4] = {{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
    void movingCountHelper(vector<vector<int>> &flag, int i, int j, int m, int n, int k) {
        if (torf(i, j, k) && i < m && j < n && i >= 0 && j >= 0 && flag[i][j] == 0) {
            flag[i][j] = 1;
            sum += 1;
            for(int z = 0; z < 4; z++) {
                int new_i = i + dir[z][0];
                int new_j = j + dir[z][1];
                if (torf(new_i, new_j, k) && new_i < m && new_j < n && new_i >= 0 && new_j >= 0 && flag[new_i][new_j] == 0)
                    movingCountHelper(flag, new_i, new_j, m, n, k);
            }
        }
    }

    int movingCount(int m, int n, int k) {
        vector<vector<int>> flag(m, vector<int> (n, 0));
        movingCountHelper(flag, 0, 0, m, n, k);
        return sum;
    }
};
```

<span id="最小的k个数"></span>
## [1538、最小的k个数(easy)](#back)
```cpp
输入整数数组 arr ，找出其中最小的 k 个数。
例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]

输入：arr = [0,1,2,1], k = 1
输出：[0]

class Solution {
public:
    int partition(vector<int>& nums, int l, int r) {
        int pivot = nums[r];
        int i = l - 1;
        for (int j = l; j < r; j++) {
            if (nums[j] <= pivot) {
                i = i + 1;
                swap(nums[i], nums[j]);
            }
        }
        swap(nums[i + 1], nums[r]);
        return i + 1;
    }

    int randomized_partition(vector<int>& nums, int l,  int r) {
        int i = rand() % (r - l + 1) + l;
        swap(nums[r], nums[i]);
        return partition(nums, l, r);
    }

    void randomized_selected(vector<int>& arr, int l, int r, int k) {
        if (l >= r) return ;
        int pos = randomized_partition(arr, l, r);
        int num = pos - l + 1;
        if (k == num) return ;
        else if (k < num) randomized_selected(arr, l, pos - 1, k);
        else randomized_selected(arr, pos + 1, r, k - num);
    }

    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        // // 最大堆实现
        // // 时间复杂度：nlogk
        // // 空间复杂度： k
        // priority_queue<int> heap;
        // for(auto a : arr) {
        //     heap.push(a);
        //     if (heap.size() > k) {
        //         heap.pop();
        //     }
        // }
        // vector<int> res;
        // for(int i = 0; i < k; i++) {
        //     res.push_back(heap.top());
        //     heap.pop();
        // }
        // return res;

        // 快排实现
        // 时间复杂度：n
        // 空间复杂度：logn
        srand((unsigned)time(NULL));  // 初始化随机种子
        randomized_selected(arr, 0, (int)arr.size() - 1, k);
        vector<int> res;
        for (int i = 0; i < k; i++) {
            res.push_back(arr[i]);
        }
        return res;
    }
};
```

<span id="圆圈中最后剩下的数字"></span>
## [1579、圆圈中最后剩下的数字(easy)](#back)
```cpp
0,1,,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。
求出这个圆圈里剩下的最后一个数字。
例如，0、1、2、3、4这5个数字组成一个圆圈，
从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

输入: n = 5, m = 3
输出: 3

输入: n = 10, m = 17
输出: 2
 
限制：
1 <= n <= 10^5
1 <= m <= 10^6

class Solution {
public:
    int lastRemaining(int n, int m) {
        // 约瑟夫环
        if (n == 0 || m == 0) return -1;
        int s = 0; 
        for (int i = 2; i <= n; i++) {
            s = (s + m) % i;
        }
        return s;
    }
};
```

<span id="数组中的逆序对"></span>
## [1591、数组中的逆序对(hard)](#back)
```cpp
在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。
输入一个数组，求出这个数组中的逆序对的总数。

输入: [7,5,6,4]
输出: 5

class Solution {
public:
    // 归并
    long InversePairsCore(vector<int> &data,vector<int> &copy,int start,int end){
        if(start==end){
            copy[start]=data[start];
            return 0;
        }
        int length=(end-start)/2;
        long left=InversePairsCore(copy,data,start,start+length);
        long right=InversePairsCore(copy,data,start+length+1,end);
        
        int i=start+length;
        int j=end;
        int index_copy=end;
        long count=0;
        while(i>=start && j>=start+length+1){
            if(data[i]>data[j]){
                copy[index_copy--]=data[i--];
                count=count+j-start-length;
            }
            else{
                copy[index_copy--]=data[j--];
            }
        }
        for(;i>=start;i--){
            copy[index_copy--]=data[i];
        }
        for(;j>=start+length+1;j--){
            copy[index_copy--]=data[j];
        }
        return count+left+right;
    }


    int reversePairs(vector<int>& nums) {
        if(nums.size()<1)
            return 0;
        vector<int> copy;
        for(int i=0;i<int(nums.size());i++){
            copy.push_back(nums[i]);
        }
        long count=InversePairsCore(nums,copy,0,int(nums.size())-1);
        return count;
    }
};
```

<span id="数组中数字出现的次数"></span>
## [1608、数组中数字出现的次数(medium)](#back)
```cpp
一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。
请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

输入：nums = [4,1,4,6]
输出：[1,6] 或 [6,1]

输入：nums = [1,2,10,4,1,4,3,3]
输出：[2,10] 或 [10,2]

class Solution {
public:
    vector<int> singleNumbers(vector<int>& nums) {
        int res = 0;
        for (auto n : nums) res ^= n;
        int div = 1;
        while ((res & div) == 0) div <<= 1;
        int a = 0, b = 0;
        for (auto n : nums) {
            if (div & n) a ^= n;
            else b ^= n;
        }
        return vector<int> {a, b};
    }
};
```
