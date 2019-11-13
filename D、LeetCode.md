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
[13、罗马数字转整数](#罗马数字转整数)  
[14、最长公共前缀](#最长公共前缀)  
[15、三数之和](#三数之和)  
[16、最接近的三数之和](#最接近的三数之和)  
[17、电话号码的字母组合](#电话号码的字母组合)  
[18、四数之和](#四数之和)  
[19、删除链表的倒数第N个节点](#删除链表的倒数第N个节点)  
[20、有效的括号](#有效的括号)  
[21、合并两个有序链表](#合并两个有序链表)  
[22、括号生成](#括号生成)  
[23、合并K个排序链表](#合并K个排序链表)  
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

<span id="罗马数字转整数"></span>
## [13、罗马数字转整数](#re_)
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
## [15、三数之和](#re_)
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
## [16、最接近的三数之和](#re_)
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
## [17、电话号码的字母组合](#re_)
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
## [18、四数之和](#re_)
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
## [19、删除链表的倒数第N个节点](#re_)
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
## [20、有效的括号](#re_)
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
## [21、合并两个有序链表](#re_)
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
## [22、括号生成](#re_)
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
## [23、合并K个排序链表](#re_)
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
