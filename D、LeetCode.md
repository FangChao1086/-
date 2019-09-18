<span id="re_"></span>
# LeetCode

[5、最长回文子串](#最长回文子串)  
[69、X的平方根](#X的平方根)   
[386、字典序排数](#字典序排数)

<span id="最长回文子串"></span>
## [5、最长回文子串](#re_)
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
            int mid = left + (right - left) / 2;
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
