* [1、找出数组中重复的数字](#找出数组中重复的数字)
* [2、不修改数组找出重复的数字](#不修改数组找出重复的数字)
* [3、二维数组中的查找](#二维数组中的查找)
* [4、替换空格](#替换空格)
* [5、从尾到头打印链表](#从尾到头打印链表)

<span id="找出数组中重复的数字"></span>
## 找出数组中重复的数字
**题目**
```
给定一个长度为n的整数数组nums，数组中所有的数字都在0∼n−1的范围内。  
数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。  
请找出数组中任意一个重复的数字。  
```
**样例**
```
给定 nums = [2, 3, 5, 4, 3, 2, 6, 7]。
返回 2 或 3。
```
**思路**
* 查看当前位置的值 
  * 假设位置（0）值（4），查看此时值对应的位置（4）上的的值
    * 如果此时位置（4）上的值为4,则返回4；
    * 否则交换位置0和位置4上的值；
* 说明：
  * 时间复杂度：n
  * 空间复杂度：1  
  
**代码**
```C++
class Solution {
public:
    int duplicateInArray(vector<int>& nums) {
        int n=nums.size();
        for(auto x:nums)
            if(x<0 || x>=n) return -1;
        for(int i=0;i<n;i++){
            while(i!=nums[i] && nums[nums[i]]!=nums[i]) swap(nums[i],nums[nums[i]]);
            if(nums[i]!=i && nums[nums[i]]==nums[i]) return nums[i];
        }
        return -1;
    }
};
```

<span id="不修改数组找出重复的数字"></span>
## 不修改数组找出重复的数字
**题目**
```
给定一个长度为n+1的数组nums，数组中所有的数均在1∼n的范围内，其中 n≥1  
请找出数组中任意一个重复的数，但不能修改输入的数组。
```
**样例**
```
给定 nums = [2, 3, 5, 4, 3, 2, 6, 7]。
返回 2 或 3。
```
**思路**
* 用数值大小将其划分成两份 
  * 数组中n+1个数
  * 划分后左边+右边=n,即左边或者右边必有一边有重复数字
    * 每次遍历若值在左边，则左边计数值加一
    * 当计数值大于左边边界范围，则重复值在左边，否则在右边
  
**代码**
```C++
class Solution {
public:
    int duplicateInArray(vector<int>& nums) {
        int l =1, r =nums.size()-1;
        while(l<r){
            int mid = l+r >> 1; // [l,mid] [mid+1,r]
            int s=0;
            for(auto x:nums) s+=x>=l && x<=mid;
            if(s>mid-l+1) r=mid;
            else l=mid+1;
        }
        return r;
    }
};
```

<span id="二维数组中的查找"></span>
## 二维数组中的查找
**题目**
```
在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。  
请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
```
**样例**
```
输入数组：

[
  [1,2,8,9]，
  [2,4,9,12]，
  [4,7,10,13]，
  [6,8,11,15]
]

如果输入查找数值为7，则返回true，

如果输入查找数值为5，则返回false。
```
**思路**
* 从右上开始查找
  * 若当前值小于目标值，则向下查找
  * 若当前值大于目标值，则向左查找
  * 若等于目标值，则找到

  
**代码**
```C++
class Solution {
public:
    bool searchArray(vector<vector<int>> array, int target) {
        if(array.empty() || array[0].empty()) return false;
        int i=0,j=array[0].size()-1,n=array.size();
        while(i<n && j>=0){
            if(array[i][j]==target) return true;
            else if(array[i][j]>target) j--;
            else i++;
        }
        return false;
    }
};
```

<span id="替换空格"></span>
## 替换空格
**题目**
```
请实现一个函数，把字符串中的每个空格替换成"%20"。
```
**样例**
```
输入："We are happy."

输出："We%20are%20happy."
```
**思路**
* 遍历
  * 为空格时，替换为%20
  
**代码**
```C++
class Solution {
public:
    string replaceSpaces(string &str) {
        string res;
        for(auto x:str){
            if(x==' ')
                res+="%20";
            else
                res+=x;
        }
        return res;
    }
};
```

<span id="从尾到头打印链表"></span>
## 从尾到头打印链表
**题目**
```
输入一个链表的头结点，按照 从尾到头 的顺序返回节点的值。
返回的结果用数组存储。
```
**样例**
```
输入：[2, 3, 5]
返回：[5, 3, 2]
```
**思路**
* 遍历节点，存值
* 反转打印
  
**代码**
```C++
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
    vector<int> printListReversingly(ListNode* head) {
        vector<int> res;
        while(head){
            res.push_back(head->val);
            head=head->next;
        }
        return vector<int>(res.rbegin(),res.rend());
    }
};
```
