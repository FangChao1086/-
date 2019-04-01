* [1、找出数组中重复的数字](#找出数组中重复的数字)

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
---
