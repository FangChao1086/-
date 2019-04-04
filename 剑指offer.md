* [1、找出数组中重复的数字](#找出数组中重复的数字)
* [2、不修改数组找出重复的数字](#不修改数组找出重复的数字)
* [3、二维数组中的查找](#二维数组中的查找)
* [4、替换空格](#替换空格)
* [5、从尾到头打印链表](#从尾到头打印链表)
* [6、重建二叉树](#重建二叉树)
* [7、二叉树的下一个节点](#二叉树的下一个节点)
* [8、用两个栈实现队列](#用两个栈实现队列)
* [9、斐波那契数列](#斐波那契数列)

<span id="找出数组中重复的数字"></span>
## 找出数组中重复的数字
> **题目**  
给定一个长度为n的整数数组nums，数组中所有的数字都在0∼n−1的范围内。  
数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。  
请找出数组中任意一个重复的数字。  
> **样例**  
给定 nums = [2, 3, 5, 4, 3, 2, 6, 7]。
返回 2 或 3。

**思路**  
* 查看当前位置的值   
  * 假设位置（0）值（4），查看此时值对应的位置（4）上的的值  
      * 如果此时位置（4）上的值为4,则返回4 
      * 否则(若值为1)，交换位置0和位置4上的值，再循环比较位置0和位置1上的值。。。  
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
> **题目**
给定一个长度为n+1的数组nums，数组中所有的数均在1∼n的范围内，其中 n≥1  
请找出数组中任意一个重复的数，但不能修改输入的数组。
> **样例**
给定 nums = [2, 3, 5, 4, 3, 2, 6, 7]。
返回 2 或 3。

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
> **题目**
在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。  
请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
> **样例**
输入数组：
[
  [1,2,8,9]，
  [2,4,9,12]，
  [4,7,10,13]，
  [6,8,11,15]
]
如果输入查找数值为7，则返回true，
如果输入查找数值为5，则返回false。

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
> **题目**
请实现一个函数，把字符串中的每个空格替换成"%20"。
> **样例**
输入："We are happy."
输出："We%20are%20happy."

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
> **题目**
输入一个链表的头结点，按照 从尾到头 的顺序返回节点的值。
返回的结果用数组存储。
**样例**
输入：[2, 3, 5]
返回：[5, 3, 2]

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

<span id="重建二叉树"></span>
## 重建二叉树
> **题目**
输入一棵二叉树前序遍历和中序遍历的结果，请重建该二叉树。
注意:
  * 二叉树中每个节点的值都互不相同；
  * 输入的前序遍历和中序遍历一定合法；
**样例**
给定：
前序遍历是：[3, 9, 20, 15, 7]
中序遍历是：[9, 3, 15, 20, 7]
返回：[3, 9, 20, null, null, 15, 7, null, null, null, null]
返回的二叉树如下所示：
    3
   / \
  9  20
    /  \
   15   7

**思路**
* 1、得到中序遍历的根节点的位置
* 2、得到左子树的前序与中序遍历的结果
* 3、得到右子树的前序与中序遍历的结果
* 4、递归得到整个二叉树
 
**代码**
```C++
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
    TreeNode* buildTree(vector<int>& pre, vector<int>& vin) {
        int vinlen=vin.size();
        if(vinlen==0) return NULL;
        vector<int> left_pre,left_vin,right_pre,right_vin;
        TreeNode* head=new TreeNode(pre[0]);
        
        int root_index=0;
        for(int i=0;i<vinlen;i++){
            if(vin[i]==pre[0]){
                root_index=i;
                break;
            }
        }
        
        for(int i=0;i<root_index;i++){
            left_pre.push_back(pre[i+1]);
            left_vin.push_back(vin[i]);
        }
        
        for(int i=root_index+1;i<vinlen;i++){
            right_pre.push_back(pre[i]);
            right_vin.push_back(vin[i]);
        }
        
        head->left=buildTree(left_pre,left_vin);
        head->right=buildTree(right_pre,right_vin);
        
        return head;
    }
};
```

<span id="二叉树的下一个节点"></span>
## 二叉树的下一个节点
> **题目**
给定一棵二叉树的其中一个节点，请找出中序遍历序列的下一个节点。
注意：
  * 如果给定的节点是中序遍历序列的最后一个，则返回空节点;
  * 二叉树一定不为空，且给定的节点一定不是空节点；
**样例**
假定二叉树是：[2, 1, 3, null, null, null, null]， 给出的是值等于2的节点。
则应返回值等于3的节点。
解释：该二叉树的结构如下，2的后继节点是3。
  2
 / \
1   3

**思路**
* 分情况讨论
* 有右子树
* 无右子树

**代码**
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode *father;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL), father(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* inorderSuccessor(TreeNode* p) {
        if(p->right){
            p=p->right;
            while(p->left) p=p->left;
            return p;
        }
        while(p->father && p==p->father->right) p=p->father;
        return p->father;
    }
};
```

<span id="用两个栈实现队列"></span>
## 用两个栈实现队列
> **题目**
请用栈实现一个队列，支持如下四种操作：
  * push(x) – 将元素x插到队尾；
  * pop() – 将队首的元素弹出，并返回该元素；
  * peek() – 返回队首元素；
  * empty() – 返回队列是否为空；
注意：
  * 你只能使用栈的标准操作：push to top，peek/pop from top, size 和 is empty；
  * 如果你选择的编程语言没有栈的标准库，你可以使用list或者deque等模拟栈的操作；
  * 输入数据保证合法，例如，在队列为空时，不会进行pop或者peek等操作；
**样例**
MyQueue queue = new MyQueue();
queue.push(1);
queue.push(2);
queue.peek();  // returns 1
queue.pop();   // returns 1
queue.empty(); // returns false

**思路**
* 两个栈实现，暴力解

**代码**
```c++
class MyQueue {
public:
    /** Initialize your data structure here. */
    stack<int> stk,cache;
    MyQueue() {
        
    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        stk.push(x);
    }
    
    void copy(stack<int> &a, stack<int> &b){
        while(a.size()){
            b.push(a.top());
            a.pop();
        }
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        copy(stk,cache);
        int res=cache.top();
        cache.pop();
        copy(cache,stk);
        return res;
    }
    
    /** Get the front element. */
    int peek() {
        copy(stk,cache);
        int res=cache.top();
        copy(cache,stk);
        return res;
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        return stk.empty();
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = MyQueue();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.peek();
 * bool param_4 = obj.empty();
 */
```

<span id="斐波那契数列"></span>
## 斐波那契数列
> **题目**
输入一个整数n，求斐波那契数列的第n项。
假定从0开始，第0项为0。(n<=39)
**样例**
输入整数 n=5 
返回 5

**思路**

**代码**
```c++
class Solution {
public:
    int Fibonacci(int n) {
        int a=0,b=1;
        while(n--){
            int c=a+b;
            a=b,b=c;
        }
        return a;
    }
};
```
