# 剑指offer_CPP版题目及答案
[1、找出数组中重复的数字](#找出数组中重复的数字)  
[2、不修改数组找出重复的数字](#不修改数组找出重复的数字)  
[3、二维数组中的查找](#二维数组中的查找)  
[4、替换空格](#替换空格)  
[5、从尾到头打印链表](#从尾到头打印链表)  
[6、（ * ）**重建二叉树**](#重建二叉树)  
[7、二叉树的下一个节点](#二叉树的下一个节点)  
[8、用两个栈实现队列](#用两个栈实现队列)  
[9、斐波那契数列、跳台阶、矩形覆盖](#斐波那契数列)  
[10、旋转数组的最小数字](#旋转数组的最小数字)  
[11、（ * ）**二进制中1的个数**](#二进制中1的个数)  
[12、数值的整数次方](#数值的整数次方)  
[13、调整数组顺序使奇数位于偶数前面](#调整数组顺序使奇数位于偶数前面)  
[14、链表中倒数第K个节点](#链表中倒数第K个节点)  
[15、反转链表](#反转链表)  
[16、合并两个排序的链表](#合并两个排序的链表)  
[17、树的子结构](#树的子结构)  
[18、二叉树的镜像](#二叉树的镜像)  
[19、顺时针打印矩阵](#顺时针打印矩阵)  
[20、包含min函数的栈](#包含min函数的栈)  
[21、栈的压入、弹出序列](#栈的压入、弹出序列)  
[22、从上往下打印二叉树](#从上往下打印二叉树)  
[23、二叉搜索树的后序遍历序列](#二叉搜索树的后序遍历序列)  
[24、二叉树中和为某一个值的路径](#二叉树中和为某一个值的路径)  
[25、复杂链表的复制](#复杂链表的复制)  
[26、二叉搜索树与双向链表](#二叉搜索树与双向链表)  
[27、字符串的排列](#字符串的排列)  
[28、数组中出现次数超过一半的数字](#数组中出现次数超过一半的数字)  
[29、最小的K个数](#最小的K个数)  
[30、连续子数组的最大和](#连续子数组的最大和)  
[31、整数中1出现的次数（从1到n整数中1出现的次数）](#整数中1出现的次数（从1到n整数中1出现的次数）)  
[32、把整数排成最小的数](#把整数排成最小的数)  
[33、丑数](#丑数)  
[34、第一次只出现一次的字符](#第一次只出现一次的字符)  
[35、数组中的逆序对](#数组中的逆序对)  
[36、两个链表的第一个公共结点](#两个链表的第一个公共结点)  
[37、数字在排序数组中出现的次数](#数字在排序数组中出现的次数)  
[38、二叉树的深度](#二叉树的深度)  
[39、平衡二叉树](#平衡二叉树)  
[40、数组中只出现一次的数字](#数组中只出现一次的数字)  
[41、和为S的连续正数序列](#和为S的连续正数序列)  
[42、和为S的两个数字](#和为S的两个数字)  
[43、左旋转字符串](#左旋转字符串)  
[44、翻转单词顺序序列](#翻转单词顺序序列)  
[45、扑克牌顺子](#扑克牌顺子)  
[46、孩子们的游戏(圆圈中最后剩下的数)](#孩子们的游戏(圆圈中最后剩下的数))  
[47、求1+2+3+...+n](#求1+2+3+...+n)  
[48、不用加减乘除做加法](#不用加减乘除做加法)  
[49、把字符串转换成整数](#把字符串转换成整数)  
[50、数组中重复的数字](#数组中重复的数字)  
[51、构建乘积数组](#构建乘积数组)  
[52、正则表达式匹配](#正则表达式匹配)  
[53、表示数值的字符串](#表示数值的字符串)  
[54、字符流中第一个不重复的字符](#字符流中第一个不重复的字符)  
[55、链表中环的入口结点](#链表中环的入口结点)  
[56、删除链表中重复的结点](#删除链表中重复的结点)  
[57、二叉树的下一个结点](#二叉树的下一个结点)  
[58、对称的二叉树](#对称的二叉树)  
[59、按之字形顺序打印二叉树](#按之字形顺序打印二叉树)  
[60、把二叉树打印成多行](#把二叉树打印成多行)  
[61、序列化二叉树](#序列化二叉树)  
[62、二叉搜索树的第k个结点](#二叉搜索树的第k个结点)  
[63、数据流中的中位数](#数据流中的中位数)  
[64、滑动窗口的最大值](#滑动窗口的最大值)  


[65、矩阵中的路径](#矩阵中的路径)  
[66、机器人的运动范围](#机器人的运动范围)  

* [在O(1)时间删除链表节点](#在O(1)时间删除链表节点)
* [删除链表中重复的节点](#删除链表中重复的节点)
* [正则表达式匹配](#正则表达式匹配)
* [表示数值的字符串](#表示数值的字符串)
* [链表中环的入口节点](#链表中环的入口节点)
* [二叉树的镜像](#二叉树的镜像)
* [栈的压入、弹出序列](#栈的压入、弹出序列)
* [二叉搜索树的后序遍历序列](#二叉搜索树的后序遍历序列)
* [二叉树中和为某一值的路径](#二叉树中和为某一值的路径)
* [复杂链表的复制](#复杂链表的复制)
* [二叉搜索树与双向链表](#二叉搜索树与双向链表)
* [最小的K个数](#最小的K个数)

<span id="找出数组中重复的数字"></span>
## 1、找出数组中重复的数字
```
题目：
给定一个长度为n的整数数组nums，数组中所有的数字都在0∼n−1的范围内。  
数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。  
请找出数组中任意一个重复的数字。  

样例：
给定 nums = [2, 3, 5, 4, 3, 2, 6, 7]。
返回 2 或 3。

思路：
* 查看当前位置的值   
  * 假设位置（0）值（4），查看此时值对应的位置（4）上的的值  
      * 如果此时位置（4）上的值为4,则返回4 
      * 否则(若值为1)，交换位置0和位置4上的值，再循环比较位置0和位置1上的值。。。  
* 说明：
    * 时间复杂度：n
    * 空间复杂度：1 
```  
```cpp
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
## 2、不修改数组找出重复的数字
```
题目：
给定一个长度为n+1的数组nums，数组中所有的数均在1∼n的范围内，其中 n≥1  
请找出数组中任意一个重复的数，但不能修改输入的数组。  

样例：
给定 nums = [2, 3, 5, 4, 3, 2, 6, 7]。  
返回 2 或 3。

思路：
* 用数值大小将其划分成两份 
  * 数组中n+1个数
  * 划分后左边+右边=n,即左边或者右边必有一边有重复数字
    * 每次遍历若值在左边，则左边计数值加一
    * 当计数值大于左边边界范围，则重复值在左边，否则在右边
```  
```cpp
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
## 3、二维数组中的查找
```
题目：
在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。  
请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

样例：
输入数组：  
[  
  [1,2,8,9]，  
  [2,4,9,12]，  
  [4,7,10,13]，  
  [6,8,11,15]  
]  
如果输入查找数值为7，则返回true，  
如果输入查找数值为5，则返回false。  

思路：
* 从右上开始查找
  * 若当前值小于目标值，则向下查找
  * 若当前值大于目标值，则向左查找
  * 若等于目标值，则找到
``` 
```cpp
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        int row = array.size();
        int col = array[0].size();
        int i = 0, j = col-1;
        while(i < row && j > -1){
            if(array[i][j] > target){
                j--;
            }
            else if(array[i][j] < target){
                i++;
            }
            else
                return true;
        }
        return false;
    }
};
```

<span id="替换空格"></span>
## 4、替换空格
```
题目：  
请实现一个函数，把字符串中的每个空格替换成"%20"。  

样例： 
输入："We are happy."  
输出："We%20are%20happy."  

思路：
* 遍历
  * 为空格时，替换为%20
``` 
```cpp
// 方法1 有返回值
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

// 方法2 无返回值
class Solution {
public:
	void replaceSpace(char *str,int length) {
        int count = 0;
        for(int i = 0; i < length; i++){
            if(str[i] == ' ')
                count++;
        }
        int len = length + 2 * count - 1;
        for(int i = len, j = length-1; j > -1; j--){
            if(str[j] == ' '){
                str[i--] = '0';
                str[i--] = '2';
                str[i--] = '%';
            }
            else
                str[i--] = str[j];
        }
	}
};
```

<span id="从尾到头打印链表"></span>
## 5、从尾到头打印链表
```
题目：
输入一个链表的头结点，按照 从尾到头 的顺序返回节点的值。  
返回的结果用数组存储。

样例：
输入：[2, 3, 5]  
返回：[5, 3, 2]  

思路：
* 遍历节点，存值
* 反转打印
```  
```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
// 方法1 先读值，后反向
class Solution {
public:
    vector<int> printListReversingly(ListNode* head) {
        vector<int> res;
        while(head){
            res.push_back(head->val);
            head = head -> next;
        }
        return vector<int>(res.rbegin(), res.rend());
    }
};

// 方法2 先反向，后读值
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        ListNode* pNew = head;
        ListNode* pPre = NULL;
        vector<int> vec;
        while(pNew){
            ListNode* pNode = pNew -> next;
            pNew -> next = pPre;
            pPre = pNew;
            pNew = pNode;
        }
        while(pPre){
            vec.push_back(pPre -> val);
            pPre = pPre -> next;
        }
        return vec;
    }
};
```

<span id="重建二叉树"></span>
## 6、（ * ）**重建二叉树**
```
题目：
输入一棵二叉树前序遍历和中序遍历的结果，请重建该二叉树。  
注意:  
  * 二叉树中每个节点的值都互不相同；  
  * 输入的前序遍历和中序遍历一定合法； 
  
样例：  
给定：前序遍历是：[3, 9, 20, 15, 7]  
     中序遍历是：[9, 3, 15, 20, 7]  
返回：[3, 9, 20, null, null, 15, 7, null, null, null, null]  
返回的二叉树如下所示：  
    3  
   / \  
  9  20  
    /  \  
   15   7  

思路：
* 1、得到中序遍历的根节点的位置
* 2、得到左子树的前序与中序遍历的结果
* 3、得到右子树的前序与中序遍历的结果
* 4、递归得到整个二叉树
``` 
```cpp
/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        if(vin.size()==0) return NULL;
        vector<int> leftPre, leftVin, rightPre, rightVin;
        TreeNode* head = new TreeNode(pre[0]);
        int count = 0;
        for(int i = 0; i < vin.size(); i++){
            if(vin[i] == pre[0]){
                count = i;
                break;
            }
        }
        for(int i = 0; i < count; i++){
            leftPre.push_back(pre[i+1]);
            leftVin.push_back(vin[i]);
        }
        for(int i = count + 1; i < vin.size(); i++){
            rightPre.push_back(pre[i]);
            rightVin.push_back(vin[i]);
        }
        head->left = reConstructBinaryTree(leftPre, leftVin);
        head->right = reConstructBinaryTree(rightPre, rightVin);
        return head;
    }
};
```

<span id="二叉树的下一个节点"></span>
## 7、二叉树的下一个节点
```
题目:  
给定一棵二叉树的其中一个节点，请找出中序遍历序列的下一个节点。  
注意：  
  * 如果给定的节点是中序遍历序列的最后一个，则返回空节点;  
  * 二叉树一定不为空，且给定的节点一定不是空节点；  
  
样例:
假定二叉树是：[2, 1, 3, null, null, null, null]， 给出的是值等于2的节点。  
则应返回值等于3的节点。  
解释：该二叉树的结构如下，2的后继节点是3。  
  2  
 / \  
1   3  

思路:
* 分情况讨论
* 有右子树
* 无右子树
```
```cpp
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
## 8、用两个栈实现队列
```
题目:  
请用栈实现一个队列，支持如下四种操作：  
  * push(x) – 将元素x插到队尾；  
  * pop() – 将队首的元素弹出，并返回该元素；  
  * peek() – 返回队首元素；  
  * empty() – 返回队列是否为空；  
注意：  
  * 你只能使用栈的标准操作：push to top，peek/pop from top, size 和 is empty；  
  * 如果你选择的编程语言没有栈的标准库，你可以使用list或者deque等模拟栈的操作；  
  * 输入数据保证合法，例如，在队列为空时，不会进行pop或者peek等操作；
  
样例:
MyQueue queue = new MyQueue();  
queue.push(1);  
queue.push(2);  
queue.peek();  // returns 1  
queue.pop();   // returns 1  
queue.empty(); // returns false  

思路:
* 两个栈实现，暴力解
```
```cpp
// 方法1
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
 
// 方法2
class Solution
{
public:
    void push(int node) {
        stack1.push(node);
    }

    int pop() {
        if(stack2.empty()){
            while(!stack1.empty()){
                stack2.push(stack1.top());
                stack1.pop();
            }
        }
        int a = stack2.top();
        stack2.pop();
        return a;
    }

private:
    stack<int> stack1;
    stack<int> stack2;
};
```

<span id="斐波那契数列"></span>
## 9、斐波那契数列、跳台阶、矩形覆盖
```
题目:
输入一个整数n，求斐波那契数列的第n项。  
假定从0开始，第0项为0。(n<=39) 

样例:
输入整数 n=5   
返回 5  

思路:
```
```cpp
class Solution {
public:
    int Fibonacci(int n) {
        int f1 = 0, f2 = 1;
        while(n--){
            f2 = f1 + f2;  // 和
            f1 = f2 - f1;  // 单个数
        }
        return f1;
    }
};

// 初级跳台阶：可跳1，2级台阶；初始化：f1 = 1, f2 = 1；同上
// 高级条台阶：可跳1，2，n级；
class Solution {
public:
    int jumpFloorII(int number) {
        int s = 1;
        for(int i = 1; i < number; i++){
            s *= 2;
        }
        return s;
    }
};

// 矩形覆盖  用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法
class Solution {
public:
    int rectCover(int number) {
        int s1 = 1, s2 = 1;
        if(number <= 0) return 0;
        while(number--){
            s2 = s1 + s2;
            s1 = s2 - s1;
        }
        return s1;
    }
};
```
<span id="旋转数组的最小数字"></span>
## 10、旋转数组的最小数字
```
题目：
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个升序的数组的一个旋转，输出旋转数组的最小元素。
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 
数组可能包含重复项。
注意：数组内所含元素非负，若数组大小为0，请返回-1。

样例：
输入：nums=[2,2,2,0,1]
输出：0

思路：
去除边界（不合法与重复值）后，使用二分查找
```
```cpp
// 方法1
class Solution{
public:
    int findMin(vector<int>& nums){
        int n=nums.size()-1;
        if(n<0) return -1;
        while(n>0 && nums[n]==nums[0]) n--;
        if(nums[n]>=nums[0]) return nums[0];
        int l=0,r=n;
        while(l<r){
            int mid=l+r>>1;
            if(nums[mid]<nums[0]) r=mid;
            else l=mid+1;
        }
        return nums[r];
    }
};

// 方法2  好理解，推荐使用
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        int len = rotateArray.size();
        int left = 0,mid = 0, right = len - 1;
        while(left < right){
            if(right - left <= 1) return rotateArray[right];
            mid = (left + right) >> 1;
            if(rotateArray[left] <= rotateArray[mid]){
                left = mid;
            }
            else if(rotateArray[mid] <= rotateArray[right]){
                right = mid;
            }
        }
    }
};
```

<span id="二进制中1的个数"></span>
## 11、（ * ）**二进制中1的个数**
```
题目：
输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
```
```cpp
class Solution {
public:
    //最优解， 从右往左计数
     int  NumberOf1(int n) {
         int count = 0;
         while(n){
             count++;
             n &= (n - 1);
         }
         return count;
     }
};
```

<span id="数值的整数次方"></span>
## 12、数值的整数次方
```
给定一个double类型的浮点数base和int类型的整数exponent。
求base的exponent次方。
```
```cpp
class Solution {
public:
    double Power(double base, int exponent) {
        int E = abs(exponent);
        double s = 1;
        while(E){
            if(E & 1){ // E % 2 == 1
                s *= base;
            }
            base *= base;
            E >>= 1;  // E /= 2;
        }
        return exponent >= 0 ? s : 1 / s;
    }
};
```

<span id="调整数组顺序使奇数位于偶数前面"></span>
## 13、调整数组顺序使奇数位于偶数前面
```
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，
并保证奇数和奇数，偶数和偶数之间的相对位置不变。
```
```cpp
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        for(int i = 0; i < array.size(); i++)
            for(int j = 0; j < array.size() - 1; j++)
                if(array[j] % 2 == 0 && array[j + 1] % 2 == 1)
                    swap(array[j], array[j + 1]);
    }
};
```

<span id="链表中倒数第K个节点"></span>
## 14、链表中倒数第K个节点
```
输入一个链表，输出该链表中倒数第k个结点。
```
```cpp
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        ListNode* p1 = pListHead;
        ListNode* p2 = pListHead;
        int i;
        for(i = 0; p1 != NULL; i++){
            if(i >= k)
                p2 = p2 -> next;
            p1 = p1 -> next;
        }
        return i >= k ? p2 : NULL;
    }
};
```

<span id="反转链表"></span>
## 15、反转链表
```
输入一个链表，反转链表后，输出新链表的表头
```
```cpp
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* ReverseList(ListNode* pHead) {
        ListNode* pNew = pHead;
        ListNode* pPre = NULL;
        while(pNew){
            ListNode* pNext = pNew -> next;
            pNew -> next = pPre;
            pPre = pNew;
            pNew = pNext;
        }
        return pPre;
    }
};
```

<span id="合并两个排序的链表"></span>
## 16、合并两个排序的链表
```
输入两个单调递增的链表，输出两个链表合成后的链表
要求：合成后的链表满足单调不减规则。
```
```cpp
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
	//递归
        /*if(pHead1==NULL)
            return pHead2;
        else if(pHead2==NULL)
            return pHead1;
        ListNode* p=NULL;
        if(pHead1->val<pHead2->val){
            p = pHead1;
            p->next = Merge(pHead1->next,pHead2);
        }
        else{
            p = pHead2;
            p->next = Merge(pHead1,pHead2->next);
        }
        return p;*/
	
	// 非递归
        ListNode* head = new ListNode(0);
        ListNode* Head = head;
        while(pHead1 && pHead2){
            if(pHead1 -> val <= pHead2 -> val){
                head -> next = pHead1;
                pHead1 = pHead1 -> next;
            }
            else{
                head -> next =pHead2;
                pHead2 = pHead2 -> next;
            }
            head = head -> next;
        }
        if(pHead1){
            head -> next = pHead1;
        }
        if(pHead2){
            head -> next = pHead2;
        }
        return Head -> next;
    }
};
```

<span id="树的子结构"></span>
## 17、树的子结构
```
输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
```
```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :<>
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
    {
        if(pRoot1 == NULL || pRoot2 == NULL) return false;
        return isSubtree(pRoot1, pRoot2) || HasSubtree(pRoot1 -> left, pRoot2) || HasSubtree(pRoot1 ->right, pRoot2);
    }
    
    bool isSubtree(TreeNode* pRoot1, TreeNode* pRoot2){
        if(pRoot2 == NULL) return true;
        if(pRoot1 == NULL || (pRoot1 -> val != pRoot2 ->val)) return false;
        return isSubtree(pRoot1 -> left, pRoot2 ->left) && isSubtree(pRoot1 -> right, pRoot2 -> right);
    }
};
```

<span id="二叉树的镜像"></span>
## 18、二叉树的镜像
```
操作给定的二叉树，将其变换为源二叉树的镜像。

输入描述：
二叉树的镜像定义：源二叉树 
    	    8
    	   /  \
    	  6   10
    	 / \  / \
    	5  7 9 11
    	镜像二叉树
    	    8
    	   /  \
    	  10   6
    	 / \  / \
    	11 9 7  5
```
```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    void Mirror(TreeNode *pRoot) {
        //递归
	/*if(pRoot == NULL) return ;
        TreeNode* pRight = pRoot -> right;
        pRoot -> right = pRoot -> left;
        pRoot -> left = pRight;
        Mirror(pRoot -> left);
        Mirror(pRoot -> right);*/
	
	//非递归
        if(pRoot == NULL)
            return ;
        stack<TreeNode*> stackNode;
        stackNode.push(pRoot);
        while(stackNode.size()){
            TreeNode *tree = stackNode.top();
            stackNode.pop();
            if(tree->left!=NULL || tree->right!=NULL){
                TreeNode *temp = tree->right;
                tree->right = tree->left;
                tree->left = temp;
            }
            if(tree->left)
                stackNode.push(tree->left);
            if(tree->right)
                stackNode.push(tree->right);
        }
    }
};
```

<span id="顺时针打印矩阵"></span>
## 19、顺时针打印矩阵
```
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 
则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
```
```cpp
class Solution {
public:
    vector<int> printMatrix(vector<vector<int> > matrix) {
        int row = matrix.size(), col = matrix[0].size();
        int left = 0, right = col-1, top = 0, bottom = row - 1;
        vector<int> res;
        while(left <= right && top <= bottom){
            if(left <= right)
                for(int i = left; i<=right; i++)
                    res.push_back(matrix[top][i]);
            if(top < bottom && left <= right)
                for(int i = top + 1; i <= bottom; i++)
                    res.push_back(matrix[i][right]);
            if(left < right && top < bottom)
                for(int i = right - 1; i >= left; i--)
                    res.push_back(matrix[bottom][i]);
            if(top + 1 < bottom && left < right)
                for(int i = bottom - 1; i > top; i--)
                    res.push_back(matrix[i][left]);
            left++, right--, top++, bottom--;
        }
        return res;
    }
};
```

<span id="包含min函数的栈"></span>
## 20、包含min函数的栈
```
定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
```
```cpp
class Solution {
public:
    stack<int> stackAll;
    stack<int> stackMin;
    void push(int value) {
        stackAll.push(value);
        if(stackMin.empty() || stackMin.top() > value) stackMin.push(value);
        else
            stackMin.push(stackMin.top());
    }
    void pop() {
        stackAll.pop();
        stackMin.pop();
    }
    int top() {
        return stackAll.top();
    }
    int min() {
        return stackMin.top();
    }
};
```

<span id="栈的压入、弹出序列"></span>
## 21、栈的压入、弹出序列
```
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。
假设压入栈的所有数字均不相等。
例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，
但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
```
```cpp
class Solution {
public:
    bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        int i = 0, j = 0;
        int len = pushV.size();
        if (len <= 0) return false;
        vector<int> res;
        while(i < len){
            res.push_back(pushV[i]);
            i++;
            while(j < popV.size() && popV[j] == res.back()){
                res.pop_back();
                j++;
            }
        }
        return res.empty();
    }
};
```

<span id="从上往下打印二叉树"></span>
## 22、从上往下打印二叉树
```
从上往下打印出二叉树的每个节点，同层节点从左至右打印。
```
```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    vector<int> PrintFromTopToBottom(TreeNode* root) {
        queue<TreeNode*> que;
        vector<int> vec;
        que.push(root);
        if(!root) return vec;
        while(!que.empty()){
            TreeNode* node = que.front();
            vec.push_back(node -> val);
            if(node -> left != NULL)
                que.push(node -> left);
            if(node -> right != NULL)
                que.push(node -> right);
            que.pop();
        }
        return vec;
    }
};
```

<span id="二叉搜索树的后序遍历序列"></span>
## 23、二叉搜索树的后序遍历序列
```
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
```
```cpp
class Solution {
public:
    bool VerifySquenceOfBST(vector<int> sequence) {
        return bst(sequence, 0, sequence.size()-1);
    }
    
    bool bst(vector<int> sequence, int begin, int end){
        if (sequence.empty()) return false;
        int root = sequence[end];
        int i = begin;
        for(; i < end; i++)
            if(sequence[i] > root)
                break;
        for(int j = i; j < end; j++)
            if (sequence[j] < root)
                return false;
        bool left = true;
        bool right = true;
        if(i > begin)
            left = bst(sequence, begin, i-1);
        if(i < end){
            right = bst(sequence, i, end - 1);
        }
        return left && right;
    }
};
```
	
<span id="二叉树中和为某一个值的路径"></span>
## 24、二叉树中和为某一个值的路径
```
输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。
(注意: 在返回值的list中，数组长度大的数组靠前)
```
```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    vector<vector<int> > FindPath(TreeNode* root,int expectNumber) {
        dfs(root, expectNumber);
        return res;
    }
    
    void dfs(TreeNode* root, int num){
        if(!root) return ;
        path.push_back(root -> val);
        num = num - root -> val;
        if(num == 0 && root -> left == NULL && root -> right == NULL)
            res.push_back(path);
        dfs(root -> left, num);
        dfs(root -> right, num);
        path.pop_back();
    }
};
```

<span id="复杂链表的复制"></span>
## 25、复杂链表的复制
```
输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），
返回结果为复制后复杂链表的head。
（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
```
```cpp
/*
struct RandomListNode {
    int label;
    struct RandomListNode *next, *random;
    RandomListNode(int x) :
            label(x), next(NULL), random(NULL) {
    }
};
*/
class Solution {
public:
    RandomListNode* Clone(RandomListNode* pHead)
    {
        if(pHead == NULL) return NULL;
        RandomListNode* Head = pHead;
        while(Head != NULL){
            RandomListNode* pNew = new RandomListNode(Head -> label);
            pNew -> next = Head -> next;
            Head -> next = pNew;
            Head = pNew -> next;
        }
        RandomListNode* p_head = pHead;
        while(p_head != NULL){
            RandomListNode* pNew = p_head -> next;
            if(p_head -> random != NULL)
                pNew -> random = p_head -> random ->next;
            p_head = pNew -> next;
        }
        p_head = pHead;
        RandomListNode* pClone = p_head -> next;
        while(p_head -> next != NULL){
            RandomListNode* pNew = p_head -> next;
            p_head -> next = pNew -> next;
            p_head = pNew;
        }
        return pClone;
    }
};
```

<span id="二叉搜索树与双向链表"></span>
## 26、二叉搜索树与双向链表
```
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。
要求不能创建任何新的结点，只能调整树中结点指针的指向。
```
```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    TreeNode* Convert(TreeNode* pRootOfTree)
    {
        //中序遍历
        //返回链表的头节点
        if(!pRootOfTree) return NULL;
        TreeNode* pre = NULL;
        convert(pRootOfTree, pre);
        while(pRootOfTree -> left)
            pRootOfTree = pRootOfTree -> left;
        return pRootOfTree;
    }
    
    void convert(TreeNode* root, TreeNode* &pre){
        if(!root) return ;
        convert(root -> left, pre);
        root -> left = pre;
        if(pre)
            pre -> right = root;
        pre = root;
        convert(root -> right, pre);
    }
};
```

<span id="字符串的排列"></span>
## 27、字符串的排列
```
输入一个字符串,按字典序打印出该字符串中字符的所有排列。
例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
```
```cpp
class Solution {
public:
    vector<string> Permutation(string str) {
        vector<string> res;
        if(str.size() == 0) return res;
        sort(str.begin(), str.end());
        do
            res.push_back(str);
        while(next_permutation(str.begin(), str.end()));
        return res;
    }
};
```

<span id="数组中出现次数超过一半的数字"></span>
## 28、数组中出现次数超过一半的数字
```
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
```
```cpp
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        //排序后，若存在符合条件的数，则一定是数组中间的那个数
        //使用sort,时间复杂度为nlogn,非最优
        int len = numbers.size();
        sort(numbers.begin(), numbers.end());
        int mid = numbers[len/2], count = 0;
        for(int i = 0; i < len; i++)
            if(numbers[i] == mid)
                count++;
        return count > len / 2 ? mid : 0;
    }
};
```

<span id="最小的K个数"></span>
## 29、最小的K个数
```
输入n个整数，找出其中最小的K个数。
例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
```
```cpp
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        //最大堆实现，前k个数建立大根堆，时间复杂度nlogk
        if(k > input.size() || k <= 0 || input.size()==0)
            return vector<int> ();
        vector<int> res(input.begin(), input.begin() + k);
        make_heap(res.begin(), res.begin() + k);
        for(int i = k; i < input.size(); i++){
            if(input[i] < res[0]){
                pop_heap(res.begin(), res.end());
                res.pop_back();
                res.push_back(input[i]);
                push_heap(res.begin(), res.end());
            }
        }
        sort(res.begin(), res.end());
        return res;
    }
};
```

<span id="连续子数组的最大和"></span>
## 30、连续子数组的最大和
```
HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:
在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。
但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？
例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。
给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)
```
```cpp
class Solution {
public:
    int FindGreatestSumOfSubArray(vector<int> array) {
        int max_single = array[0], max_all = array[0];
        for(int i = 1; i < array.size(); i++){
            max_single = max(array[i], max_single + array[i]);
            max_all = max(max_all, max_single);
        }
        return max_all;
    }
};
```

<span id="整数中1出现的次数（从1到n整数中1出现的次数）"></span>
## 31、整数中1出现的次数（从1到n整数中1出现的次数）
```
求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？
为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,
但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,
可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。
```
```cpp
class Solution {
public:
    int NumberOf1Between1AndN_Solution(int n)
    {
        int sum = 0;
        for(int i = 0; i <= n; i++)
            sum += Count(i);
        return sum;
    }
    
    int Count(int i){
        int count = 0;
        while(i){
            if(i % 10 == 1)
                count++;
            i /= 10;
        }
        return count;
    }
};
```

<span id="把整数排成最小的数"></span>
## 32、把整数排成最小的数
```
输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
```
```cpp
class Solution {
public:
    string PrintMinNumber(vector<int> numbers) {
        string s;
        sort(numbers.begin(), numbers.end(), cmp);
        for(int i = 0; i < numbers.size(); i++){
            s += to_string(numbers[i]);
        }
        return s;
    }
    
    static bool cmp(int a, int b){
        string A = to_string(a) + to_string(b);
        string B = to_string(b) + to_string(a);
        return A < B;
    }
};
```

<span id="丑数"></span>
## 33、丑数
```
把只包含质因子2、3和5的数称作丑数（Ugly Number）。
例如6、8都是丑数，但14不是，因为它包含质因子7。 
习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
```
```cpp
class Solution {
public:
    int GetUglyNumber_Solution(int index) {
        if(index < 7) return index;
        vector<int> res(index);
        res[0] = 1;
        int i1 = 0, i2 = 0, i3 = 0;
        for(int i = 1; i < index; i++){
            res[i] = min(res[i1] * 2, min(res[i2] * 3, res[i3] * 5));
            if(res[i] == 2 * res[i1])
                i1++;
            if(res[i] == 3 * res[i2])
                i2++;
            if(res[i] == 5 * res[i3])
                i3++;
        }
        return res[index - 1];
        
    }
};
```

<span id="第一次只出现一次的字符"></span>
## 34、第一次只出现一次的字符
```
在一个字符串(0<=字符串长度<=10000，
全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 
如果没有则返回 -1（需要区分大小写）.
```
```cpp
class Solution {
public:
    int FirstNotRepeatingChar(string str) {
        map<char, int> mp;
        for(int i = 0; i < str.size(); i++)
            mp[str[i]]++;
        for(int i = 0; i < str.size(); i++)
            if(mp[str[i]] == 1)
                return i;
        return -1;
    }
};
```

<span id="数组中的逆序对"></span>
## 35、数组中的逆序对
```
在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。
输入一个数组,求出这个数组中的逆序对的总数P。
并将P对1000000007取模的结果输出。 即输出P%1000000007
```
```cpp
class Solution {
public:
    int InversePairs(vector<int> data) {
        if(data.size()<1)
            return 0;
        vector<int> copy;
        for(int i=0;i<int(data.size());i++){
            copy.push_back(data[i]);
        }
        long count=InversePairsCore(data,copy,0,int(data.size())-1);
        return count%1000000007;
    }
    
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
};
```

<span id="两个链表的第一个公共结点"></span>
## 36、两个链表的第一个公共结点
```
输入两个链表，找出它们的第一个公共结点。
```
```cpp
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
        ListNode* p1 = pHead1;
        ListNode* p2 = pHead2;
        while(p1 != p2){
            p1 = p1 == NULL ? pHead2 : p1 -> next;
            p2 = p2 == NULL ? pHead1 : p2 -> next;
        }
        return p1;
    }
};
```

<span id="数字在排序数组中出现的次数"></span>
## 37、数字在排序数组中出现的次数
```
统计一个数字在排序数组中出现的次数。
```
```cpp
class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        // 二分查找
        int mid = -1, start = 0, end = data.size()-1;
        while(start < end){
            mid = (start + end) >> 1;
            if(data[mid] < k)
                start = mid + 1;
            else if(data[mid] > k)
                end = mid - 1;
            else
                break;
        }
        int i = mid;
        int count = 0;
        while(i>=0 && data[i] == k){
            count++;
            i--;
        }
        i = mid + 1;
        while(i< data.size() && data[i] == k){
            count++;
            i++;
        }
        return count;
    }
};
```

<span id="二叉树的深度"></span>
## 38、二叉树的深度
```
输入一棵二叉树，求该树的深度。
从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
```
```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    int TreeDepth(TreeNode* pRoot)
    {
        if(pRoot == NULL) return 0;
        int left = TreeDepth(pRoot -> left);
        int right = TreeDepth(pRoot -> right);
        return left > right ? (left + 1) : (right + 1);
    }
};
```

<span id="平衡二叉树"></span>
## 39、平衡二叉树
```
输入一棵二叉树，判断该二叉树是否是平衡二叉树。
```
```cpp
class Solution {
public:
    bool IsBalanced_Solution(TreeNode* pRoot) {
        //1、树为空是平衡二叉树
        //2、左右树高相差不能超过1
        //3、其子树也要是平衡二叉树
        if(pRoot == NULL) return true;
        if(abs(getDepth(pRoot -> left) - getDepth(pRoot -> right)) > 1)
            return false;
        return IsBalanced_Solution(pRoot -> left) && IsBalanced_Solution(pRoot -> right);
    }
    
    int getDepth(TreeNode* root){
        if(root ==NULL) return 0;
        int left = getDepth(root -> left);
        int right = getDepth(root -> right);
        return left > right ? (left + 1) : (right + 1);
    }
};
```

<span id="数组中只出现一次的数字"></span>
## 40、数组中只出现一次的数字
```
一个整型数组里除了两个数字之外，其他的数字都出现了偶数次。请写程序找出这两个只出现一次的数字。
```
```cpp
class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        //相同的数异或为0，异或结果为1的那一位可以将两个数字分开；
        int num = data[0];
        for(int i = 1; i < data.size(); i++)
            num ^= data[i];
        if(num == 0) return ;
        int move = 0;
        while((num & 1)==0){
            num >>= 1;
            move++;
        }
        *num1 = 0, *num2 = 0;
        for(int i = 0; i < data.size(); i++){
            if((data[i] >> move) & 1)
                *num1 ^= data[i];
            else
                *num2 ^= data[i];
        }
    }
};
```

<span id="和为S的连续正数序列"></span>
## 41、和为S的连续正数序列
```
题目描述
小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。
但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。
没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。
现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!

输出描述:
输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序
```
```cpp
class Solution {
public:
    vector<vector<int> > FindContinuousSequence(int sum) {
        //双指针问题，当总和大于sum,左指针右移；
        vector<vector<int>> res;
        int left = 1, right = 1, sum_ = 1;
        while(left <= right){
            right++;
            sum_ += right;
            while(sum_ > sum){
                sum_ -= left;
                left++;
            }
            if(left != right && sum_ == sum){
                vector<int> vec;
                for(int i = left; i <= right; i++)
                    vec.push_back(i);
                res.push_back(vec);
            }
        }
        return res;
    }
};
```

<span id="和为S的两个数字"></span>
## 42、和为S的两个数字
```
题目描述
输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，
如果有多对数字的和等于S，输出两个数的乘积最小的。

输出描述:
对应每个测试案例，输出两个数，小的先输出。
```
```cpp
class Solution {
public:
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        //递增排序的数组，两端的数相乘积小于较中间的数相乘
        vector<int> res;
        int len = array.size(), left = 0, right = len - 1;
        if(len <= 0) return res;
        while(left < right){
            int sum_ = array[left] + array[right];
            if(sum_ < sum) left++;
            else if(sum_ > sum) right--;
            else{
                res.push_back(array[left]);
                res.push_back(array[right]);
                break;
            }
        }
        return res;
    }
};
```

<span id="左旋转字符串"></span>
## 43、左旋转字符串
```
汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。
对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。
例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。
是不是很简单？OK，搞定它！
```
```cpp
class Solution {
public:
    string LeftRotateString(string str, int n) {
        int len = str.size();
        flip_string(str, 0, len - 1);
        flip_string(str, 0, len - 1 - n);
        flip_string(str, len - n, len - 1);
        return str;
    }
    
    void flip_string(string &str, int left, int right){
        while(left < right){
            swap(str[left], str[right]);
            left++;
            right--;
        }
    }
};
```

<span id="翻转单词顺序序列"></span>
## 44、翻转单词顺序序列
```
牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。
同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。
例如，“student. a am I”。
后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。
Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？
```
```cpp
class Solution {
public:
    string ReverseSentence(string str) {
        //1.先将字符串翻转
        //2、将翻转后的字符串，在句末加上一个空格字符，
        //   利用空格字符，将每个单词分开，在进行翻转。得到结果
        int len = str.size();
        flip_string(str, 0 , len-1);
        str += ' ';
        int mark = 0;
        for(int i = 0; i < str.size(); i++){
            if(str[i] == ' '){
                flip_string(str, mark, i-1);
                mark = i + 1;
            }
        }
        return str.substr(0, len);
    }
    
    void flip_string(string &str, int left, int right){
        while(left < right){
            swap(str[left], str[right]);
            left++;
            right--;
        }
    }
};
```

<span id="扑克牌顺子"></span>
## 45、扑克牌顺子
```
LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...
他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！
“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,
决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。
上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。
LL决定去买体育彩票啦。现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 
如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。
```
```cpp
class Solution {
public:
    bool IsContinuous( vector<int> numbers ) {
        //1、无重复数字
        //2、num_max-num_min<5;
        int len =  numbers.size();
        int a[14] = {0};
        int min_num = 13, max_num = 0;
        if(len != 5) return false;
        for(int i = 0; i < len; i++){
            a[numbers[i]]++;
            if(numbers[i] == 0) continue;
            if(a[numbers[i]] > 1) return false;
            if(numbers[i] <= min_num) min_num = numbers[i];
            if(numbers[i] >= max_num) max_num = numbers[i];
        }
        if(max_num - min_num >= 5) return false;
        else return true;
    }
};
```

<span id="孩子们的游戏（圆圈中最后剩下的数）"></span>
## 46、孩子们的游戏（圆圈中最后剩下的数）
```
每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。
HF作为牛客的资深元老,自然也准备了一些小游戏。
其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。
然后,他随机指定一个数m,让编号为0的小朋友开始报数。
每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,
从他的下一个小朋友开始,继续0...m-1报数....这样下去....
直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。
请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)
```
```cpp
class Solution {
public:
    int LastRemaining_Solution(int n, int m)
    {
        //约瑟夫环问题
        if(n == 0 || m == 0) return -1;
        int s = 0;
        for(int i = 2; i <= n; i++)
            s = (s + m) % i; 
        return s;
    }
};
```

<span id="求1+2+3+...+n"></span>
## 47、求1+2+3+...+n
```
求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
```
```cpp
class Solution {
public:
    int Sum_Solution(int n) {
        int sum = n;
        sum && (sum += Sum_Solution(n - 1));
        return sum;
    }
};
```

<span id="不用加减乘除做加法"></span>
## 48、不用加减乘除做加法
```
写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
```
```cpp
class Solution {
public:
    int Add(int num1, int num2)
    {
        while(num2 != 0){
            int temp = num1 ^ num2;
            num2 = (num1 & num2) << 1;
            num1 = temp;
        }
        return num1;
    }
};
```

<span id="把字符串转换成整数"></span>
## 49、把字符串转换成整数
```
将一个字符串转换成一个整数(实现Integer.valueOf(string)的功能，
但是string不符合数字要求时返回0)，要求不能使用字符串转换整数的库函数。 
数值为0或者字符串不是一个合法的数值则返回0。

输入描述:
输入一个字符串,包括数字字母符号,可以为空
输出描述:
如果是合法的数值表达则返回该数字，否则返回0
示例1 
输入
+2147483647
    1a33
	
输出
2147483647
    0
```
```cpp
class Solution {
public:
    int StrToInt(string str) {
        int len = str.size(), s = 1, res = 0;
        if(str[0]== '-') s = -1;
        for(int i = (str[0] == '+' || str[0] == '-') ? 1 : 0; i < len; i++){
            if(str[i] < '0' || str[i] > '9') return 0;
            res = (res << 1) + (res << 3) + (str[i] & 0xf);
        }
        return res * s;
    }
};
```

<span id="数组中重复的数字"></span>
## 50、数组中重复的数字
```
在一个长度为n的数组里的所有数字都在0到n-1的范围内。 
数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。
请找出数组中任意一个重复的数字。 
例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
```
```cpp
class Solution {
public:
    // Parameters:
    //        numbers:     an array of integers
    //        length:      the length of array numbers
    //        duplication: (Output) the duplicated number in the array number
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false
    bool duplicate(int numbers[], int length, int* duplication) {
        map<int, int> mp;
        for(int i = 0; i < length; i++)
            mp[numbers[i]]++;
        bool flag = false;
        for(int i = 0; i < length; i++){
            if(mp[numbers[i]] > 1){
                *duplication = numbers[i];
                flag = true;
                break;
            }
        }
        return flag;
    }
};
```

<span id="构建乘积数组"></span>
## 51、构建乘积数组
```
给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],
其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。
```
```cpp
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        int len = A.size();
        vector<int> vec;
        vec.push_back(1);
        for(int i = 1; i < len; i++){
            vec.push_back(vec.back()*A[i-1]);
        }
        int tmp = 1;
        for(int i = len - 1; i >= 0; i--){
            vec[i] = vec[i] * tmp;
            tmp *= A[i];
        }
        return vec;
    }
};
```

<span id="正则表达式匹配"></span>
## 52、正则表达式匹配
```
请实现一个函数用来匹配包括'.'和'*'的正则表达式。
模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 
在本题中，匹配是指字符串的所有字符匹配整个模式。
例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配
```
```cpp
class Solution {
public:
    bool match(char* str, char* pattern)
    {
        if(*str == '\0' && *pattern == '\0') return true;
        if(*str != '\0' && *pattern == '\0') return false;
        if(*(pattern + 1) == '*'){
            if(*str == *pattern || (*str != '\0' && *pattern == '.')){
                return match(str + 1, pattern) || match(str, pattern + 2);
            }
            else
                return match(str, pattern + 2);
        }
        else{
            if(*str == *pattern || (*str != '\0' && *pattern == '.')){
                return match(str + 1, pattern + 1);
            }
            else
                return false;
        }
    }
};
```

<span id="表示数值的字符串"></span>
## 53、表示数值的字符串
```
请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 
但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
```
```cpp
class Solution {
public:
    bool isNumeric(char* string)
    {
        int count_dot = 0, count_e = 0, count_num = 0;
        if(*string == '+' || *string == '-') string++;
        if(*string == '\0') return false;
        while(*string != '\0'){
            if(*string >= '0' && *string <= '9'){
                count_num = 1;
                string++;
            }
            else if(*string == '.'){
                if(count_dot > 0 || count_e > 0)
                    return false;
                count_dot = 1;
                string++;
            }
            else if(*string == 'E' || *string == 'e'){
                if(count_num == 0 || count_e > 0)
                    return false;
                count_e = 1;
                string++;
                if(*string == '+' || *string == '-')
                    string++;
                if(*string == '\0')
                    return false;
            }
            else
                return false;
        }
        return true;
    }
};
```

<span id="字符流中第一个不重复的字符"></span>
## 54、字符流中第一个不重复的字符
```
请实现一个函数用来找出字符流中第一个只出现一次的字符。
例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。
当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。

如果当前字符流没有存在出现一次的字符，返回#字符。
```
```cpp
class Solution
{
public:
  //Insert one char from stringstream
    void Insert(char ch)
    {
        s += ch;
         hash[ch]++;
    }
  //return the first appearence once char in current stringstream
    char FirstAppearingOnce()
    {
        for(int i = 0; i < s.size(); i++)
            if(hash[s[i]] == 1)
                return s[i];
        return '#';
    }
    
    string s;
    char hash[128] = {0};
};
```

<span id="链表中环的入口节点"></span>
## 55、链表中环的入口节点
```
给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。
```
```cpp
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead)
    {
        if(pHead == NULL || pHead -> next == NULL) return NULL;
        ListNode* p1 = pHead;
        ListNode* p2 = pHead;
        while(p2 != NULL && p2 -> next != NULL){
            p1 = p1 -> next;
            p2 = p2 -> next -> next;
            if(p1 == p2){
                p1 = pHead;
                while(p1 != p2){
                    p1 = p1 -> next;
                    p2 = p2 -> next;
                }
                if(p1 == p2)
                    return p1;
            }
        }
        return NULL;
    }
};
```

<span id=""></span>
## 56、
```

```
```cpp

```

<span id=""></span>
## 57、
```

```
```cpp

```

<span id=""></span>
## 58、
```

```
```cpp

```

<span id=""></span>
## 59、
```

```
```cpp

```

<span id=""></span>
## 60、
```

```
```cpp

```

<span id=""></span>
## 61、
```

```
```cpp

```

<span id=""></span>
## 62、
```

```
```cpp

```

<span id=""></span>
## 63、
```

```
```cpp

```

<span id=""></span>
## 64、
```

```
```cpp

```

<span id="矩阵中的路径"></span>
## 65、矩阵中的路径
```
题目：
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。
路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。
如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。
注意：
* 输入的路径不为空；
* 所有出现的字符均为大写英文字母；

样例：
matrix=
[
  ["A","B","C","E"],
  ["S","F","C","S"],
  ["A","D","E","E"]
]
str="BCCE" , return "true" 
str="ASAE" , return "false"

思路：
* 从每个点开始遍历，使用dfs深度优先遍历
* 找到该字符串后输出true,遍历完没有找到输出false
```
```cpp
class Solution {
public:
    bool hasPath(vector<vector<char>>& matrix, string str) {
        for(int i=0;i < matrix.size();i++){
            for(int j=0;j<matrix[i].size();j++){
                if(dfs(matrix,str,0,i,j))
                return true;
            }
        }
        return false;
    }
    
    bool dfs(vector<vector<char>>& matrix, string &str,int u,int x,int y){
        if(matrix[x][y]!=str[u]) return false;
        if(u==str.size()-1) return true;
        int a[4]={-1,0,1,0}, b[4]={0,1,0,-1};
        char t=matrix[x][y];
        matrix[x][y]='*';
        for(int i=0;i<4;i++){
            int p=x+a[i],q=y+b[i];
            if(p>=0 && p<matrix.size() && q>=0 && q<=matrix[p].size()){
                if(dfs(matrix,str,u+1,p,q)) return true;
            }
        }
        matrix[x][y]=t;  // 回溯
        return false;
    }
};
```

<sapn id="机器人的运动范围"></span>
## 66、机器人的运动范围
```
题目：
地上有一个 m 行和 n 列的方格，横纵坐标范围分别是 0∼m−1 和 0∼n−1。
一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格。
但是不能进入行坐标和列坐标的数位之和大于 k 的格子。
请问该机器人能够达到多少个格子？

样例：
输入：k=7, m=4, n=5
输出：20
```
```cpp
class Solution {
public:
    int movingCount(int threshold, int rows, int cols)
    {
        if(!rows || !cols) return 0;
        vector<vector<bool>> dp(rows,vector<bool>(cols,false));
        queue<pair<int,int>> que;
        que.push({0,0});
        int count_s=0;
        int a[4]={-1,0,1,0},b[4]={0,1,0,-1};
        while(!que.empty()){
            auto t=que.front();
            que.pop();
            if(sum_of_num(t)>threshold || dp[t.first][t.second]) continue;
            count_s++;
            dp[t.first][t.second]=true;
            for(int i=0;i<4;i++){
                int x=t.first+a[i],y=t.second+b[i];
                if(x>=0 && x<rows&& y>=0 && y<cols) que.push({x,y});
            }
        }
        return count_s;
    }
    
    int sum_of_num(pair<int,int> p){
        int count_num=0;
        while(p.first){
            count_num+=p.first%10;
            p.first/=10;
        }
        while(p.second){
            count_num+=p.second%10;
            p.second/=10;
        }
        return count_num;
    }
};
```

<span id ="在O(1)时间删除链表节点"></span>
## 在O(1)时间删除链表节点
```
题目:
给定单向链表的一个节点指针，定义一个函数在O(1)时间删除该结点。
假设链表一定存在，并且该节点一定不是尾节点。
注意：实际代码中不存在输入输出，只是在后台测试时会使用输入输出样例

样例：
输入：链表 1->4->6->8
      删掉节点：第2个节点即6（头节点为第0个节点）
输出：新链表 1->4->8
```
**代码**
```cpp
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
    void deleteNode(ListNode* node) {
        auto p=node->next;
        node->val=p->val;
        node->next=p->next;
        delete p;
    }
};
```

<span id="删除链表中重复的节点"></span>
## 删除链表中重复的节点
```
题目:
在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留。

样例：
输入：1->2->3->3->4->4->5
输出：1->2->5

说明:
代码中的tmp存储的是重复节点后的节点
```
**代码**
```cpp
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
    ListNode* deleteDuplication(ListNode* head) {
        ListNode* dummy=new ListNode(0);
        dummy->next=head;
        ListNode* p=dummy;
        while(p->next){
            ListNode* tmp=p->next;
            while(tmp && p->next->val==tmp->val){
                tmp=tmp->next;
            }
            if(p->next->next==tmp){
                p=p->next;
            }
            else{
                p->next=tmp;
            }
        }
        return dummy->next;
    }
};
```

<span id="正则表达式匹配"></span>
## 正则表达式匹配
```
题目：
请实现一个函数用来匹配包括'.'和'*'的正则表达式。
模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。
在本题中，匹配是指字符串的所有字符匹配整个模式。
例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配。

样例：
输入：s="aa" p="a*"
输出:true
```
**代码**
```cpp
class Solution {
public:
    vector<vector<int>>f;
    int n,m;
    bool isMatch(string s, string p) {
        n=s.size();
        m=p.size();
        f=vector<vector<int>>(n+1,vector<int>(m+1,-1));
        return dp(0,0,s,p);
    }
    
    bool dp(int x,int y,string &s,string &p){
        if(f[x][y]!=-1) return f[x][y];
        if(y==m){
            return f[x][y]=x==n;
        }
        bool firstMatch=x<n && (s[x]==p[y] || p[y]=='.');
        bool ans;
        if(y+1<m && p[y+1]=='*'){
            ans = dp(x,y+2,s,p) || firstMatch && dp(x+1,y,s,p);
        }
        else{
            ans = firstMatch && dp(x+1,y+1,s,p);
        }
        return f[x][y]=ans;
    }
};
```

<span id="表示数值的字符串"></span>
## 表示数值的字符串
```
题目：
请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。
但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
注意:
小数可以没有整数部分，例如.123等于0.123；
小数点后面可以没有数字，例如233.等于233.0；
小数点前面和后面可以有数字，例如233.666;
当e或E前面没有数字时，整个字符串不能表示数字，例如.e1、e1；
当e或E后面没有整数时，整个字符串不能表示数字，例如12e、12e+5.4;

样例：
输入: "0"
输出: true
```
**代码**
```cpp
class Solution {
public:
    bool isNumber(string s) {
        int n=s.size();
        if(n==0) return false;
        int s_num=0,s_dot=0,s_e=0;
        if(s[0]=='+' ||s[0]=='-') s=s.substr(1,n-1);
        if(s[0]=='.' && s.size()==1) return false;
        for(int i=0;i<s.size();i++){
            if(s[i]>='0' && s[i]<='9') {
                s_num++;
                continue;
            }
            else if(s[i]=='.'){
                if(s_e>0 || s_dot>0) return false;
                s_dot++;
            }
            else if(s[i]=='E' || s[i]=='e'){
                if(s_e>0 || s_num==0) return false;
                s_e++;
                i++;
                if(s[i]=='+' || s[i]=='-'){
                    i++;
                }
                if(s[i]=='\0') return false;
            }
            else return false;
        }
        return true;
    }
};
```

<span id="链表中环的入口节点"></span>
## 链表中环的入口节点
```
题目：
给定一个链表，若其中包含环，则输出环的入口节点。
若其中不包含环，则输出null。

样例：
给定如上所示的链表：    注意：3与6形成环，即入口节点为3.
[1, 2, 3, 4, 5, 6]
2
注意，这里的2表示编号是2的节点，节点编号从0开始。所以编号是2的节点就是val等于3的节点。
则输出环的入口节点3.
```
**代码**
```cpp
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
    ListNode *entryNodeOfLoop(ListNode *head) {
        ListNode* p1=head;
        ListNode* p2=head;
        while(p2!=NULL && p2->next->next){
            p1=p1->next;
            p2=p2->next->next;
            if(p1==p2){
                p2=head;
                while(p1!=p2){
                    p1=p1->next;
                    p2=p2->next;
                }
                if(p1==p2) return p2; 
            }
        }
        return NULL;
        
    }
};
```

<span id="二叉树的镜像"></span>
## 二叉树的镜像
```
题目：
输入一个二叉树，将它变换为它的镜像。

样例：
输入树：
      8
     / \
    6  10
   / \ / \
  5  7 9 11

 [8,6,10,5,7,9,11,null,null,null,null,null,null,null,null] 
输出树：
      8
     / \
    10  6
   / \ / \
  11 9 7  5

 [8,10,6,11,9,7,5,null,null,null,null,null,null,null,null]
```
**代码**
```cpp
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
    void mirror(TreeNode* root) {
        if (!root) return;
        swap(root->left, root->right);
        mirror(root->left);
        mirror(root->right);
    }
};
```

<span id="栈的压入、弹出序列"></span>
## 栈的压入、弹出序列
```
题目：
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。
假设压入栈的所有数字均不相等。
例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。
注意：若两个序列长度不等则视为并不是一个栈的压入、弹出序列。若两个序列都为空，则视为是一个栈的压入、弹出序列。

样例：
输入：[1,2,3,4,5]
      [4,5,3,2,1]
输出：true
```
**代码**
```cpp
class Solution {
public:
    bool isPopOrder(vector<int> pushV,vector<int> popV) {
        if(pushV.empty() && popV.empty()) return true;
        if(pushV.empty() || popV.empty() || pushV.size()==popV.empty()) return false;
        stack<int> s;
        int p_size=0;
        for(int i=0;i<pushV.size();i++){
            s.push(pushV[i]);
            while(!s.empty() && s.top()==popV[p_size]){
                p_size++;
                s.pop();
            }
        }
        if(s.empty()) return true;
        return false;
    }
};
```

<span id="二叉搜索树的后序遍历序列"></span>
## 二叉搜索树的后序遍历序列
```
题目：
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
如果是则返回true，否则返回false。
假设输入的数组的任意两个数字都互不相同。

样例：
输入：[4, 8, 6, 12, 16, 14, 10]
输出：true
```
**代码**
```cpp
class Solution {
public:
    bool verifySequenceOfBST(vector<int> sequence) {
        return dfs(sequence,0,sequence.size()-1);
    }
    
    bool dfs(vector<int> sequence,int left,int right){
        if(left>=right) return true;
        int root=sequence[right];
        int k=left;
        while(k<right && sequence[k]<root) k++;
        for(int i=k;i<right;i++){
            if(sequence[i]<root){
                return false;
            }
        }
        return dfs(sequence,left,k-1) && dfs(sequence,k,right-1);
    }
};
```

<span id="二叉树中和为某一值的路径"></span>
## 二叉树中和为某一值的路径
```
题目：
输入一棵二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。

样例：
给出二叉树如下所示，并给出num=22。
      5
     / \
    4   6
   /   / \
  12  13  6
 /  \    / \
9    1  5   1
输出：[[5,4,12,1],[5,6,6,5]]
```
**代码**
```cpp
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
    vector<int> path;
    vector<vector<int>> findPath(TreeNode* root, int sum) {
        dfs(root,sum);
        return res;
    }
    
    void dfs(TreeNode* root, int sum){
        if(!root) return;
        path.push_back(root->val);
        sum=sum-root->val;
        if(!root->left && !root->right && sum==0) res.push_back(path);
        dfs(root->left,sum);
        dfs(root->right,sum);
        path.pop_back();
    }
};
```

<sapn id="复杂链表的复制"></span>
## 复杂链表的复制
```
题目:
请实现一个函数可以复制一个复杂链表。
在复杂链表中，每个结点除了有一个指针指向下一个结点外，还有一个额外的指针指向链表中的任意结点或者null。
```
**代码**
```cpp
/**
 * Definition for singly-linked list with a random pointer.
 * struct ListNode {
 *     int val;
 *     ListNode *next, *random;
 *     ListNode(int x) : val(x), next(NULL), random(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *copyRandomList(ListNode *head) {
        if(head==NULL) return NULL;
        ListNode* p1=head;
        
        while(p1){
            ListNode* pNew=new  ListNode(p1->val);
            pNew->next=p1->next;
            p1->next=pNew;
            p1=pNew->next;
        }
        
        p1=head;
        while(p1){
            ListNode* pNext= p1->next;
            if(p1->random){
                pNext->random=p1->random->next;
            }
            p1=pNext->next;
        }
        
        p1=head;
        ListNode* p2=head->next;
        while(p1->next){
            ListNode* pNext=p1->next;
            p1->next=pNext->next;
            p1=pNext;
        }
        return p2;
    }
};
```

<span id="二叉搜索树与双向链表"></span>
## 二叉搜索树与双向链表
```
题目：
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。
要求不能创建任何新的结点，只能调整树中结点指针的指向。

注意：
需要返回双向链表最左侧的节点。
例如，输入下图中左边的二叉搜索树，则输出右边的排序双向链表。
```

![二叉搜索树与双向链表](https://i.ibb.co/qF2mdYP/image.png)  
```cpp
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
    TreeNode* convert(TreeNode* root) {
        if(root==NULL) return NULL;
        TreeNode* pre=NULL;
        convert(root,pre);
        TreeNode* p1=root;
        while(p1->left){
            p1=p1->left;
        }
        return p1;
    }
    
    void convert(TreeNode* root,TreeNode* &pre){
        if(!root) return ;
        convert(root->left,pre);
        root->left=pre;
        if(pre){
            pre->right=root;
        }
        pre=root;
        convert(root->right,pre);
    }
};
```

<span id="最小的K个数"></span>
## 最小的K个数
```
题目：
输入n个整数，找出其中最小的k个数。

注意：
数据保证k一定小于等于输入数组的长度;
输出数组内元素请按从小到大顺序排序;

样例：
输入：[1,2,3,4,5,6,7,8] , k=4
输出：[1,2,3,4]
```
```cpp
class Solution {
public:
    vector<int> getLeastNumbers_Solution(vector<int> input, int k) {
        priority_queue<int> heap; // 创建一个大根堆
        for(auto x:input){
            heap.push(x);
            if(heap.size()>k){
                heap.pop();
            }
        }
        vector<int> res;
        while(heap.size()){
            res.push_back(heap.top());
            heap.pop();
        }
        reverse(res.begin(),res.end());
        return res;
        
        /*
        vector<int> res;
        for(int i=0;i<k;i++){
            res.push_back(input[i]);
        }
        make_heap(res.begin(),res.end());
        for(int i=k;i<input.size();i++){
            if(input[i]<res[0]){
                pop_heap(res.begin(),res.end());
                res.pop_back();
                res.push_back(input[i]);
                push_heap(res.begin(),res.end());
            }
        }
        sort(res.begin(),res.end());
        return res;
        */
    }
};
```
