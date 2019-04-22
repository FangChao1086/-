* [1、找出数组中重复的数字](#找出数组中重复的数字)
* [2、不修改数组找出重复的数字](#不修改数组找出重复的数字)
* [3、二维数组中的查找](#二维数组中的查找)
* [4、替换空格](#替换空格)
* [5、从尾到头打印链表](#从尾到头打印链表)
* [6、重建二叉树](#重建二叉树)
* [7、二叉树的下一个节点](#二叉树的下一个节点)
* [8、用两个栈实现队列](#用两个栈实现队列)
* [9、斐波那契数列](#斐波那契数列)
* [10、旋转数组的最小数字](#旋转数组的最小数字)
* [11、矩阵中的路径](#矩阵中的路径)
* [12、机器人的运动范围](#机器人的运动范围)
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
## 找出数组中重复的数字
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
```
题目:
输入一个整数n，求斐波那契数列的第n项。  
假定从0开始，第0项为0。(n<=39) 

样例:
输入整数 n=5   
返回 5  

思路:
```
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
<span id="旋转数组的最小数字"></span>
## 旋转数组的最小数字
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
**代码**
```cpp
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
```

<span id="矩阵中的路径"></span>
## 矩阵中的路径
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
**代码**
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
## 机器人的运动范围
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
**代码**
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
