# LeetCode Solutions
* [剑指offer](https://github.com/FangChao1086/Data_structures_and_algorithms/blob/master/C、剑指offer.md)
* [LeetCode](https://github.com/FangChao1086/Data_structures_and_algorithms/blob/master/D、LeetCode.md)
* [面试算法题](https://github.com/FangChao1086/Data_structures_and_algorithms/blob/master/E、面试算法题.md)
* [专题](https://github.com/FangChao1086/Data_structures_and_algorithms/tree/master/专题)
  * [数组](https://github.com/FangChao1086/Data_structures_and_algorithms/blob/master/专题/数组.md)  
  * [排序](https://github.com/FangChao1086/Data_structures_and_algorithms/blob/master/专题/排序.md)
  * [链表](https://github.com/FangChao1086/Data_structures_and_algorithms/blob/master/专题/链表.md)
  * [二叉树](https://github.com/FangChao1086/Data_structures_and_algorithms/blob/master/专题/二叉树.md)
  * [动态规划](https://github.com/FangChao1086/Data_structures_and_algorithms/blob/master/专题/动态规划.md)
  * [回溯](https://github.com/FangChao1086/data_structures_and_algorithms/blob/master/专题/回溯.md)
  * [并查集](https://blog.csdn.net/weixin_43824059/article/details/88535734)

## 基础
### 输入
* 数据个数未知
  ```cpp
  1 3 5 10

  // C++
  vector<int> v;
  int tmp;
  while(cin >> tmp){
      v.push_back(tmp);
      if (getchar() == '\n')
          break;
  }
  ```

### 生成数组
```cpp
// C++

// 二维数组
int dp[n][m];  // 方法1
memset(dp,0,sizeof(dp));  // 用0填充

vector<vector<int>> dp(n,vector<int>(m,0));  // 方法2；n*m填充0

vector<int> res(input.begin(), input.begin() + k);  // 方法3；input是已经存在的vector
```

### 数据类型转换
* char转string
  ```cpp
  // 假设mp[0]是char;
  string str;
  str.push_back(mp[0])  // str变成了string类型
  ```
