# LeetCode Solutions
* [OJ编程基础](https://github.com/FangChao1086/Data_structures_and_algorithms/blob/master/B、OJ编程基础.md)
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
* 输入数据个数未知
```cpp
vector<int> v;
int tmp;
while(cin >> tmp){
    v.push_back(tmp);
    if (getchar() == '\n')
        break;
}
```
