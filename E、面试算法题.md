# 面试算法题
[找到数组中第K个数](#找到数组中第K个数)  
[求字符串的最长无重复字符子串](#求字符串的最长无重复字符子串)

<span id="找到数组中第K个数"></span>
## 找到数组中第K个数(python)
* 第一次使用整个快排
* 接下来每次丢掉快排的一边
```python
def quicksort(num ,low, high):  # 快速排序
    if low < high:
        location = partition(num, low, high)
        quicksort(num, low, location - 1)
        quicksort(num, location + 1, high)

def partition(num, low, high):
    pivot = num[low]
    while (low < high):
        while (low < high and num[high] > pivot):
            high -= 1
        while (low < high and num[low] < pivot):
            low += 1
        temp = num[low]
        num[low] = num[high]
        num[high] = temp
        if(num[low]==num[high]):
            high -= 1
    num[low] = pivot
    return low


def findkth(num, low, high, k):  # 找到数组里第k个数
    index = partition(num, low, high)
    if index == k: return num[index]
    if index < k:
        return findkth(num, index + 1, high, k)
    else:
        return findkth(num, low, index - 1, k)

pai = [2, 3, 1, 5, 4, 6]
# quicksort(pai, 0, len(pai) - 1)
print(findkth(pai, 0, len(pai) - 1, 5))
```

<span id="求字符串的最长无重复字符子串"></span>
## 求字符串的最长无重复字符子串长度
```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        map<char, int> m;
        int ret = 0;
        int start = 1;//设置起始点
        char c;
        for(int i=1;i<=s.size();i++){
            c = s[i-1];
            if(m[c]>=start){//发现map中已经存在该字符
                start = m[c] + 1;//更新起始点，从该字符的下一个字符为起始点
            }
            else{
                ret = max(ret, i - start + 1);
            }
            m[c] = i;
        }
        return ret;
    }
};
```
