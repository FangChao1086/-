# 面试算法题
[找到数组中第K个数](#找到数组中第K个数)  

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
## cpp版
```cpp
int partition(vector<int> vec, int low, int high) {
	int pre = vec[low];
	while(low < high){
		while(low < high && vec[high] > pre) high--;
		while(low < high && vec[low] < pre) low++;
		int temp = vec[low];
		num[low] = vec[high];
		vec[high] = temp;;
		if(vec[low] == vec[high]) high--;
	}
	vec[low] = pre;
	return low;
}

// 第K小的数
int topK(vector<int> vec, int low,int high, int k) {
	int index = partition(num, low, high);
	if (index == k) {
		return vec[low];
	}
	if (index < k) {
		return topk(vec, index + 1, high, k);
	}
	else return topk(vec, low, index-1, k);
}
```
