//如何得到一个数据流中的中位数？
//如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。
//如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
//我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。


/*
将数据数据分为大根堆（元素值小的数据）和小根堆（元素值大的数据）
1、原始的数据长度为偶，将数据加入到小根堆（加入前新的数据要先加入大根堆，经过大根堆筛选出最大元素进入小根堆）
   这样每次加入数据后小根堆始终保持里面是比大根堆大的元素
2、原始的数据长度为奇，将数据加入大根堆（同样加入前要经过小根堆筛选出最小元素加入大根堆）。。。
*/

/*得到中位数，当数据数量为奇，中位数是小根堆堆顶的元素
当数据数量为偶时，中位数时小根堆堆顶与大根堆堆顶元素和的一半
*/

class Solution {
private:
    vector<int> min;
    vector<int> max;
public:
    void Insert(int num)
    {
        int size=min.size()+max.size();
        if((size&1)==0)    //当插入前的数据长度为偶
        {
            if(max.size()>0 && num<max[0])    //当插入的数据小于大根堆中的最大元素
            {
                max.push_back(num);            //将数据加入大根堆
                push_heap(max.begin(),max.end(),less<int>());    //重新进行堆排序，将最大元素放在第一个
                num=max[0];
                pop_heap(max.begin(),max.end(),less<int>());    //将堆顶元素与最后一个元素互换；弹出堆顶，但并未删除
                max.pop_back();    //删除
            }
            min.push_back(num);
            push_heap(min.begin(),min.end(),greater<int>());
        }
        else
        {
            if(min.size()>0 && num>min[0])
            {
                min.push_back(num);
                push_heap(min.begin(),min.end(),greater<int>());
                num=min[0];
                pop_heap(min.begin(),min.end(),greater<int>());
                min.pop_back();
            }
            max.push_back(num);
            push_heap(max.begin(),max.end(),less<int>());
        }
    }

    double GetMedian()
    {
        int size=min.size()+max.size();
        if((size&1)==0)
            return (min[0]+max[0])/2.0;
        else
            return min[0];
    }

};