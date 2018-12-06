//给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。
//例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，
//那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 
//针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： 
//{[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， 
//{2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。


class Solution {
public:
    vector<int> maxInWindows(const vector<int>& num, unsigned int size)
    {
        //方法一
        /*vector<int> res;
        int length=num.size();
        if(size>length ||size==0 || length==0)
            return res;
        for(int i = 0;i<length-size+1;i++){
            int max = num[i];
            for(int j=i;j<i+size;j++){
                if(num[j]>max)
                    max = num[j];
            }
            res.push_back(max);
        }
        return res;*/
        
        //方法二：
        vector<int> res;    //存最大值
        deque<int> s;    //存下标
        for(int i = 0;i<num.size();i++){
            while(s.size() && num[i]>=num[s.back()])
                s.pop_back();
            while(s.size() && i-s.front()+1>size)
                s.pop_front();
            s.push_back(i);
            if(size && i+1>=size)
                res.push_back(num[s.front()]);
        }
        return res;
    }
};