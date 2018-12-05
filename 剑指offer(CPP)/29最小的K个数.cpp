//输入n个整数，找出其中最小的K个数。
//例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。


class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        //全排列，时间复杂度nlogn
        /*vector<int> res;
        if(k>input.size())
            return res;
        sort(input.begin(),input.end());
        for(int i=0;i<k;i++)
            res.push_back(input[i]);
        return res;*/
        
        //最大堆实现，前k个数建立大根堆，时间复杂度nlogk
        //make_heap,pop_heap,push_heap,sort_heap的使用
        if(k>input.size() || k==0 || input.size()<=0)
            return vector<int> ();
        //建堆
        vector<int> res(input.begin(),input.begin()+k);
        make_heap(res.begin(),res.begin()+k);
        //从小到大输出
        for(int i=k;i<input.size();i++){
            if(input[i]<res[0]){
                //取出堆顶放在末尾
                pop_heap(res.begin(),res.end());
                //将尾部的值删除
                res.pop_back();
                //将新的替换值放在尾部
                res.push_back(input[i]);
                //重新形成最大堆
                push_heap(res.begin(),res.end());
            }
        }
        sort(res.begin(),res.end());
        return res;
    }
};