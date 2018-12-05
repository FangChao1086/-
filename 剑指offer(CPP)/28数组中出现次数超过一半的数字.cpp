//数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
//例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
//由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。


class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        //排序后，若存在符合条件的数，则一定是数组中间的那个数
        //使用sort,时间复杂度为nlogn,非最优
        if(numbers.size()==0)
            return 0;
        sort(numbers.begin(),numbers.end());
        int middle = numbers[numbers.size()/2],count=0;
        for(int i=0;i<numbers.size();i++)
            if (numbers[i]==middle)
                count++;
        return count>numbers.size()/2?middle:0;
    }
};