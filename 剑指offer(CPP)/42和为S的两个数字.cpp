//题目描述
//输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，
//如果有多对数字的和等于S，输出两个数的乘积最小的。
//输出描述:
//对应每个测试案例，输出两个数，小的先输出。


class Solution {
public:
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        //递增排序的数组，两端的数相乘积小于较中间的数相乘
        int left = 0,right = array.size()-1,sum_x = 0;
        vector<int> res;
        while(left<right){
            sum_x = array[left]+array[right];
            if(sum_x>sum)
                right--;
            if(sum_x<sum)
                left++;
            if(sum_x==sum){
                res.push_back(array[left]);
                res.push_back(array[right]);
                break;
            }
        }
        return res;
    }
};