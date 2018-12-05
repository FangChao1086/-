//一个整型数组里除了两个数字之外，其他的数字都出现了偶数次。请写程序找出这两个只出现一次的数字。


class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        //相同的数异或为0，异或结果为1的那一位可以将两个数字分开；
        int a = data[0],count=0;
        for(int i=1;i<data.size();i++){
            a^=data[i];
        }
        if(!a)
            return ;
        while((a&1)==0){
            a=a>>1;
            count++;
        }
        *num1=0,*num2=0;
        for(int i=0;i<data.size();i++){
            if(data[i]>>count&1)
                *num1^=data[i];
            else
                *num2^=data[i];
        }
    }
};