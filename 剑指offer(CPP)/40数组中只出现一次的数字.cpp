//һ�����������������������֮�⣬���������ֶ�������ż���Ρ���д�����ҳ�������ֻ����һ�ε����֡�


class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        //��ͬ�������Ϊ0�������Ϊ1����һλ���Խ��������ַֿ���
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