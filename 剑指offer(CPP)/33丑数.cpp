//��ֻ����������2��3��5��������������Ugly Number����
//����6��8���ǳ�������14���ǣ���Ϊ������������7�� 
//ϰ�������ǰ�1�����ǵ�һ���������󰴴�С�����˳��ĵ�N��������


class Solution {
public:
    int GetUglyNumber_Solution(int index) {
        if(index<7)
            return index;
        vector<int> res(index);
        res[0]=1;
        int i1=0,i2=0,i3=0;
        for(int i=1;i<index;i++){
            res[i]=min(res[i1]*2,min(res[i2]*3,res[i3]*5));
            if(res[i]==res[i1]*2)
                i1++;
            if(res[i]==res[i2]*3)
                i2++;
            if(res[i]==res[i3]*5)
                i3++;
        }
        return res[index-1];
    }
};