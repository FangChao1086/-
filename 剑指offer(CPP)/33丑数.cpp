//把只包含质因子2、3和5的数称作丑数（Ugly Number）。
//例如6、8都是丑数，但14不是，因为它包含质因子7。 
//习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。


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