//ͳ��һ�����������������г��ֵĴ�����


class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        //��������
        /*int count=0;
        for(int i=0;i<data.size();i++)
            if(data[i]==k)
                count++;
        return count;*/
        
        //���ֲ���
        int len=data.size(),mid=-1,start=0,end=data.size()-1,count=0;
        while(start<=end){
            mid=(start+end)>>1;
            if(data[mid]<k)
                start=mid+1;
            if(data[mid]>k)
                end=mid-1;
            if(data[mid]==k)
                break;
        }
        int i=mid;
        while(i>=0 && data[i]==k){
            i--;
            count++;
        }
        i=mid+1;
        while(i<len && data[i]==k){
            i++;
            count++;
        }
        return count;
    }
};