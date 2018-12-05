//�������е��������֣����ǰ��һ�����ִ��ں�������֣����������������һ������ԡ�
//����һ������,�����������е�����Ե�����P��
//����P��1000000007ȡģ�Ľ������� �����P%1000000007


class Solution {
public:
    int InversePairs(vector<int> data) {
        if(data.size()<1)
            return 0;
        vector<int> copy;
        for(int i=0;i<int(data.size());i++)
            copy.push_back(data[i]);
        long count=InversePairsCore(data,copy,0,int(data.size())-1);
        return count%1000000007;
    }
    
    long InversePairsCore(vector<int> &data,vector<int>& copy,int start,int end)
    {
        if(start==end)
        {
            copy[start]=data[start];
            return 0;
        }
        int length=(end-start)/2;
        long left=InversePairsCore(copy,data,start,start+length);
        long right=InversePairsCore(copy,data,start+length+1,end);
        
        int i=start+length;
        int j=end;
        int index_copy=end;
        long count=0;
        while(i>=start && j>=start+length+1)
        {
            if(data[i]>data[j])
            {
                copy[index_copy--]=data[i--];
                count=count+j-start-length;
            }
            else{
                copy[index_copy--]=data[j--];
            }
        }
        for(;i>=start;i--)
            copy[index_copy--]=data[i];
        for(;j>=start+length+1;j--)
            copy[index_copy--]=data[j];
        return count+left+right;
    }
};