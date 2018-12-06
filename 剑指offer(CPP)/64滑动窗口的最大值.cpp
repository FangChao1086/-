//����һ������ͻ������ڵĴ�С���ҳ����л�����������ֵ�����ֵ��
//���磬�����������{2,3,4,2,6,2,5,1}���������ڵĴ�С3��
//��ôһ������6���������ڣ����ǵ����ֵ�ֱ�Ϊ{4,4,6,6,6,5}�� 
//�������{2,3,4,2,6,2,5,1}�Ļ�������������6���� 
//{[2,3,4],2,6,2,5,1}�� {2,[3,4,2],6,2,5,1}�� {2,3,[4,2,6],2,5,1}�� 
//{2,3,4,[2,6,2],5,1}�� {2,3,4,2,[6,2,5],1}�� {2,3,4,2,6,[2,5,1]}��


class Solution {
public:
    vector<int> maxInWindows(const vector<int>& num, unsigned int size)
    {
        //����һ
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
        
        //��������
        vector<int> res;    //�����ֵ
        deque<int> s;    //���±�
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