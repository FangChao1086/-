//��������һ�����ֳ��ֵĴ����������鳤�ȵ�һ�룬���ҳ�������֡�
//��������һ������Ϊ9������{1,2,3,2,2,2,5,4,2}��
//��������2�������г�����5�Σ��������鳤�ȵ�һ�룬������2����������������0��


class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        //����������ڷ���������������һ���������м���Ǹ���
        //ʹ��sort,ʱ�临�Ӷ�Ϊnlogn,������
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