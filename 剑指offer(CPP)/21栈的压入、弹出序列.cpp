//���������������У���һ�����б�ʾջ��ѹ��˳�����жϵڶ��������Ƿ����Ϊ��ջ�ĵ���˳��
//����ѹ��ջ���������־�����ȡ�
//��������1,2,3,4,5��ĳջ��ѹ��˳������4,5,3,2,1�Ǹ�ѹջ���ж�Ӧ��һ���������У�
//��4,3,5,1,2�Ͳ������Ǹ�ѹջ���еĵ������С���ע�⣺���������еĳ�������ȵģ�


class Solution {
public:
    bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        if(pushV.empty())
            return false;
        vector<int> res;
        for(int i=0,j=0;i<pushV.size();i++){
            res.push_back(pushV[i]);
            while(j<popV.size() && res.back() == popV[j]){
                res.pop_back();
                j++;
            }
        }
        return res.empty();
    }
};