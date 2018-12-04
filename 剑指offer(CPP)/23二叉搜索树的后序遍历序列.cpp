//����һ���������飬�жϸ������ǲ���ĳ�����������ĺ�������Ľ����
//����������Yes,�������No���������������������������ֶ�������ͬ��


class Solution {
public:
    bool VerifySquenceOfBST(vector<int> sequence) {
        return bst(sequence, 0, sequence.size()-1);
    }
    
    bool bst(vector<int> sequence, int begin, int end){
        if(sequence.empty() || begin>end)
            return false;
        //���ڵ�
        int root = sequence[end];
        //�ҵ���һ���������Ľڵ�
        int i = begin;
        for(;i<end;i++)
            if(sequence[i]>root)
                break;
        for(int j=i;j<end;j++)
            if(sequence[j]<root)
                return false;
        bool left = true;
        bool right = true;
        if(i>begin)
            left = bst(sequence, begin, i-1);
        if(i<end)
            right = bst(sequence, i, end-1);
        return left && right;
    }
};