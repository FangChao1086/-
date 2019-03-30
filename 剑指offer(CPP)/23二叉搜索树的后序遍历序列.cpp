//输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
//如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。


class Solution {
public:
    bool VerifySquenceOfBST(vector<int> sequence) {
        return bst(sequence, 0, sequence.size()-1);
    }
    
    bool bst(vector<int> sequence, int begin, int end){
        if(sequence.empty())
            return false;
        //根节点
        int root = sequence[end];
        //找到第一个右子树的节点
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
