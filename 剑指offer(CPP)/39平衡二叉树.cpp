//����һ�ö��������жϸö������Ƿ���ƽ���������


class Solution {
public:
    bool IsBalanced_Solution(TreeNode* pRoot) {
        //1����Ϊ����ƽ�������
        //2�������������ܳ���1
        //3��������ҲҪ��ƽ�������
        if(!pRoot)
            return true;
        int left = getDepth(pRoot->left);
        int right = getDepth(pRoot->right);
        if(abs(left-right)>1)
            return false;
        return IsBalanced_Solution(pRoot->left) && IsBalanced_Solution(pRoot->right);
    }
    
    int getDepth(TreeNode* pRoot){
        if(!pRoot)
            return 0;
        int left = getDepth(pRoot->left);
        int right = getDepth(pRoot->right);
        return max(left+1,right+1);
    }
};