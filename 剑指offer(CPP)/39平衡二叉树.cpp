//输入一棵二叉树，判断该二叉树是否是平衡二叉树。


class Solution {
public:
    bool IsBalanced_Solution(TreeNode* pRoot) {
        //1、树为空是平衡二叉树
        //2、左右树高相差不能超过1
        //3、其子树也要是平衡二叉树
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