//输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。
//要求不能创建任何新的结点，只能调整树中结点指针的指向。


/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    TreeNode* Convert(TreeNode* pRootOfTree)
    {
        //返回链表的头节点
        if(pRootOfTree == NULL)
            return NULL;
        TreeNode* pre = NULL;
        convertHelper(pRootOfTree,pre);
        TreeNode* res = pRootOfTree;
        while(res->left)
            res = res->left;
        return res;
    }
    
    void convertHelper(TreeNode* cur, TreeNode* &pre){
        if(cur==NULL)
            return ;
        convertHelper(cur->left,pre);
        cur->left=pre;
        if(pre)
            pre->right = cur;
        pre = cur;
        convertHelper(cur->right, pre);
    }
};