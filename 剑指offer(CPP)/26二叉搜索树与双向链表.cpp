//����һ�ö��������������ö���������ת����һ�������˫������
//Ҫ���ܴ����κ��µĽ�㣬ֻ�ܵ������н��ָ���ָ��


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
        //���������ͷ�ڵ�
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