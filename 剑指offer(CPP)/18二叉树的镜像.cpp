//���������Ķ�����������任ΪԴ�������ľ���

/*����������
�������ľ����壺Դ������ 
    	    8
    	   /  \
    	  6   10
    	 / \  / \
    	5  7 9 11
    	���������
    	    8
    	   /  \
    	  10   6
    	 / \  / \
    	11 9 7  5
*/


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
    void Mirror(TreeNode *pRoot) {
        //�ݹ�
        /*if(pRoot == NULL)
            return ;
        TreeNode *pRight = pRoot->right;
        pRoot->right = pRoot->left;
        pRoot->left = pRight;
        Mirror(pRoot->left);
        Mirror(pRoot->right);*/
        
        //�ǵݹ�
        if(pRoot == NULL)
            return ;
        stack<TreeNode*> stackNode;
        stackNode.push(pRoot);
        while(stackNode.size()){
            TreeNode *tree = stackNode.top();
            stackNode.pop();
            if(tree->left!=NULL || tree->right!=NULL){
                TreeNode *temp = tree->right;
                tree->right = tree->left;
                tree->left = temp;
            }
            if(tree->left)
                stackNode.push(tree->left);
            if(tree->right)
                stackNode.push(tree->right);
        }
    }
};