//����һ�ö��������������ҳ����еĵ�kС�Ľ�㡣
//���磬��5��3��7��2��4��6��8���У��������ֵ��С˳�����С����ֵΪ4��


/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    int count = 0;
    TreeNode* KthNode(TreeNode* pRoot, int k)
    {
        if(pRoot!=NULL){
            TreeNode* node = KthNode(pRoot->left,k);
            if(node!=NULL)
                return node;
            count++;
            if(count == k)
                return pRoot;
            node = KthNode(pRoot->right,k);
            if(node != NULL)
                return node;
        }
        return NULL;
    }
};