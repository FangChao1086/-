//�������´�ӡ����������ÿ���ڵ㣬ͬ��ڵ�������Ҵ�ӡ��


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
    vector<int> PrintFromTopToBottom(TreeNode* root) {
        queue<TreeNode *> q;
        vector<int> r;
        q.push(root);
        while(!q.empty()){
            root=q.front();
            q.pop();
            if(!root)
                continue;
            r.push_back(root->val);
            q.push(root->left);
            q.push(root->right);
        }
        return r;
    }
};