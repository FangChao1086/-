//请实现两个函数，分别用来序列化和反序列化二叉树


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
    vector<int> buffer;
    void dfs(TreeNode* root){
        if(root==NULL)
            buffer.push_back(0xFFFFFFFF);
        else{
            buffer.push_back(root->val);
            dfs(root->left);
            dfs(root->right);
        }
    }
    
    TreeNode* dfs2(int* &p){
        if(*p==0xFFFFFFFF){
            p++;
            return NULL;
        }
        TreeNode* res = new TreeNode(*p);
        p++;
        res->left = dfs2(p);
        res->right = dfs2(p);
        return res;
    }
    
    char* Serialize(TreeNode *root) {    
        buffer.clear();
        dfs(root);
        int size = buffer.size();
        int *res = new int[size];
        for(int i=0;i<size;i++)
            res[i]=buffer[i];
        return (char*) res;
    }
    TreeNode* Deserialize(char *str) {
        int *p =(int*)str;
        return dfs2(p);
    }
};