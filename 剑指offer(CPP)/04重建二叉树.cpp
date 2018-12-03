//输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
//假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
//例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。


/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
		//1、得到中序遍历的根节点的位置
		//2、得到左子树的前序与中序遍历的结果
		//3、得到右子树的前序与中序遍历的结果
		//4、递归得到整个二叉树

		int vinlen=vin.size();
		if(vinlen==0)
			return NULL;
		vector<int> left_pre,left_vin,right_pre,right_vin;
		TreeNode* head=new TreeNode(pre[0]);
		
		//1
		int root_index=0;
		for(int i=0;i<vinlen;i++){
			if(vin[i]==pre[0]){
				root_index=i;
				break;
			}
		}
		
		//2
		for(int i=0;i<root_index;i++){
			left_pre.push_back(pre[i+1]);
			left_vin.push_back(vin[i]);
		}
		
		//3
		for(int i=root_index+1;i<vinlen;i++){
			right_pre.push_back(pre[i]);
			right_vin.push_back(vin[i]);
		}
		
		head->left=reConstructBinaryTree(left_pre,left_vin);
		head->right=reConstructBinaryTree(right_pre,right_vin);
		
		return head;
    }
};