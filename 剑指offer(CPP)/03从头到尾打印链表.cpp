//����һ������������ֵ��β��ͷ��˳�򷵻�һ��ArrayList��


/**
*  struct ListNode {
*        int val;
*        struct ListNode *next;
*        ListNode(int x) :
*              val(x), next(NULL) {
*        }
*  };
*/
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        //���뺯��insert,ÿ����ͷ������
		vector<int> vec;
		while(head!=NULL){
			vec.insert(vec.begin(),head->val);
			head=head->next;
		}
		return vec;
    }
};