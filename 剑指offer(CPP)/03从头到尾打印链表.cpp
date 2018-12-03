//输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。


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
        //插入函数insert,每次往头部插入
		vector<int> vec;
		while(head!=NULL){
			vec.insert(vec.begin(),head->val);
			head=head->next;
		}
		return vec;
    }
};