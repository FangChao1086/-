//输入一个链表，反转链表后，输出新链表的表头。


/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* ReverseList(ListNode* pHead) {
        ListNode* pNode = pHead;
        ListNode* pPre = NULL;
        while(pNode != NULL){
            ListNode* pNext = pNode->next;
            pNode->next = pPre;
            pPre = pNode;
            pNode = pNext;
        }
        return pPre;
    }
};