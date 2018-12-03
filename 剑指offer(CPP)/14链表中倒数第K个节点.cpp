//输入一个链表，输出该链表中倒数第k个结点。


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
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        ListNode* list1 = pListHead;
        ListNode* list2 = pListHead;
        int i;
        for(i=0;list1!=NULL;i++){
            if(i>=k)
                list2 = list2->next;
            list1 = list1->next;
        }
        return (i>=k)?list2:NULL;
    }
};