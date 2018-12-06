//给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。


/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead)
    {
        if(pHead==NULL || pHead->next==NULL)
            return NULL;
        ListNode* p1 = pHead;
        ListNode* p2 = pHead;
        while(p2!=NULL && p2->next!=NULL){
            p1 = p1->next;
            p2 = p2->next->next;
            if(p1==p2){
                p2 = pHead;
                while(p1!=p2){
                    p1=p1->next;
                    p2=p2->next;
                }
                if(p1==p2)
                    return p1;
            }
        }
        return NULL;
    }
};