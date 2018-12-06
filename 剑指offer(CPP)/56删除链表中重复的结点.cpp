//在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 
//例如，链表1->2->3->3->4->4->5 处理后为 1->2->5


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
    ListNode* deleteDuplication(ListNode* pHead)
    {
        if(pHead==NULL)
            return NULL;
        ListNode* p=pHead;
        ListNode* q = NULL;
        ListNode* pre = NULL;
        while(p!=NULL){
            if(p->next!=NULL && p->val==p->next->val){
                q = p->next;
                while(q->next!=NULL && q->next->val==p->val)
                    q = q->next;
                if(p==pHead)
                    pHead=q->next;
                else
                    pre->next = q->next;
                p=q->next;
            }
            else{
                pre=p;
                p = p->next;
            }
        }
        return pHead;
    }
};