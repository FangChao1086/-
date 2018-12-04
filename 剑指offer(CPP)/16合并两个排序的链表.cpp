//���������������������������������ϳɺ��������Ȼ������Ҫ�ϳɺ���������㵥����������


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
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        //�ݹ�
        /*if(pHead1==NULL)
            return pHead2;
        else if(pHead2==NULL)
            return pHead1;
        ListNode* p=NULL;
        if(pHead1->val<pHead2->val){
            p = pHead1;
            p->next = Merge(pHead1->next,pHead2);
        }
        else{
            p = pHead2;
            p->next = Merge(pHead1,pHead2->next);
        }
        return p;*/
        
        //�ǵݹ�
        if(pHead1==NULL)
            return pHead2;
        else if(pHead2==NULL)
            return pHead1;
        ListNode* head;
        if(pHead1->val<pHead2->val){
            head = pHead1;
            pHead1 = pHead1->next;
        }
        else{
            head = pHead2;
            pHead2 = pHead2->next;
        }
        ListNode* p = head;
        while(pHead1&&pHead2){
            if(pHead1->val<pHead2->val){
                p->next = pHead1;
                pHead1 = pHead1->next;
                p = p->next;
            }
            else{
                p->next = pHead2;
                pHead2 = pHead2->next;
                p = p->next;
            }
        }
        if(pHead1==NULL)
            p->next = pHead2;
        if(pHead2 == NULL)
            p->next = pHead1;
        return head;
    }
};