//����һ����������ÿ���ڵ����нڵ�ֵ���Լ�����ָ�룬һ��ָ����һ���ڵ㣬��һ������ָ��ָ������һ���ڵ㣩��
//���ؽ��Ϊ���ƺ��������head��
//��ע�⣬���������벻Ҫ���ز����еĽڵ����ã�������������ֱ�ӷ��ؿգ�


/*
struct RandomListNode {
    int label;
    struct RandomListNode *next, *random;
    RandomListNode(int x) :
            label(x), next(NULL), random(NULL) {
    }
};
*/
class Solution {
public:
    RandomListNode* Clone(RandomListNode* pHead)
    {
        if(pHead==NULL)
            return NULL;
        
        //���ƽڵ�,��������
        RandomListNode* currNode = pHead;
        while(currNode){
            RandomListNode* newNode = new RandomListNode(currNode->label);
            newNode->next = currNode->next;
            currNode->next=newNode;
            currNode = newNode->next;
        }
        
        //�������ָ��
        currNode = pHead;
        while(currNode){
            RandomListNode* newNode = currNode->next;
            if(currNode->random)
                newNode->random = currNode->random->next;
            currNode = newNode->next;
        }
        
        //���
        currNode = pHead;
        RandomListNode* pClone = pHead->next;
        while(currNode->next){
            RandomListNode* newNode = currNode->next;
            currNode->next = newNode->next;
            currNode = newNode;
        }
        return pClone;
    }
};