//输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），
//返回结果为复制后复杂链表的head。
//（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）


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
        
        //复制节点,插入链表
        RandomListNode* currNode = pHead;
        while(currNode){
            RandomListNode* newNode = new RandomListNode(currNode->label);
            newNode->next = currNode->next;
            currNode->next=newNode;
            currNode = newNode->next;
        }
        
        //复制随机指针
        currNode = pHead;
        while(currNode){
            RandomListNode* newNode = currNode->next;
            if(currNode->random)
                newNode->random = currNode->random->next;
            currNode = newNode->next;
        }
        
        //拆分
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