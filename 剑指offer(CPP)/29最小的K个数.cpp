//����n���������ҳ�������С��K������
//��������4,5,1,6,2,7,3,8��8�����֣�����С��4��������1,2,3,4,��


class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        //ȫ���У�ʱ�临�Ӷ�nlogn
        /*vector<int> res;
        if(k>input.size())
            return res;
        sort(input.begin(),input.end());
        for(int i=0;i<k;i++)
            res.push_back(input[i]);
        return res;*/
        
        //����ʵ�֣�ǰk������������ѣ�ʱ�临�Ӷ�nlogk
        //make_heap,pop_heap,push_heap,sort_heap��ʹ��
        if(k>input.size() || k==0 || input.size()<=0)
            return vector<int> ();
        //����
        vector<int> res(input.begin(),input.begin()+k);
        make_heap(res.begin(),res.begin()+k);
        //��С�������
        for(int i=k;i<input.size();i++){
            if(input[i]<res[0]){
                //ȡ���Ѷ�����ĩβ
                pop_heap(res.begin(),res.end());
                //��β����ֵɾ��
                res.pop_back();
                //���µ��滻ֵ����β��
                res.push_back(input[i]);
                //�����γ�����
                push_heap(res.begin(),res.end());
            }
        }
        sort(res.begin(),res.end());
        return res;
    }
};