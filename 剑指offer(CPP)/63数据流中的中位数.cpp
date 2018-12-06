//��εõ�һ���������е���λ����
//������������ж�����������ֵ����ô��λ������������ֵ����֮��λ���м����ֵ��
//������������ж���ż������ֵ����ô��λ������������ֵ����֮���м���������ƽ��ֵ��
//����ʹ��Insert()������ȡ��������ʹ��GetMedian()������ȡ��ǰ��ȡ���ݵ���λ����


/*
���������ݷ�Ϊ����ѣ�Ԫ��ֵС�����ݣ���С���ѣ�Ԫ��ֵ������ݣ�
1��ԭʼ�����ݳ���Ϊż�������ݼ��뵽С���ѣ�����ǰ�µ�����Ҫ�ȼ������ѣ����������ɸѡ�����Ԫ�ؽ���С���ѣ�
   ����ÿ�μ������ݺ�С����ʼ�ձ��������Ǳȴ���Ѵ��Ԫ��
2��ԭʼ�����ݳ���Ϊ�棬�����ݼ������ѣ�ͬ������ǰҪ����С����ɸѡ����СԪ�ؼ������ѣ�������
*/

/*�õ���λ��������������Ϊ�棬��λ����С���ѶѶ���Ԫ��
����������Ϊżʱ����λ��ʱС���ѶѶ������ѶѶ�Ԫ�غ͵�һ��
*/

class Solution {
private:
    vector<int> min;
    vector<int> max;
public:
    void Insert(int num)
    {
        int size=min.size()+max.size();
        if((size&1)==0)    //������ǰ�����ݳ���Ϊż
        {
            if(max.size()>0 && num<max[0])    //�����������С�ڴ�����е����Ԫ��
            {
                max.push_back(num);            //�����ݼ�������
                push_heap(max.begin(),max.end(),less<int>());    //���½��ж����򣬽����Ԫ�ط��ڵ�һ��
                num=max[0];
                pop_heap(max.begin(),max.end(),less<int>());    //���Ѷ�Ԫ�������һ��Ԫ�ػ����������Ѷ�������δɾ��
                max.pop_back();    //ɾ��
            }
            min.push_back(num);
            push_heap(min.begin(),min.end(),greater<int>());
        }
        else
        {
            if(min.size()>0 && num>min[0])
            {
                min.push_back(num);
                push_heap(min.begin(),min.end(),greater<int>());
                num=min[0];
                pop_heap(min.begin(),min.end(),greater<int>());
                min.pop_back();
            }
            max.push_back(num);
            push_heap(max.begin(),max.end(),less<int>());
        }
    }

    double GetMedian()
    {
        int size=min.size()+max.size();
        if((size&1)==0)
            return (min[0]+max[0])/2.0;
        else
            return min[0];
    }

};