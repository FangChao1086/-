//LL���������ر��,��Ϊ��ȥ����һ���˿���,���������Ȼ��2������,2��С��(һ����ԭ����54��^_^)...
//��������г����5����,�����Լ�������,�����ܲ��ܳ鵽˳��,����鵽�Ļ�,������ȥ��������Ʊ,�ٺ٣���
//������A,����3,С��,����,��Ƭ5��,��Oh My God!������˳��.....LL��������,��������,
//������\С �����Կ����κ�����,����A����1,JΪ11,QΪ12,KΪ13��
//�����5���ƾͿ��Ա�ɡ�1,2,3,4,5��(��С���ֱ���2��4),��So Lucky!����
//LL����ȥ��������Ʊ��������,Ҫ����ʹ�������ģ������Ĺ���,Ȼ���������LL��������Σ� 
//����������˳�Ӿ����true����������false��Ϊ�˷������,�������Ϊ��С����0��


class Solution {
public:
    bool IsContinuous( vector<int> numbers ) {
        //1�����ظ�����
        //2��num_max-num_min<5;
        int max=0,min=14,len=numbers.size(),count[14]={0};
        if(len==0)
            return 0;
        for(int i=0;i<len;i++){
            count[numbers[i]]++;
            if(numbers[i]==0)
                continue;
            if(count[numbers[i]]>1)
                return 0;
            if(numbers[i]>max)
                max = numbers[i];
            if(numbers[i]<min)
                min = numbers[i];
        }
        if(max-min<5)
            return 1;
        else
            return 0;
    }
};