//����һ���������飬ʵ��һ�����������������������ֵ�˳��
//ʹ�����е�����λ�������ǰ�벿�֣����е�ż��λ������ĺ�벿�֣�
//����֤������������ż����ż��֮������λ�ò��䡣


class Solution {
public:
    void reOrderArray(vector<int> &array) {
        //ż��������
        for(int i=0;i<array.size();i++){
            for(int j=0;j<array.size()-i-1;j++){
                if(array[j]%2==0 && array[j+1]%2==1)
                    swap(array[j],array[j+1]);
            }
        }
    }
};