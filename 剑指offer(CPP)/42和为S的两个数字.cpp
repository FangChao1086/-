//��Ŀ����
//����һ����������������һ������S���������в�����������ʹ�����ǵĺ�������S��
//����ж�����ֵĺ͵���S������������ĳ˻���С�ġ�
//�������:
//��Ӧÿ�����԰����������������С���������


class Solution {
public:
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        //������������飬���˵�����˻�С�ڽ��м�������
        int left = 0,right = array.size()-1,sum_x = 0;
        vector<int> res;
        while(left<right){
            sum_x = array[left]+array[right];
            if(sum_x>sum)
                right--;
            if(sum_x<sum)
                left++;
            if(sum_x==sum){
                res.push_back(array[left]);
                res.push_back(array[right]);
                break;
            }
        }
        return res;
    }
};