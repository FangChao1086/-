//��Ŀ����
//С����ϲ����ѧ,��һ����������ѧ��ҵʱ,Ҫ������9~16�ĺ�,�����Ͼ�д������ȷ����100��
//���������������ڴ�,�����뾿���ж������������������еĺ�Ϊ100(���ٰ���������)��
//û���,���͵õ���һ������������Ϊ100������:18,19,20,21,22��
//���ڰ����⽻����,���ܲ���Ҳ�ܿ���ҳ����к�ΪS��������������? Good Luck!
�������:
������к�ΪS�������������С������ڰ��մ�С�����˳�����м䰴�տ�ʼ���ִ�С�����˳��


class Solution {
public:
    vector<vector<int> > FindContinuousSequence(int sum) {
        //˫ָ�����⣬���ܺʹ���sum,��ָ�����ƣ�
        vector<vector<int>> res;
       int left=1,right=1,sum_x=1;
        while(left<=right){
            right++;
            sum_x+=right;
            while(sum_x>sum){
                sum_x-=left;
                left++;
            }
            if(sum_x==sum && left!=right){
                vector<int> tmp;
                for(int i=left;i<=right;i++)
                    tmp.push_back(i);
                res.push_back(tmp);
            }
        }
        return res;
    }
};