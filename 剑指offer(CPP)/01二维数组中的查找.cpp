//��һ����ά�����У�ÿ��һά����ĳ�����ͬ����
//ÿһ�ж����մ����ҵ�����˳������ÿһ�ж����մ��ϵ��µ�����˳������
//�����һ������������������һ����ά�����һ���������ж��������Ƿ��и�������

class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        //�����ϵ����¿�ʼ����
		int row=array.size();
		int col=array[0].size();
		int i=0,j=col-1;
		while(i<row && j>-1){
			if(array[i][j]<target)
				i++;
			else if(array[i][j]>target)
				j--;
			else
				return true;
		}
		return false;
    }
};