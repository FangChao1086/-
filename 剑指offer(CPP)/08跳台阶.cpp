//һֻ����һ�ο�������1��̨�ף�Ҳ��������2����
//�����������һ��n����̨���ܹ��ж������������Ⱥ����ͬ�㲻ͬ�Ľ������


class Solution {
public:
    int jumpFloor(int number) {
        //��̬�滮
		int f1=1,f2=1;
		while(number--){
			f2=f1+f2;
			f1=f2-f1;
		}
		return f1;
    }
};