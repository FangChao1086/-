//һֻ����һ�ο�������1��̨�ף�Ҳ��������2��������Ҳ��������n����
//�����������һ��n����̨���ܹ��ж�����������


class Solution {
public:
    int jumpFloorII(int number) {
		//f(n-1)=f(1)+f(2)+...+f(n-2)
		//f(n)=f(1)+f(2)+...+f(n-2)+f(n-1)=f(n-1)+f(n-1)
		int total=1;
		for(int i=1;i<number;i++){
			total*=2;
		}
		return total;
    }
};