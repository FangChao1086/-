//��Ҷ�֪��쳲��������У�����Ҫ������һ������n��
//�������쳲��������еĵ�n���0��ʼ����0��Ϊ0���� n<=39 


class Solution {
public:
    int Fibonacci(int n) {
		//��̬�滮����
		int f1=0,f2=1;
		while(n--){
			f2=f1+f2;
			f1=f2-f1;
		}
		return f1;
    }
};