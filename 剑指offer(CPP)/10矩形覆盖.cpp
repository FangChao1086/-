//���ǿ�����2*1��С���κ��Ż�������ȥ���Ǹ���ľ��Ρ�
//������n��2*1��С�������ص��ظ���һ��2*n�Ĵ���Σ��ܹ��ж����ַ�����


class Solution {
public:
    int rectCover(int number) {
		int f1=1,f2=1;
		if(number<=0)
			return 0;
		while(number--){
			f2+=f1;
			f1=f2-f1;
		}
		return f1;
    }
};