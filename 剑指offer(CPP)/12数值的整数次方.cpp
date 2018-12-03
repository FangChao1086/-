//����һ��double���͵ĸ�����base��int���͵�����exponent��
//��base��exponent�η���


class Solution {
public:
    double Power(double base, int exponent) {
        //�ǵݹ�
        int E = abs(exponent);
        double res=1;
        while(E){
            if(E&1)
                res*=base;
            base*=base;
            E>>=1;
        }
        return exponent<0?(1/res):res;
    }
};