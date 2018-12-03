//给定一个double类型的浮点数base和int类型的整数exponent。
//求base的exponent次方。


class Solution {
public:
    double Power(double base, int exponent) {
        //非递归
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