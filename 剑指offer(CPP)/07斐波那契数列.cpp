//大家都知道斐波那契数列，现在要求输入一个整数n，
//请你输出斐波那契数列的第n项（从0开始，第0项为0）。 n<=39 


class Solution {
public:
    int Fibonacci(int n) {
		//动态规划问题
		int f1=0,f2=1;
		while(n--){
			f2=f1+f2;
			f1=f2-f1;
		}
		return f1;
    }
};