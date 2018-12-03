//我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。
//请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？


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