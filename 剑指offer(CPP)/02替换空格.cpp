//请实现一个函数，将一个字符串中的每个空格替换成“%20”。
//例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

class Solution {
public:
	void replaceSpace(char *str,int length) {
		//1、先查找空字符串个数
		//2、从后往前替换
		int count=0;
		for(int i=0;i<length;i++){
			if(str[i]==' ')
				count++;
		}
		int len=length+2*count-1;
		for(int i=length-1;i>=0;i--){
			if(str[i]==' '){
				str[len--]='0';
				str[len--]='2';
				str[len--]='%';
			}
			else
				str[len--]=str[i];
		}
	}
};