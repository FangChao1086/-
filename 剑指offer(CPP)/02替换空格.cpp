//��ʵ��һ����������һ���ַ����е�ÿ���ո��滻�ɡ�%20����
//���磬���ַ���ΪWe Are Happy.�򾭹��滻֮����ַ���ΪWe%20Are%20Happy��

class Solution {
public:
	void replaceSpace(char *str,int length) {
		//1���Ȳ��ҿ��ַ�������
		//2���Ӻ���ǰ�滻
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