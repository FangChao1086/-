//��һ���ַ���ת����һ������(ʵ��Integer.valueOf(string)�Ĺ��ܣ�
//����string����������Ҫ��ʱ����0)��Ҫ����ʹ���ַ���ת�������Ŀ⺯���� 
//��ֵΪ0�����ַ�������һ���Ϸ�����ֵ�򷵻�0��

/*
//��������:
����һ���ַ���,����������ĸ����,����Ϊ��
//�������:
����ǺϷ�����ֵ����򷵻ظ����֣����򷵻�0

ʾ��1 
����
+2147483647
    1a33
	
���
2147483647
    0
*/


class Solution {
public:
    int StrToInt(string str) {
        long long res = 0;
        int len = str.size(),s=1;
        if(len==0)
            return 0;
        if(str[0]=='-')
            s=-1;
        for(int i = ((str[0]=='+' || str[0]=='-')?1:0);i<len;i++){
            if(str[i]<'0' || str[i]>'9')
                return 0;
            res = (res<<1)+(res<<3)+(str[i]&0xf);
        }
        return res*s;
    }
};