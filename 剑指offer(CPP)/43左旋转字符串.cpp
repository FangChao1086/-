//�����������һ����λָ�����ѭ�����ƣ�ROL���������и��򵥵����񣬾������ַ���ģ�����ָ�����������
//����һ���������ַ�����S���������ѭ������Kλ������������
//���磬�ַ�����S=��abcXYZdef��,Ҫ�����ѭ������3λ��Ľ��������XYZdefabc����
//�ǲ��Ǻܼ򵥣�OK���㶨����


class Solution {
public:
    //�ַ�����ת
    void fun(string &s, int start,int end){
        while(start<end){
            swap(s[start],s[end]);
            start++;
            end--;
        }
    }
    
    string LeftRotateString(string str, int n) {
        //����һ���ţ����Զ���fun���������ַ�����ת
        int len=str.size();
        if(len==0 || n==0)
            return str;
        string &temp = str;
        fun(temp,0,n-1);
        fun(temp,n,len-1);
        fun(temp,0,len-1);
        return str;
        
        //������
        /*if(str.size()==0)
            return "";
        int length=str.size();
        n=n%length;
        string A;
        for (int i=0,j=length-n,z=0;i<length;++i)
        {
            if(i<n)
            {
                A[j]=str[i];
                j++;
            }
            else
            {
                A[z]=str[i];
                z++;
            }
        }
        return A;*/
    }
};