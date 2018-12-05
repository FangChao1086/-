//汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。
//对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。
//例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。
//是不是很简单？OK，搞定它！


class Solution {
public:
    //字符串翻转
    void fun(string &s, int start,int end){
        while(start<end){
            swap(s[start],s[end]);
            start++;
            end--;
        }
    }
    
    string LeftRotateString(string str, int n) {
        //方法一（优），自定义fun函数用于字符串反转
        int len=str.size();
        if(len==0 || n==0)
            return str;
        string &temp = str;
        fun(temp,0,n-1);
        fun(temp,n,len-1);
        fun(temp,0,len-1);
        return str;
        
        //方法二
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