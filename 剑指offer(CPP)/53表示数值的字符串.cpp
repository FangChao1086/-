//��ʵ��һ�����������ж��ַ����Ƿ��ʾ��ֵ������������С������
//���磬�ַ���"+100","5e2","-123","3.1416"��"-1E-16"����ʾ��ֵ�� 
//����"12e","1a3.14","1.2.3","+-5"��"12e+4.3"�����ǡ�


class Solution {
public:
    bool isNumeric(char* string)
   {
        if(*string=='+'||*string=='-')
            string++;
        if(*string=='\0')
            return false;
        int dot=0,num=0,nume=0;//�ֱ��������С���㡢�������ֺ�eָ���Ƿ����
        while(*string!='\0'){
            if(*string>='0'&&*string<='9')
            {   
                string++;
                num=1;
            }
            else if(*string=='.'){
                if(dot>0||nume>0)
                    return false;
                string++;
                dot=1;
            }
            else if(*string == 'e' || *string == 'E'){
                if(nume>0 || num==0)
                    return false;
                string++;
                nume++;
                if(*string=='+' || *string=='-')
                    string++;
                if(*string=='\0')
                    return false;
            }
            else
                return false;
        }
        return true;
    }
};