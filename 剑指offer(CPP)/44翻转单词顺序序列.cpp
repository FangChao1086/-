//ţ���������һ����Ա��Fish��ÿ���糿���ǻ�����һ��Ӣ����־��дЩ�����ڱ����ϡ�
//ͬ��Cat��Fishд�������ĸ���Ȥ����һ������Fish������������ȴ������������˼��
//���磬��student. a am I����
//��������ʶ������һ�ԭ���Ѿ��ӵ��ʵ�˳��ת�ˣ���ȷ�ľ���Ӧ���ǡ�I am a student.����
//Cat��һһ�ķ�ת��Щ����˳��ɲ����У����ܰ�����ô��


class Solution {
public:
    string ReverseSentence(string str) {
        //1.�Ƚ��ַ�����ת
        //2������ת����ַ������ھ�ĩ����һ���ո��ַ���
        //   ���ÿո��ַ�����ÿ�����ʷֿ����ڽ��з�ת���õ����
        if(str.size()==0)
            return str;
        reverseWord(str,0,str.size()-1);
        str+=' ';
        int mark=0;
        for(int i=0;i<int(str.size());i++){
            if(str[i]==' '){
                reverseWord(str,mark,i-1);
                mark = i+1;
            }
        }
        str = str.substr(0,str.size()-1);
        return str;
    }
    
    //��ת����
    void reverseWord(string &word, int start,int end){
        while(start<end){
            swap(word[start],word[end]);
            start++;
            end--;
        }
    }
};