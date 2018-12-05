//牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。
//同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。
//例如，“student. a am I”。
//后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。
//Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？


class Solution {
public:
    string ReverseSentence(string str) {
        //1.先将字符串翻转
        //2、将翻转后的字符串，在句末加上一个空格字符，
        //   利用空格字符，将每个单词分开，在进行翻转。得到结果
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
    
    //翻转函数
    void reverseWord(string &word, int start,int end){
        while(start<end){
            swap(word[start],word[end]);
            start++;
            end--;
        }
    }
};