//输入一个字符串,按字典序打印出该字符串中字符的所有排列。
//例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。


class Solution {
public:
    vector<string> Permutation(string str) {
        vector<string> res;
        if(str.size()==0)
            return res;
        sort(str.begin(),str.end());
        do{
            res.push_back(str);
        }
        while(next_permutation(str.begin(),str.end()));
        return res;
    }
};