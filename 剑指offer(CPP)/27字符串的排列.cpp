//����һ���ַ���,���ֵ����ӡ�����ַ������ַ����������С�
//���������ַ���abc,���ӡ�����ַ�a,b,c�������г����������ַ���abc,acb,bac,bca,cab��cba��


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