//��Ŀ����
//��ʵ��һ�����������ҳ��ַ����е�һ��ֻ����һ�ε��ַ���
//���磬�����ַ�����ֻ����ǰ�����ַ�"go"ʱ����һ��ֻ����һ�ε��ַ���"g"��
//���Ӹ��ַ����ж���ǰ�����ַ���google"ʱ����һ��ֻ����һ�ε��ַ���"l"��
//�������:
//�����ǰ�ַ���û�д��ڳ���һ�ε��ַ�������#�ַ���


class Solution
{
public:
  //Insert one char from stringstream
    void Insert(char ch)
    {
        s+=ch;
        hash[ch]++;
    }
  //return the first appearence once char in current stringstream
    char FirstAppearingOnce()
    {
        int length=s.size();
        for(int i=0;i<length;i++)
            if(hash[s[i]]==1)
                return s[i];
        return '#';
    }

private:
    string s;
    char hash[128]={0};
};