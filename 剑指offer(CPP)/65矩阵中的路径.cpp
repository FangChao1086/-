//�����һ�������������ж���һ���������Ƿ����һ������ĳ�ַ��������ַ���·����
//·�����ԴӾ����е�����һ�����ӿ�ʼ��ÿһ�������ھ������������ң����ϣ������ƶ�һ�����ӡ�
//���һ��·�������˾����е�ĳһ�����ӣ���֮�����ٴν���������ӡ� 
//���� a b c e s f c s a d e e ������3 X 4 �����а���һ���ַ���"bcced"��·����
//���Ǿ����в�����"abcb"·������Ϊ�ַ����ĵ�һ���ַ�bռ���˾����еĵ�һ�еڶ�������֮��
//·�������ٴν���ø��ӡ�


class Solution {
public:
    bool isPath(vector<char> flags,char* matrix,int rows,int cols,char* str,int i,int j){
        if(i<0 || i>=rows || j<0 || j>=cols)
            return false;
        if(matrix[i*cols+j]==*str && flags[i*cols+j]==0){
            flags[i*cols+j]=1;
            if(*(str+1)==0)
                return true;
            bool condition = isPath(flags,matrix,rows,cols,str+1,i-1,j)
                || isPath(flags,matrix,rows,cols,str+1,i+1,j)
                || isPath(flags,matrix,rows,cols,str+1,i,j-1)
                || isPath(flags,matrix,rows,cols,str+1,i,j+1);
            if(condition == false)
                flags[i*cols+j]=0;
            return condition;
        }
        else
            return false;
    }
    
    
    bool hasPath(char* matrix, int rows, int cols, char* str)
    {
        vector<char> flags(rows*cols,0);
        bool condition =false;
        for(int i = 0;i<rows;i++){
            for(int j=0;j<cols;j++){
                if(isPath(flags,matrix,rows,cols,str,i,j)){
                    condition = true;
                    break;
                }
            }
        }
        return condition;
    }
};