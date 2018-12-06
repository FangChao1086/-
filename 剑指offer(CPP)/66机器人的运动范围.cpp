//������һ��m�к�n�еķ���һ�������˴�����0,0�ĸ��ӿ�ʼ�ƶ���ÿһ��ֻ�������ң��ϣ����ĸ������ƶ�һ��
//���ǲ��ܽ�������������������λ֮�ʹ���k�ĸ��ӡ� 
//���磬��kΪ18ʱ���������ܹ����뷽��35,37������Ϊ3+5+3+7 = 18�����ǣ������ܽ��뷽��35,38����
//��Ϊ3+5+3+8 = 19�����ʸû������ܹ��ﵽ���ٸ����ӣ�


class Solution {
public:
    int movingCount(int threshold, int rows, int cols)
    {
        bool *flags = new bool[rows*cols];
        for(int i=0;i<rows*cols;i++)
            flags[i]=false;
        int count = moving(flags,rows,cols,threshold,0,0);
        return count;
    }
    
    int sumOfNum(int m){
        int sum=0;
        while(m){
            sum+=m%10;
            m=m/10;
        }
        return sum;
    }
    
    int moving(bool *flags,int rows, int cols,int threshold,int i,int j){
        int count=0;
        if(i>=0 && i<rows && j>=0 && j<cols && flags[i*cols+j]==false && sumOfNum(i)+sumOfNum(j)<=threshold){
            flags[i*cols+j]=true;
            count =1+moving(flags,rows,cols,threshold,i-1,j)
                + moving(flags,rows,cols,threshold,i+1,j)
                + moving(flags,rows,cols,threshold,i,j-1)
                + moving(flags,rows,cols,threshold,i,j+1);
        }
        return count;
    }
};