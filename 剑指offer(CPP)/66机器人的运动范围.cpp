//地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，
//但是不能进入行坐标和列坐标的数位之和大于k的格子。 
//例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），
//因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？


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