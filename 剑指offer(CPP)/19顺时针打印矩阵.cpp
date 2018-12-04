//����һ�����󣬰��մ���������˳ʱ���˳�����δ�ӡ��ÿһ�����֣�
//���磬�����������4 X 4���� 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 
//�����δ�ӡ������1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.


class Solution {
public:
    vector<int> printMatrix(vector<vector<int> > matrix) {
        int row = matrix.size();
        int col = matrix[0].size();
        vector<int> res;
        int left = 0,right = col-1,top = 0,bottom = row-1;
        while(left<=right && top<=bottom){
            for(int i=left;i<=right;i++)
                res.push_back(matrix[top][i]);
            if(top<bottom)
                for(int i=top+1;i<=bottom;i++)
                    res.push_back(matrix[i][right]);
            if(top<bottom && left<right)
                for(int i=right-1;i>=left;i--)
                    res.push_back(matrix[bottom][i]);
            if(top+1<bottom && left<right)
                for(int i=bottom-1;i>=top+1;i--)
                    res.push_back(matrix[i][left]);
            left++,right--,top++,bottom--;
        }
        return res;
    }
};