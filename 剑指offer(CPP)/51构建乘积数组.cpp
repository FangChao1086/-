//����һ������A[0,1,...,n-1],�빹��һ������B[0,1,...,n-1],
//����B�е�Ԫ��B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]������ʹ�ó�����


class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        /*
        int n=A.size();
        vector<int> B(n,1);
        vector<int> B1(n,1);
        vector<int> B2(n,1);
        for(int i=1;i<n;i++)
            B1[i] = B1[i-1]*A[i-1];
        for(int i=n-2;i>=0;--i)
            B2[i] = B2[i+1]*A[i+1];
        for(int i=0;i<n;i++)
            B[i]=B1[i]*B2[i];
        return B;*/
        
        
        //������
        vector<int> res;
        int n = A.size();
        if(n==0)
            return res;
        res.push_back(1);
        for(int i=1;i<n;i++){
            res.push_back(res.back()*A[i-1]);
        }
        int tmp=1;
        for(int i=n-1;i>=0;i--){
            res[i] = res[i]*tmp;
            tmp = tmp*A[i];
        }
        return res;
    }
};