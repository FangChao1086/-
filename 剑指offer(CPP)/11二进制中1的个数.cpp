//����һ��������������������Ʊ�ʾ��1�ĸ��������и����ò����ʾ��


class Solution {
public:
     int  NumberOf1(int n) {
         //���Ž� �����������
         int count=0;
         while(n!=0){
             count++;
             n&=(n-1);
         }
         return count;
     }
};