//��һ�������ʼ�����ɸ�Ԫ�ذᵽ�����ĩβ�����ǳ�֮Ϊ�������ת�� 
//����һ���Ǽ�����������һ����ת�������ת�������СԪ�ء� 
//��������{3,4,5,1,2}Ϊ{1,2,3,4,5}��һ����ת�����������СֵΪ1�� 
//NOTE������������Ԫ�ض�����0���������СΪ0���뷵��0��


class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        //���ֲ���
		int len=rotateArray.size();
		int left=0,mid=0,right=len-1;
		while(rotateArray[left]>=rotateArray[right]){
			if(right-left==1){
				mid=right;
				break;
			}
			mid=left+(right-left)/2;
			if(rotateArray[mid]>=rotateArray[left])
				left=mid;
			if(rotateArray[mid]<=rotateArray[right])
				right=mid;
		}
		return rotateArray[mid];
    }
};