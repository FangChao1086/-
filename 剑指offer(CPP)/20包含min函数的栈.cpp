//定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。


class Solution {
public:
    void push(int value) {
        if(stackMin.empty())
            stackMin.push(value);
        else if(stackMin.top()>value)
            stackMin.push(value);
         else
             stackMin.push(stackMin.top());
		stack1.push(value);
    }
    void pop() {
        if(!stack1.empty()){
            stack1.pop();
            stackMin.pop();
        }
    }
    int top() {
        return stack1.top();
    }
    int min() {
        return stackMin.top();
    }
private:
    stack<int> stack1;
    stack<int> stackMin;
};