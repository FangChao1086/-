<span id="re_"></span>
# 多线程_CPP版题目及答案

[1、按序打印](#按序打印)  
[2、交替打印FooBar](#交替打印FooBar)  

<span id="按序打印"></span>
## [1、按序打印](#re_)
```cpp
我们提供了一个类：
public class Foo {
  public void one() { print("one"); }
  public void two() { print("two"); }
  public void three() { print("three"); }
}
三个不同的线程将会共用一个 Foo 实例。
线程 A 将会调用 one() 方法
线程 B 将会调用 two() 方法
线程 C 将会调用 three() 方法
请设计修改程序，以确保 two() 方法在 one() 方法之后被执行，three() 方法在 two() 方法之后被执行。

输入: [1,2,3]
输出: "onetwothree"
解释: 
有三个线程会被异步启动。
输入 [1,2,3] 表示线程 A 将会调用 one() 方法，线程 B 将会调用 two() 方法，线程 C 将会调用 three() 方法。
正确的输出是 "onetwothree"。

输入: [1,3,2]
输出: "onetwothree"
解释: 
输入 [1,3,2] 表示线程 A 将会调用 one() 方法，线程 B 将会调用 three() 方法，线程 C 将会调用 two() 方法。
正确的输出是 "onetwothree"。
 
注意:
尽管输入中的数字似乎暗示了顺序，但是我们并不保证线程在操作系统中的调度顺序。
你看到的输入格式主要是为了确保测试的全面性。

解题思路：
依赖关系可以通过并发机制实现。
使用一个共享变量 firstJobDone 协调第一个方法与第二个方法的执行顺序，
使用另一个共享变量 secondJobDone 协调第二个方法与第三个方法的执行顺序。

#include <semaphore.h>

class Foo {
protected:
    sem_t firstJobDone;
    sem_t secondJobDone;

public:
    Foo() {
        sem_init(&firstJobDone, 0, 0);
        sem_init(&secondJobDone, 0, 0);
    }

    void first(function<void()> printFirst) {
        
        // printFirst() outputs "first". Do not change or remove this line.
        printFirst();
        sem_post(&firstJobDone);
    }

    void second(function<void()> printSecond) {
        
        // printSecond() outputs "second". Do not change or remove this line.
        sem_wait(&firstJobDone);
        printSecond();
        sem_post(&secondJobDone);
    }

    void third(function<void()> printThird) {
        
        // printThird() outputs "third". Do not change or remove this line.
        sem_wait(&secondJobDone);
        printThird();
    }
};
```

<span id="交替打印FooBar"></span>
## [2、交替打印FooBar](#re_)
```cpp
我们提供一个类：
class FooBar {
  public void foo() {
    for (int i = 0; i < n; i++) {
      print("foo");
    }
  }

  public void bar() {
    for (int i = 0; i < n; i++) {
      print("bar");
    }
  }
}
两个不同的线程将会共用一个 FooBar 实例。其中一个线程将会调用 foo() 方法，另一个线程将会调用 bar() 方法。
请设计修改程序，以确保 "foobar" 被输出 n 次。

输入: n = 1
输出: "foobar"
解释: 这里有两个线程被异步启动。其中一个调用 foo() 方法, 另一个调用 bar() 方法，"foobar" 将被输出一次。

输入: n = 2
输出: "foobarfoobar"
解释: "foobar" 将被输出两次。

class FooBar {
private:
    int n;
    mutex foo_mutex;
    mutex bar_mutex;

public:
    FooBar(int n) {
        this->n = n;
        foo_mutex.unlock();
        bar_mutex.lock();
    }

    void foo(function<void()> printFoo) {
        
        for (int i = 0; i < n; i++) {
            foo_mutex.lock();
        	// printFoo() outputs "foo". Do not change or remove this line.
        	printFoo();
            bar_mutex.unlock();
        }
    }

    void bar(function<void()> printBar) {
        
        for (int i = 0; i < n; i++) {
            bar_mutex.lock();
        	// printBar() outputs "bar". Do not change or remove this line.
        	printBar();
            foo_mutex.unlock();
        }
    }
};
```
