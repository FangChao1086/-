<span id="re_"></span>
## OJ编程基础
* [输入与输出](#输入与输出)
  * [预先知道数据组数](#预先知道数据组数)
  * [预先不知道数据组数](#预先不知道数据组数)

<span id="输入与输出"></span>
## [输入与输出](#re_)
<span id="预先知道数据组数"></span>
## 预先知道数据组数
```c,c++,java
// C
scanf("%d",&n);
for(int i=0; i<n; i++){
    int a,b;
    scanf("%d%d",&a,&b);
    printf("%d\n",a+b);
}

// C++
cin >> n;
for(int i=0; i<n; i++){
    int a,b;
    cin >> a >> b;
    cout << a+b << endl;
}

// java
Scanner scanner = new Scanner(System.in);
int n = scanner.nextInt();
for(int i=0;i<n;i++){
    int a = scanner.nextInt();
    int b = scanner.nextInt();
    System.out.println(a + b);
}
```

<span id="预先不知道数据组数"></span>
## 预先不知道数据组数
```c,c++,java
// C
while(scanf("%d%d",&a,&b) != EOF){
    printf("%d\n",a+b);
}

// C++
while(cin >> a >> b){
    cout << a+b << endl;
}

// java
Scanner scanner = new Scanner(System.in);
while(scanner.hasNextInt()){
    int a = scanner.nextInt();
    int b = scanner.nextInt();
    System.out.println(a + b);
}
```
