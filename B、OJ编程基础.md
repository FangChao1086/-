<span id="re_"></span>
## OJ编程基础
* [输入](#输入)
  * [只有一组数据](#只有一组数据)
  * [预先知道数据组数](#预先知道数据组数)
  * [预先不输入数据的组数](#预先不输入数据的组数)
* [输出](#输出)
  * [不需要输出case数](#不需要输出case数)
  * [需要输出case数](#需要输出case数)
  * [每个case后有空行](#每个case后有空行)
  * [只有两个case之间有空行](#只有两个case之间有空行)
* [生成数组](#生成数组)

<span id="输入"></span>
## [输入](#re_)
<span id="只有一组数据"></span>
## 只有一组数据
```c,c++,java
//c
scanf("%d%d",&a,&b);
printf("%d\n", a+b);

//c++
cin >> a >> b;
cout << a+b << endl;

//java
int a = scanner.nextInt();
int b = scanner.nextInt();
```

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

<span id="预先不输入数据的组数"></span>
## 预先不输入数据的组数
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

<span id="输出"></span>
## [输出](#re_)
<span id="不需要输出case数"></span>
## 不需要输出case数
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

<span id="需要输出case数"></span>
## 需要输出case数
```c,c++,java
// C
scanf("%d",&n);
for(int i=0; i<n; i++){
    int a,b;
    scanf("%d%d",&a,&b);
    printf("Case %d %d\n",i+1,a+b);
}

// C++
cin >> n;
for(int i=0; i<n; i++){
    int a,b;
    cin >> a >> b;
    cout << "Case " << i+1 << a+b << endl;
}

// java
Scanner scanner = new Scanner(System.in);
int n = scanner.nextInt();
for(int i=0;i<n;i++){
    int a = scanner.nextInt();
    int b = scanner.nextInt();
    System.out.println("Case "+ (i+1)+" "+ (a + b));
}
```

<span id="每个case后有空行"></span>
## 每个case后有空行
```c,c++,java
// C
scanf("%d",&n);
for(int i=0; i<n; i++){
    int a,b;
    scanf("%d%d",&a,&b);
    printf("Case %d %d\n\n",i+1,a+b);
}

// C++
cin >> n;
for(int i=0; i<n; i++){
    int a,b;
    cin >> a >> b;
    cout << "Case " << i+1 << a+b << endl << endl;
}

// java
Scanner scanner = new Scanner(System.in);
int n = scanner.nextInt();
for(int i=0;i<n;i++){
    int a = scanner.nextInt();
    int b = scanner.nextInt();
    System.out.println("Case "+ (i+1)+" "+ (a + b)+"\n");
}
```

<span id="只有两个case之间有空行"></span>
## 只有两个case之间有空行
```c,c++,java
// C
scanf("%d",&n);
for(int i=0; i<n; i++){
    int a,b;
    if(i>0){
        puts("");
    }
    scanf("%d%d",&a,&b);
    printf("Case %d %d",i+1,a+b);
}

// C++
cin >> n;
for(int i=0; i<n; i++){
    int a,b;
    cin >> a >> b;
    if(i>0){
        cout<<endl;
    }
    cout << "Case " << i+1 << a+b << endl;
}

// java
Scanner scanner = new Scanner(System.in);
int n = scanner.nextInt();
for(int i=0;i<n;i++){
    int a = scanner.nextInt();
    int b = scanner.nextInt();
    if(i>0){
        System.out.println();
    }
    System.out.println("Case "+ (i+1)+" "+ (a + b));
}
```

<span id="生成数组"></span>
## [生成数组](#re_)
```cpp
// 二维数组
int dp[n][m];  // 方法1
memset(dp,0,sizeof(dp));

vector<vector<int>> dp(n,vector<int>(m,0));  // 方法2
```

