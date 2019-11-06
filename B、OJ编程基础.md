<span id="re_"></span>
## OJ编程基础
* [输入与输出](#输入与输出)
  * [预先知道数据组数](#预先知道数据组数)
  * [预先不知道数据组数](#预先不知道数据组数)
* [生成数组](#生成数组)
* [数据类型转换](#数据类型转换)

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

<span id="生成数组"></span>
## [生成数组](#re_)
```cpp
// C++

// 二维数组
int dp[n][m];  // 方法1
memset(dp,0,sizeof(dp));  // 用0填充

vector<vector<int>> dp(n,vector<int>(m,0));  // 方法2；n*m填充0

vector<int> res(input.begin(), input.begin() + k);  // 方法3；input是已经存在的vector
```

<span id="数据类型转换"></span>
## [数据类型转换](#re_)
* char转string
  ```cpp
  // 假设mp[0]是char;
  string str;
  str.push_back(mp[0])  // str变成了string类型
  ```
