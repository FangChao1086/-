<span id="back"></span>
# shell
[查找文件中一个字符出现次数](#查找文件中一个字符出现次数)  
[统计词频](#统计词频)  
[有效电话号码](#有效电话号码)  
[转置文件](#转置文件)  
[第十行](#第十行)  

<span id="查找文件中一个字符出现次数"></span>
## [查找文件中一个字符出现次数](#back)
```shell
比如要在/tmp/1.sh文件中找a这个字符的个数
grep -o 'a' /tmp/1.sh | wc -l
```

<span id="统计词频"></span>
## [统计词频](#back)
```shell
写一个 bash 脚本以统计一个文本文件 words.txt 中每个单词出现的频率。
为了简单起见，你可以假设：
 words.txt 只包括小写字母和 ' ' 。
每个单词只由小写字母组成。
单词间由一个或多个空格字符分隔。

假设 words.txt 内容如下：
the day is sunny the the
the sunny is is
你的脚本应当输出（以词频降序排列）：
the 4
is 3
sunny 2
day 1

说明:
不要担心词频相同的单词的排序问题，每个单词出现的频率都是唯一的。
你可以使用一行 Unix pipes 实现吗？

解答：
本题先用 cat 命令和管道命令 | 将文件内容传给 awk 。
在 awk 中我们用一个字典 count 储存每个单词的词频，
先遍历每一行( awk 自身机制)的每一个字段 (i<=NF)，
然后用该字段本身作为 key ,将其 value++；
最后用一个 for 循环输出 count 数组中的每个元素的 key (词)及其 value (词频)。
最后用 | 管道命令传给 sort 命令：
 -r 是倒序排序，相当于DESC
 -n 是将字符串当作numeric数值排序
 -k 则指定用于排序的字段位置，后跟 2 指将第二位的 count[k] (词频)作为排序的 key

cat words.txt | 
awk '{ 
    for(i=1;i<=NF;i++){
        count[$i]++
    } 
} END { 
    for(k in count){
        print k" "count[k]
    } 
}' | 
sort -rnk 2
```

<span id="有效电话号码"></span>
## [有效电话号码](#back)
```shell
给定一个包含电话号码列表（一行一个电话号码）的文本文件 file.txt，写一个 bash 脚本输出所有有效的电话号码。
你可以假设一个有效的电话号码必须满足以下两种格式： (xxx) xxx-xxxx 或 xxx-xxx-xxxx。（x 表示一个数字）
你也可以假设每行前后没有多余的空格字符。

假设 file.txt 内容如下：
987-123-4567
123 456 7890
(123) 456-7890
你的脚本应当输出下列有效的电话号码：
987-123-4567
(123) 456-7890

# Read from the file file.txt and output all valid phone numbers to stdout.
grep -P '^([0-9]{3}-|\([0-9]{3}\) )[0-9]{3}-[0-9]{4}$' file.txt
```

<span id="转置文件"></span>
## [转置文件](#back)
```shell
给定一个文件 file.txt，转置它的内容。
你可以假设每行列数相同，并且每个字段由 ' ' 分隔.

假设 file.txt 文件内容如下：
name age
alice 21
ryan 30
应当输出：
name alice ryan
age 21 30

解答：
 awk 常量：NF 是当前行的 field 字段数；NR 是正在处理的当前行数。
数组res来储存新文本，将新文本的每一行存为数组res的一个元素。
在 END 之前我们遍历 file.txt 的每一行，并做一个判断：
在第一行时，每碰到一个字段就将其按顺序放在res数组中；
从第二行开始起，每碰到一个字段就将其追加到对应元素的末尾（中间添加一个空格）。
文本处理完了，最后需要输出。
在 END 后遍历数组，输出每一行。
注意printf不会自动换行，而print会自动换行。

# Read from the file file.txt and print its transposed content to stdout.
awk '{
    for(i=1;i<=NF;i++){
        if (NR==1) {
            res[i]=$i;
        }
        else {
            res[i]=res[i]" "$i
        }
    }
}END{
    for(j=1;j<=NF;j++){
        print res[j]
    }
}' file.txt
```

<span id="第十行"></span>
## [第十行](#back)
```shell
给定一个文本文件 file.txt，请只打印这个文件中的第十行。

假设 file.txt 有如下内容：
Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
Line 10
你的脚本应当显示第十行：
Line 10
说明:
1. 如果文件少于十行，你应当输出什么？
2. 至少有三种不同的解法，请尝试尽可能多的方法来解题。

# Read from the file file.txt and output the tenth line to stdout.
sed -n '10p' file.txt
awk 'NR==10' file.txt
head -10 file.txt | tail -1
```
