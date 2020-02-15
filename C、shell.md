<span id="back"></span>
# shell
[1、统计词频](#统计词频)  
[2、有效电话号码](#有效电话号码)  

<span id="统计词频"></span>
## [1、统计词频](#back)
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
## [2、有效电话号码](#back)
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
