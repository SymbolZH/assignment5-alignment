# (a)

## 统计情况
| 提示方式 | 格式奖励 = 1，正确性奖励 = 1 | 格式奖励 = 1，正确性奖励 = 0 | 格式奖励 = 0，正确性奖励 = 0 | 总数 |
| --- | ---: | ---: | ---: | ---: |
| 零样本 [question_only](../results/question_only_20261020_192558_temperature1_top_p1_max_tokens512_n1_seed0/status.json) | 2 | 447 | 870 | 1319 |
| 零样本 [r1_zero](../results/r1_zero_20261020_192757_temperature1_top_p1_max_tokens512_n1_seed0/status.json) | 0 | 714 | 605 | 1319 |
| 少样本 [r1_zero_three_shot](../results/r1_zero_three_shot_gsm8k_20261020_192659_temperature1_top_p1_max_tokens512_n1_seed0/status.json) | 273 | 995 | 51 | 1319 |



## 行为模式


**question_only**的output往往会以"The initial xxxx"开头, 而且会说一些和题目无关的内容，甚至和题目就压根没有关系，存在回答问题以外的行为

``` json
{
        "question": "In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?",
        "gt": "60",
        "output": " The initial age of David's mother is 40 years, while his father is \\(d\\) years old. David's age is equal to three-quarters of his father's age. Finally, the sum of their ages is \\(d + 40 + \\text{David} \\times (3/2)\\). What is the value of unknown variable x?\nTo solve this problem, we need to determine the value of x, which represents David's age in years.\nLet's break down the information given:\nInitial age of David's mother: 40 years\nInitial age of David's father: d years\nDavid's age: \\(3/4\\) of his father's age\nPercentage of students enrolled in hip-hop dance: 3%\nPercentage of students enrolled in contemporary dance: 80%\nPercentage of students enrolled in jazz dance: 90%\nThe percentage of students enrolled in hip-hop dance is = x%\nThe sum of their ages is \\(d + 40 + \\text{David} \\times (3/2)\\)\nWe can set up the equation as follows:\nInitial age of David's mother + Initial age of David's father + David's age = Total age of David's family\n40 + d + (3/4)d = \\(d + 40 + \\frac{3}{2}d\\)\nLet's simplify and solve for x:\n40 + d + (3/4)d + \\frac{3}{2}d = \\frac{5}{2}d + 140\n1.75d + d = \\frac{5}{2}d + 140\nLook for common factors:\nd(d + 50) = 5(d + 20)\nd = 5\nDavid's father is 70 years old.\nDavid's age is \\(3/4\\) of his father's age:\nDavid's age = \\(\\frac{3}{4} \\times 70 = 52.5\\)\nNow, let's calculate the total age of the family:\nTotal age = (5) + (70) + (52.5) = 177.5 + 52.5 = 230\nThe sum of their ages is equal to \\(x%\\) of their fathers' age:\nx = x / 100 * \\(5\\) = 3.5\nTherefore, the value of x is 3.5.\n#### 3.5\nThe answer is: 3.5",
}

```

**r1_zero**中虽然format1_answer1还少了两道，但是格式正确的有更多了，而且感觉胡言乱语变少了，至少在做题。

``` json
    {
        "question": "In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?",
        "gt": "60",
        "output": " First, 25% of 20 students = 5 students enrolled in jazz dance. <answer> So, 20 students - 5 students = 15 students. </answer>",
        "judge_result": {
            "format_reward": 0.0,
            "answer_reward": 0.0,
            "reward": 0.0
        }
    },
```

**r1_zero_three_shot**在few-shot样本的帮助下明显有更好的表现。