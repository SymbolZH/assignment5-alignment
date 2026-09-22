# CS336 作业 5（对齐）：推理强化学习

版本：26.0.0  
CS336 教学团队，2026 年春季

原文：[cs336_spring2026_assignment5_alignment.pdf](../cs336_spring2026_assignment5_alignment.pdf)（39 页）。

> 翻译说明：按原文章节顺序翻译，保留题目标识、分值、公式编号和接口名称。代码、实验用英文提示词及数据示例保留原样，避免改变实验行为；代码注释和接口说明译为中文。公式使用 LaTeX，需在支持数学公式的 Markdown 阅读器中查看。文中的“本作业”“我们”均指原作业及教学团队。

## 1 作业概览

本作业将让你实际体验如何训练语言模型，使其能够推理并解决下游任务。

### 你将实现的内容

1. 零样本、少样本和思维链提示。
2. 组相对策略优化（Group Relative Policy Optimization，GRPO）：利用外部奖励提升模型表现的强化学习算法。
3. 策略梯度估计的变体，用于探索方差降低和重要性权重裁剪策略。

### 你将运行的实验

1. 测量 OLMo-2-0425-1B 在 GSM8K 上使用不同提示的表现。
2. 对 OLMo-2-0425-1B 运行同策略（on-policy）GRPO，提高其在 GSM8K 上的表现。
3. 运行 RFT、Dr. GRPO、MaxRL 等强化学习变体，探索算法设计选择。
4. 运行异策略（off-policy）GRPO，加快训练并探索各种裁剪策略。

### 代码结构

作业代码和本文档均位于 [GitHub 仓库](https://github.com/stanford-cs336/assignment5-alignment)。请使用 `git clone` 克隆。如有更新，我们会通知你，届时可以通过 `git pull` 获取最新版。

1. `cs336_alignment/*`：编写作业 5 代码的位置。除下述起始代码外，这里没有现成实现，因此你可以自由地从零组织代码。
2. `cs336_alignment/vllm_utils.py`：运行 vLLM 服务器、生成文本并同步权重的代码。
3. `cs336_alignment/drgrpo_grader.py`：为模型生成的数学题答案评分的代码。
4. `cs336_alignment/prompts/*`：便于使用的提示词文本文件。
5. `tests/*.py`：必做的 GRPO 测试，以及可选的对齐／安全补充作业测试。
6. `README.md`：环境配置的基本说明。

`tests/test_grpo.py` 中的必做测试会调用 `tests/adapters.py` 定义的接口。你需要实现这些 adapter，将自己的代码接入测试。添加测试或修改测试代码有助于调试，但最终实现应通过原始提供的测试套件。

### 如何提交

向 Gradescope 提交以下文件：

- `writeup.pdf`：回答所有书面问题。请使用排版工具整理答案。
- `code.zip`：包含你编写的全部代码。

运行 `test_and_make_submission.sh` 脚本生成 `code.zip`。

## 2 引言

### 2.1 背景

在前四次作业中，我们学习了如何预训练基础模型。现在开始学习后训练：有了基础模型之后，如何将它变成能够解决下游任务的实用工具？

后训练的一个组成部分是对齐。预训练的数据混合和训练目标会赋予模型广泛的知识和行为模式。然而，当我们要求模型解决任务时，希望它表现出某种特定行为，即成为有帮助且无害的助手。将通用基础模型转变为专注于对话的模型的过程称为“指令微调”或“对齐”，我们将在作业 5 的可选补充部分介绍这些技术。

另一个组成部分是强化学习。在预训练中，为使模型具备广泛知识，我们使用广泛的数据，例如网络文本。在后训练强化学习中，目标变得更集中：希望模型在某个特定任务上取得高准确率，例如解数学题。这个较窄的目标与预训练至少有两点不同：（a）我们拥有的数据更少；（b）训练目标由“覆盖广泛知识”变为“生成准确回答”。这些差异促使我们采用强化学习（RL）。

在 RL 中，我们得到一个问题数据集，以及一个判断回答是否正确解决问题的评分函数。例如，编程任务可能要求“写一个将列表反转的 Python 函数”，评分函数则是一组测试，例如 `assert f([0, 1, 2]) == [2, 1, 0]`。数学任务可能是：“四年前 Tom 的年龄是 John 的一半，John 现在 20 岁，Tom 现在多大？”评分函数解析最终答案并检查它是否等于 12。

与预训练不同，这里没有给出供模型模仿的回答数据集：编程例子没有提供正确的 Python 程序，数学推理例子也没有提供通向答案的正确推理链。RL 直接以模型准确率为目标进行梯度更新。总体而言，就是从模型采样回答，用评分函数打分，再提高正确回答的概率。

本作业将从数学和实验两方面研究 RL。RL 速度慢、稳定性差，因此并不容易；研究 RL 同样困难，因为不同随机种子的运行结果方差很大，看似微小的实现细节也会产生重大影响。我们将介绍大语言模型强化学习，并探索其中的一些挑战。

### 2.2 模型与数据集

本作业使用基础模型 [OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B)。它在 [OLMo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) 上预训练，再在 [Dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124) 上进行中期训练，总计使用 4 万亿个 token。OLMo-mix-1124 主要由课堂介绍过的 DCLM-Baseline [J. Li 等，2024] 组成；Dolmino-mix-1124 更有针对性，大约一半为 DCLM，另一半为指令跟随、数学、代码、STEM 论文和百科数据。其他细节参见 OLMo 2 技术报告 [T. OLMo 等，2024]。学到这里，你应已具备理解其各项设计选择的背景知识。

下游任务使用 GSM8K 数据集 [K. Cobbe 等，2021]，它位于仓库的 `data/gsm8k/train.jsonl` 和 `data/gsm8k/test.jsonl`，也可[在线获取](https://huggingface.co/datasets/openai/gsm8k)。该数据集包含较简单的小学数学应用题，例如：

```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}
```

例题含义：Natalia 四月卖出 48 个发夹，五月卖出四月的一半，两个月合计 $48+24=72$ 个。

在 RL 过程中，模型将学习为这类问题生成推理链，从而提高解数学题的能力。

关于模型和数据集的选择：受资源限制，我们只选取一个小规模的模型与数据集组合用作 RL 试验平台。这个小模型经过海量 token 训练，训练 token 数是参数量的 4000 倍，因此能力较强，使我们能在较小规模和真实数据集上观察到合理的 RL 效果。遗憾的是，本课程此前训练的模型还不足以解数学题。完成本作业后，如果感兴趣，你也可以将课堂上训练的模型用于更简单任务的 RL。RL 训练动态高度依赖模型和数据集，所以本作业观察到的结果未必能迁移到其他组合。

### 2.3 符号约定

本作业涉及一些数学推导。语言建模和强化学习经常用不同术语指代同一对象，因此下表同时列出两种叫法。忘记符号含义时可以随时回查。

| 符号 | 语言模型术语 | 强化学习术语 | 含义 |
|---|---|---|---|
| $\rho$ | 提示分布／数据集 | 初始状态分布 | 提示或问题的分布 |
| $x$ | 提示、问题 | 初始状态 | 从 $\rho$ 采样的问题 |
| $y$ | 回答、补全、生成结果、样本 | rollout、轨迹、采样动作序列 | 针对提示 $x$ 采样的回答 |
| $y_t$ | token | 动作 | 回答中第 $t$ 个生成 token |
| $y_{<t}$ | 前缀 | — | 位置 $t$ 之前的所有生成 token，即 $y_1,\ldots,y_{t-1}$ |
| $\pi_\theta$ | 模型 | 策略 | 参数为 $\theta$ 的模型，为给定提示 $x$ 的回答 $y$ 分配概率 $\pi_\theta(y\mid x)$ |
| $\pi_\theta(y_t\mid x,y_{<t})$ | 下一 token 分布 | 时刻 $t$ 的策略 | 给定提示和已生成 token 时，$y_t$ 的条件概率 |
| $r(y\mid x)$ | — | 奖励 | 表示回答正确性的标量分数，本作业取 0 或 1 |
| $B$ | 每批提示数 | — | 每个推理批次中的提示数 |
| $G$ | 每个提示的生成数 | 组大小 | 每个提示采样的回答数 |
| $\operatorname{len}(y)$ 或 $L$ | 回答长度 | 时域长度 | 一个回答中生成的 token 数 |
| $A^{(i,j)}$ | — | 优势（advantage） | 对提示 $i$ 的回答 $j$，经基线调整和归一化后的权重 |

## 3 提示

将预训练基础模型用于下游任务的第一步是给它提示。基础模型通过预训练学到了广泛的行为，提示是一种轻量方法，可以让其行为更偏向解决任务。后文还会看到，提示选择会影响 RL 动态和探索。

最基本的方法是给出问题，再从模型的下一 token 分布中采样答案；我们称其为 `question_only`。我们会将它与 `r1_zero` 比较：后者除了问题，还包含要求模型进行思维链推理的指令 [DeepSeek-AI 等，2025]。

### 3.1 使用 vLLM 进行推理

生成模型回答需要推理引擎。实现推理引擎不在本作业范围内，因此我们使用 vLLM [W. Kwon 等，2023]。它实现了快速 CUDA 内核、用于高效管理注意力 KV 缓存的 PagedAttention 等优化。`cs336_alignment/vllm_utils.py` 已提供启动服务器和生成文本的代码，接口如下：

```python
@dataclass
class VLLMCompletion:
    text: str
    token_ids: list[int]
    finish_reason: str | None

@dataclass
class VLLMServer:
    model_id: str
    gpu: int = 0
    seed: int = 0
    gpu_memory_utilization: float = 0.9

    def start(self) -> None: ...

    def generate_completions(
        self,
        prompts: list[str],
        sampling_params: dict,
        batch_size: int | None = None,
    ) -> list[VLLMCompletion]: ...
```

### 3.2 零样本、少样本和思维链提示

除非另有说明，GSM8K 实验使用以下来自 DeepSeek R1-Zero [DeepSeek-AI 等，2025] 的提示，称为 `r1_zero`：

```text
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and the answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
```

提示含义：用户提出问题，助手先思考推理过程，再给出答案；推理放在 `<think>...</think>` 中，答案放在 `<answer>...</answer>` 中。

该提示位于 `cs336_alignment/prompts/r1_zero.prompt`。其中 `question` 是插入的问题，例如前述 Natalia 卖发夹的题目。模型应扮演助手，从思考过程开始生成，因为开头的 `<think>` 已经包含在提示中；随后以 `</think>` 结束思考，再在答案标签内生成最终的符号答案，例如 `<answer> 4x + 10 </answer>`。这些标签便于解析输出并与标准答案比较，也使我们可以在遇到 `</answer>` 时停止生成。

另一种方法称为少样本提示：在真正的问题之前放几个问答示例。`r1_zero` 的少样本形式如下：

```text
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and the answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question-1}
Assistant: <think> {reasoning-1} </think> <answer> {answer-1} </answer>
User: {question-2}
Assistant: <think> {reasoning-2} </think> <answer> {answer-2} </answer>
User: {question-3}
Assistant: <think> {reasoning-3} </think> <answer> {answer-3} </answer>
User: {question}
Assistant: <think>
```

少样本提示通过展示待解决任务的几个示例来改善模型表现。开放基础模型报告的基准成绩通常采用少样本设置，例如 OLMo-2-0425-1B 模型卡上的 GSM8K 结果采用 8-shot 提示。我们在 `cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt` 中提供了 3-shot 版本，示例来自 [OLMES 仓库](https://github.com/allenai/olmes/blob/main/oe_eval/tasks/fewshot_sources.py#L2247)。

最后，作为基线，我们还使用 `cs336_alignment/prompts/question_only.prompt` 中的提示：

```text
{question} Please put your final answer within \\boxed{{}}.
```

虽然称为 `question_only`，它仍要求将最终答案放进方框，以便评分器从回答中解析答案，下一节会进一步说明。

### 3.3 评分函数

模型生成回答后，我们需要检查其正确性。数学题带有标准答案，例如 0.5，但模型可以用多种形式表达同一个正确答案，如 `<answer> 1/2 </answer>` 或 `The answer is 0.5.`。因此，我们需要一个以模型输出和已知标准答案为输入、返回正确与否的布尔值的答案解析函数。

实验使用近期推理 RL 研究中的一个快速且较准确的解析器 [Z. Liu 等，2025]。对于 `r1_zero` 提示，奖励函数为 `cs336_alignment.drgrpo_grader.r1_zero_reward_fn`。`question_only` 不要求 `<think>` 和 `<answer>` 标签，应使用同文件中的 `cs336_alignment.drgrpo_grader.question_only_reward_fn`；它会在 `\boxed{}` 中寻找最终答案。

这些函数返回总奖励，以及独立的格式奖励和答案奖励，分别表示是否符合预期格式、解析出的答案是否正确。有些研究对格式正确但答案错误的回答也给部分分数，即非零总奖励。本实验不给部分分：总奖励就是答案奖励，格式奖励仅用于日志记录。

注意，`ground_truth` 参数应只包含最终答案。GSM8K 的答案字段格式为 `{rationale} #### {answer}`，因此应按 `####` 分割，并去掉答案两端空白。

### 3.4 实验

现在可以评估基础模型在不同提示下的表现。

**生成超参数。** 采样温度为 1.0，top-p 为 1.0，最大生成长度为 512。`r1_zero` 要求模型以 `</answer>` 结束，因此可以指示 vLLM 在输出该字符串时停止：

```python
# 参考 Dr. GRPO：在模型完成答案时停止
# https://github.com/sail-sg/understand-r1-zero/blob/
#   c18804602b85da9e88b4aeeb6c43e2f08c594fbc/train_zero_math.py#L167
sampling_params['stop'] = ["</answer>"]
sampling_params['include_stop_str_in_output'] = True
```

这个停止字符串仅用于 `r1_zero` 和 `r1_zero_three_shot`，不要用于 `question_only`。

#### 题目（prompting_baselines）：在 GSM8K 上运行 OLMo-2-0425-1B（5 分）

**（a）** 编写脚本，评估零样本 `question_only`、零样本 `r1_zero` 和少样本 `r1_zero_three_shot` 下的模型表现。运行脚本并观察输出。对每种提示，统计生成结果属于以下三类的数量：（1）格式和正确性奖励均为 1；（2）格式奖励为 1、正确性奖励为 0；（3）两项奖励均为 0。至少观察第 2 类的十个例子：其中多少其实回答正确，只是未被正确解析？第 3 类又如何？

**交付内容：** 几句话的分析、评估指标，以及若干提示和回答示例。

**（b）** 根据输出描述每种提示下的模型行为。例如，若希望模型回答问题，仅提供问题是否足够？模型是否还有回答之外的其他行为？零样本 `r1_zero` 和少样本 `r1_zero_three_shot` 如何塑造模型行为？

**交付内容：** 几句话的分析，并用示例支持。

## 4 组相对策略优化

测量了仅靠提示得到的表现后，下一步是通过训练改进它。具体来说，我们希望优化准确率，等价地，优化期望奖励：

$$
J_\theta=\mathbb E_{x\sim\rho}\mathbb E_{y\sim\pi_\theta(y\mid x)}[r(y\mid x)]. \tag{1}
$$

其中 $\rho$ 是提示／问题 $x$ 的任务分布，$\pi_\theta$ 是模型，$y$ 是采样得到的回答，$r(y\mid x)$ 表示 $y$ 是否正确回答了 $x$，我们也将 $r$ 称为奖励函数。

这与此前学习的预训练目标很不同：交叉熵损失 $\mathbb E_{x\sim\mathcal D}[-\log\pi_\theta(x)]$ 对数据集样本取期望，而准确率目标对模型自身生成的样本取期望。因此需要新的优化工具，尤其是 RL。其基本过程是从模型采样回答、评分、强化正确回答。RL 通常缓慢且困难，所以本作业的大量内容将探索如何使它更快、更稳定。

由于 RL 从模型自身采样，它的动态也受采样时提示的影响。语言模型研究中的一个令人振奋的发现是：使用思维链提示，对强基础模型执行 RL，可以大幅提高推理能力 [OpenAI 等，2024；DeepSeek-AI 等，2025]。因此，本作业除了核心 RL 算法，还将研究它与思维链提示等语言建模技术的相互作用。

### 4.1 推导同策略 GRPO

我们先研究 RL 的数学基础，逐步推导语言模型训练中常用的 GRPO 算法 [Z. Shao 等，2024]。这里的介绍紧密参考了两份更深入的优秀资料：OpenAI 的 *Spinning Up in Deep RL* [J. Achiam，2018]，以及 Nathan Lambert 的 *Reinforcement Learning from Human Feedback (RLHF) Book* [N. Lambert，2024]。

#### 4.1.1 将语言模型视为策略

传统强化学习中，策略接收状态 $s_t$ 并输出动作 $a_t$。该动作带来奖励 $r_t=r(s_t,a_t)$，并按照下一状态分布转移到 $s_{t+1}\sim\operatorname{next\_state}(s_t,a_t)$。RL 的目标是优化策略，使其获得高奖励。

参数为 $\theta$ 的因果语言模型 $\pi_\theta$，给定文本前缀 $y_{<t}=(y_1,\ldots,y_{t-1})$，定义下一 token $y_t$ 的概率分布。因此，下一 token 对应动作，当前文本前缀对应状态。语言模型是一个类别分布上的随机策略：

$$
a_t\sim\pi_\theta(\cdot\mid s_t),\qquad
\pi_\theta(a_t\mid s_t)=[\operatorname{softmax}(f_\theta(s_t))]_{a_t}. \tag{2}
$$

使用策略梯度优化时需要两个基本操作：

1. 从策略采样：从上述类别分布抽取动作 $a_t$。
2. 计算动作的对数似然：求 $\log\pi_\theta(a_t\mid s_t)$。

在 LLM 的 RL 中，$s_t$ 通常是当前已生成的部分回答／解答，每个 $a_t$ 是下一个 token；输出文本结束 token 时回合结束，例如 `<|end_of_text|>`，或 `r1_zero` 中的 `</answer>`。

#### 4.1.2 轨迹

RL 从初始分布采样状态 $s_0\sim\rho$，再从策略采样动作，按照下一状态分布转移，并不断重复。状态和动作构成有限时域轨迹：

$$
\tau=(s_0,a_0,s_1,a_1,\ldots,s_T,a_T). \tag{3}
$$

其中 $T$ 表示轨迹长度：$a_T$ 是文本结束 token，或者此时已经达到最大生成 token 预算。

在本作业中，初始状态就是包含数学题的提示 $x$，可能附有思维链指令或少样本示例。以此为前缀逐个采样 token，下一状态就是将生成 token 拼接到旧前缀，即 $s_{t+1}=(s_t,a_t)$。轨迹也称为 episode 或 rollout，后文交替使用这些叫法。

#### 4.1.3 奖励与回报

标量奖励 $r_t=r(s_t,a_t)$ 衡量状态 $s_t$ 下动作的即时质量。在数学应用题这类可验证任务中，中间推理步骤奖励为零，直到输出答案时，才在终止动作上获得可验证奖励：

$$
r_T=r(s_T,a_T):=
\begin{cases}
1,&\text{奖励函数判定轨迹 }\tau\text{ 与标准答案一致},\\
0,&\text{否则}.
\end{cases} \tag{4}
$$

也就是说，最终答案正确则 $r_T=1$，否则为 0。回报 $R(\tau)$ 聚合整条轨迹上的奖励，在这里就是 $r_T$。

智能体的目标是最大化期望回报：

$$
J_\theta=\mathbb E_{\tau\sim\pi_\theta}[r(\tau)]. \tag{5}
$$

$\tau\sim\pi_\theta$ 表示先采样 $s_0\sim\rho$，然后依策略采样动作、依转移分布采样状态所得到的轨迹分布。在这里，就是先从数据集抽取数学题 $x\sim\rho$，再逐 token 采样直至回答结束，可记为 $y\sim\pi_\theta(y\mid x)$。相应优化问题为：

$$
\theta^*=\arg\max_\theta J_\theta. \tag{6}
$$

#### 4.1.4 策略梯度

现在可以介绍策略梯度算法。此后使用提示／回答的语言模型记号 $(x,y)$，不再使用状态／动作记号 $(s_t,a_t)$。

回顾期望奖励，即准确率目标：

$$
J_\theta=\mathbb E_{x\sim\rho}\mathbb E_{y\sim\pi_\theta(y\mid x)}[r(y\mid x)]. \tag{7}
$$

$r(y\mid x)$ 表示回答是否正确。我们采用梯度上升：

$$
\theta_{k+1}=\theta_k+\alpha\nabla_\theta J_{\theta_k}. \tag{8}
$$

为此必须用样本估计梯度。把梯度改写成对样本的期望，就得到以 REINFORCE 论文命名的“REINFORCE 策略梯度” [R. J. Williams，1992]：

$$
\nabla_\theta\mathbb E_{x\sim\rho}\mathbb E_{y\sim\pi_\theta(y\mid x)}[r(y\mid x)]
=\mathbb E_{x\sim\rho}\mathbb E_{y\sim\pi_\theta(y\mid x)}[r(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)]. \tag{9}
$$

在推导之前，先注意这给出了一个直观算法：从数据集采样问题，从模型采样回答，计算样本上 $r(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)$ 的平均值，再做梯度更新。该表达式意味着提高高奖励回答的对数概率。重复此过程就得到基本 RL 训练循环。

推导用到对数导数技巧。回顾 $(\log f(x))'=f'(x)/f(x)$，于是：

$$
\nabla_\theta\pi_\theta(y\mid x)=\pi_\theta(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x). \tag{10}
$$

直接应用可得：

$$
\nabla_\theta\mathbb E_{y\sim\pi_\theta(y\mid x)}[r(y\mid x)]
=\sum_y\nabla_\theta\pi_\theta(y\mid x)r(y\mid x) \tag{11}
$$
$$
=\sum_y\pi_\theta(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)r(y\mid x) \tag{12}
$$
$$
=\mathbb E_{y\sim\pi_\theta(y\mid x)}[r(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)]. \tag{13}
$$

这里省略了提示上的期望 $\mathbb E_{x\sim\rho}$，因为它可以与 $\nabla_\theta$ 交换顺序。

给定独立同分布采样的提示 $x^{(1)},\ldots,x^{(B)}\sim\rho$，以及每个提示的 $G$ 个独立回答 $y^{(i,j)}\sim\pi_\theta(y\mid x^{(i)})$，策略梯度的样本估计为：

$$
\hat g\leftarrow\frac1{BG}\sum_{i=1}^B\sum_{j=1}^G
r(y^{(i,j)}\mid x^{(i)})\nabla_\theta\log\pi_\theta(y^{(i,j)}\mid x^{(i)}). \tag{14}
$$

它提高高奖励回答的权重。下面从这个基本估计器出发，逐项修改，得到 GRPO。

#### 4.1.5 基线

基本 REINFORCE 估计器虽然期望正确，等于 $\nabla_\theta J_\theta$，但方差较大，使训练不稳定。基线是降低方差、稳定训练的一种工具。

基线可以很复杂，最简单的做法是从奖励中减去常数 $b$：

$$
\mathbb E_{x\sim\rho}\mathbb E_{y\sim\pi_\theta(y\mid x)}[(r(y\mid x)-b)\nabla_\theta\log\pi_\theta(y\mid x)]. \tag{15}
$$

例如奖励为二元值时，取 $b=0.5$，就以 0.5 的权重提高正确回答的概率，以 −0.5 的权重降低错误回答的概率。原始估计器则以 1 的权重强化正确回答，对错误回答不做处理，因为其权重为 0。

本节关键结论是：只要 $b$ 不依赖动作／回答 $y$，减去它就保持估计器的期望不变。它可以是常数、$x$ 的函数或策略的函数。即：

$$
\mathbb E_{x\sim\rho}\mathbb E_{y\sim\pi_\theta(y\mid x)}[(r(y\mid x)-b)\nabla_\theta\log\pi_\theta(y\mid x)]
=\mathbb E_{x\sim\rho}\mathbb E_{y\sim\pi_\theta(y\mid x)}[r(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)]. \tag{16}
$$

这直接来自恒等式：

$$
\mathbb E_{y\sim\pi_\theta(y\mid x)}[\nabla_\theta\log\pi_\theta(y\mid x)]
=\nabla_\theta\mathbb E_{y\sim\pi_\theta(y\mid x)}[1]=0. \tag{17}
$$

第一步逆向使用了对数导数技巧。

基线保持期望，却可能增大或减小方差。若将估计器写为样本均值 $\frac1n\sum_{i=1}^n Z_i$，每个 $Z_i$ 是一个样本 $(x,y)$ 对应的 $(r(y\mid x)-b)\nabla_\theta\log\pi_\theta(y\mid x)$，则估计器方差为 $[\mathbb E(Z_i^2)-\mathbb E(Z_i)^2]/n$。下一题探索何时基线增大或减小方差。

#### 题目（baseline_calcs）：计算策略梯度估计器的方差（5 分）

设 $\pi_\theta$ 是二元动作空间 $\mathcal A=\{0,1\}$ 上的策略，$\pi_\theta(A=1)=p=\sigma(\theta)$，其中 $\sigma(\theta)=1/(1+e^{-\theta})$ 为 sigmoid。奖励函数对 $A=1$ 给 1 分，否则给 0 分，即 $r(A)=\mathbf1\{A=1\}$。

**（a）** 对 $n$ 个独立同分布样本 $A_i\sim\pi_\theta$，策略梯度估计器为：

$$
\frac1n\sum_{i=1}^n r(A_i)\nabla_\theta\log\pi_\theta(A_i). \tag{18}
$$

求它的方差。

**交付内容：** 以 $n,p$ 表示的表达式及推导。

**（b）** 加入基线后，估计器为：

$$
\frac1n\sum_{i=1}^n(r(A_i)-b)\nabla_\theta\log\pi_\theta(A_i). \tag{19}
$$

求它在 $n$ 个独立同分布样本下的方差。

**交付内容：** 以 $n,b,p$ 表示的表达式、推导及讨论。

**（c）** 代入“总体均值”基线 $b=p$ 后，方差是多少？与未经基线调整的估计器比较，它总是更低、总是更高，还是随 $p$ 不同而有高有低？

**GRPO 中的基线。** GRPO 的第一项修改是使用“组均值”基线：

$$
\hat g\leftarrow\frac1{BG}\sum_{i=1}^B\sum_{j=1}^G
\bigl(r(y^{(i,j)}\mid x^{(i)})-\mu_i\bigr)\nabla_\theta\log\pi_\theta(y^{(i,j)}\mid x^{(i)}), \tag{20}
$$

其中 $\mu_i=\frac1G\sum_{j=1}^G r(y^{(i,j)}\mid x^{(i)})$，即模型在提示 $x^{(i)}$ 上的平均奖励。尽管 $\mu_i$ 依赖回答 $y$，它仍然保持期望方向，仅产生 $(G-1)/G$ 的缩放：

$$
\mathbb E_{y^{(1)},\ldots,y^{(G)}\sim\pi_\theta(y\mid x)}
\left[\frac1G\sum_{j=1}^G(r(y^{(j)}\mid x)-\mu_i)\nabla_\theta\log\pi_\theta(y^{(j)}\mid x)\right] \tag{21}
$$
$$
=\frac1G\sum_{j=1}^G\mathbb E_{y^{(2)},\ldots,y^{(G)}\sim\pi_\theta(y\mid x)}
\left[\mathbb E_{y^{(1)}\sim\pi_\theta(y\mid x)}
[(r(y^{(1)}\mid x)-\mu_i)\nabla_\theta\log\pi_\theta(y^{(1)}\mid x)]\right] \tag{22}
$$
$$
=\frac1G\sum_{j=1}^G\mathbb E_{y^{(1)}\sim\pi_\theta(y\mid x)}
\left[\left(r(y^{(1)}\mid x)-\frac1G r(y^{(1)}\mid x)\right)\nabla_\theta\log\pi_\theta(y^{(1)}\mid x)\right] \tag{23}
$$
$$
=\frac{G-1}{G}\mathbb E_{y\sim\pi_\theta(y\mid x)}[r(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)]. \tag{24}
$$

经过组均值调整的奖励通常称为“优势”：它不再表示绝对奖励，而表示当前 rollout 相对平均奖励的优势。

#### 4.1.6 优势归一化

减去组均值之后，GRPO 的下一项修改是除以组标准差：

$$
\hat g\leftarrow\frac1{BG}\sum_{i=1}^B\sum_{j=1}^G
\frac{r(y^{(i,j)}\mid x^{(i)})-\mu_i}{\mathrm{std}_i}
\nabla_\theta\log\pi_\theta(y^{(i,j)}\mid x^{(i)}). \tag{25}
$$

其中 $\mathrm{std}_i=\sqrt{\frac1G\sum_{j=1}^G(r(y^{(i,j)}\mid x^{(i)})-\mu_i)^2}$。注意，默认 `torch.std` 使用略有不同的样本标准差表达式进行偏差校正；实现时应使用默认 `torch.std`。

除以标准差不再保持估计器期望，因此我们不再对原始期望奖励目标 $J_\theta$ 做梯度上升。可以把它理解为稳定性技巧：假设各梯度向量独立同分布且服从高斯分布，那么除以组标准差会使各组更新的范数大致相同。下一节将进一步研究优势归一化。

#### 4.1.7 序列归一化

GRPO 还有一个似乎继承自早期 LLM-PPO 实现的细节，称为序列归一化。把回答的对数概率展开为时间步之和，当前估计器为：

$$
\hat g\leftarrow\frac1{BG}\sum_{i=1}^B\sum_{j=1}^G
\frac{r(y^{(i,j)}\mid x^{(i)})-\mu_i}{\mathrm{std}_i}
\nabla_\theta\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\log\pi_\theta(y_t^{(i,j)}\mid x^{(i)},y_{<t}^{(i,j)}). \tag{26}
$$

GRPO 加入额外的序列长度归一化项：

$$
\hat g\leftarrow\frac1{BG}\sum_{i=1}^B\sum_{j=1}^G
\frac{r(y^{(i,j)}\mid x^{(i)})-\mu_i}{\mathrm{std}_i}
\nabla_\theta\left(\frac1{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}\log\pi_\theta(y_t^{(i,j)}\mid x^{(i)},y_{<t}^{(i,j)})\right). \tag{27}
$$

这进一步改变了期望：不同序列的 token 不再具有相同权重，长序列 token 相对短序列 token 的权重更低。下一节会讨论并通过消融实验判断这种修改是否合适。目前先按原始论文实现标准 GRPO，因此保留序列归一化。

#### 4.1.8 将所有部分组合起来

尚未介绍的 GRPO 组件是带裁剪的重要性加权，后面讨论异策略 RL 时会讲解。简而言之，同策略 RL 对每个推理批次做一次梯度更新，异策略 RL 则做多次更新以提高速度。“异策略”指第二次及之后更新使用的样本已经过时，因为它们来自较旧版本的模型。重要性加权修改梯度估计器，使其能够使用旧样本。现在只做同策略 RL，暂时不需要它。

同策略 GRPO 所需组件已全部具备，算法如下。

**算法 1：同策略组相对策略优化（GRPO）**

输入：初始策略模型 $\pi_{\theta_0}$；奖励函数 $r$；任务分布／数据集 $\rho$；学习率 $\alpha$。  
输出：$\pi_\theta$。

1. 初始化策略模型 $\pi_\theta\leftarrow\pi_{\theta_0}$。
2. 对 `step = 1, …, n_grpo_steps` 重复以下步骤。
3. 独立采样一批问题 $x^{(1)},\ldots,x^{(B)}\sim\rho$。
4. 每题独立采样 $G$ 个回答 $y^{(i,1)},\ldots,y^{(i,G)}\sim\pi_\theta(y\mid x^{(i)})$。
5. 计算同策略 GRPO 策略梯度估计器：

$$
\hat g\leftarrow\frac1{BG}\sum_{i=1}^B\sum_{j=1}^G
\frac1{\operatorname{len}(y^{(i,j)})}\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\frac{r(y^{(i,j)}\mid x^{(i)})-\mu_i}{\mathrm{std}_i}
\nabla_\theta\log\pi_\theta(y_t^{(i,j)}\mid x^{(i)},y_{<t}^{(i,j)}), \tag{28}
$$

其中组均值与组标准差分别为：

$$
\mu_i=\frac1G\sum_{j=1}^G r(y^{(i,j)}\mid x^{(i)}), \tag{29}
$$
$$
\mathrm{std}_i=\sqrt{\frac1G\sum_{j=1}^G(r(y^{(i,j)}\mid x^{(i)})-\mu_i)^2}. \tag{30}
$$

6. 使用你选择的优化器更新 $\theta$。以下为随机梯度上升：

$$
\theta\leftarrow\theta+\alpha\hat g. \tag{31}
$$

### 4.2 实现同策略 GRPO

#### 4.2.1 使用 Hugging Face 模型

此前作业使用 `cs336_basics` 中自行实现的语言模型，本作业直接使用 Hugging Face 的 `transformers` 库加载预训练基础模型。你也可以自行实现 Transformer 和预训练权重加载器，只需确保架构与 OLMo-2-0425-1B 一致。

以下起始代码位于 `cs336_alignment/checkpoint.py`，以 bfloat16 和 FlashAttention-2 加载模型以节省显存：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if device=='cpu' else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer
```

模型标识可以是 `allenai/OLMo-2-0425-1B` 这样的名称，也可以是目录路径。目录通常由 `save_pretrained` 生成。训练后可这样保存模型：

```python
# 保存模型权重
model.save_pretrained(save_directory=output_dir)
tokenizer.save_pretrained(save_directory=output_dir)
```

首先实现一个辅助函数，使用预训练 tokenizer 对提示和回答分词。两者必须分别分词，不添加特殊 token，然后直接按 `prompt_ids + response_ids` 拼接，中间不插入 EOS、BOS 或分隔 token。

还需要构造 `response_mask`：它是与错位后的 labels 对齐的布尔掩码，仅当对应 label token 来自回答时为 `True`，提示和 padding 位置均为 `False`。训练时用它确保只在回答 token 上计算 loss。

#### 题目（tokenize_prompt_and_output）：提示与输出分词（1 分）

**交付内容：** 实现 `tokenize_prompt_and_output`，分别对提示和输出分词，无特殊 token 地拼接，并构造 `response_mask`。推荐接口：

```python
def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
```

功能：对提示和输出字符串分词，构造与 labels 对齐的掩码；回答 token 为 1，其他 token（提示或 padding）为 0。

参数：

- `prompt_strs: list[str]`：提示字符串列表。
- `output_strs: list[str]`：输出字符串列表。
- `tokenizer: PreTrainedTokenizer`：使用的 tokenizer。

返回 `dict[str, torch.Tensor]`。设 `prompt_and_output_lens` 为各样本分词后拼接长度的列表，字典包含：

| 键 | 张量形状 | 含义 |
|---|---|---|
| `input_ids` | `(batch_size, max(prompt_and_output_lens) - 1)` | 拼接后的提示和输出 token，去掉最后一个 token |
| `labels` | 同上 | 错位后的输入 ID，即去掉第一个 token |
| `response_mask` | 同上 | 与 labels 对齐，对应 label 属于回答时为 1，否则为 0 |

实现 `adapters.run_tokenize_prompt_and_output`，然后运行并通过：

```bash
uv run pytest -k test_tokenize_prompt_and_output
```

得到分词后的输入后，可这样送入模型：

```python
input_ids = train_batch["input_ids"].to(device)
labels = train_batch["labels"].to(device)
logits = model(input_ids).logits
```

接下来实现计算回答逐 token 对数概率的函数，这是计算策略梯度所需的基本操作。RL 中记录逐 token 熵也很有用，因此还需要实现 `return_token_entropy` 选项。

#### 题目（get_response_log_probs）：回答的对数概率与熵（1 分）

**交付内容：** 实现 `get_response_log_probs`，从因果语言模型获得各 token 在前序 token 条件下的对数概率，并可选返回模型下一 token 分布的熵。推荐接口：

```python
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
```

参数：

- `model: PreTrainedModel`：用于评分的 Hugging Face 模型，应放到正确设备上；不需梯度时使用[推理模式](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html#inference-mode)。
- `input_ids: torch.Tensor`：形状 `(batch_size, sequence_length)`，由分词函数产生的提示与回答拼接 token。
- `labels: torch.Tensor`：同样形状，由分词函数产生的 labels。
- `return_token_entropy: bool`：为 `True` 时同时返回逐 token 熵。

返回字典：

- `"log_probs"`：形状 `(batch_size, sequence_length)`，条件对数概率 $\log p_\theta(x_t\mid x_{<t})$。
- `"token_entropy"`：可选，同样形状，每个位置的 token 熵，仅在 `return_token_entropy=True` 时出现。

实现 `adapters.run_get_response_log_probs`，运行并通过：

```bash
uv run pytest -k test_get_response_log_probs
```

#### 4.2.2 在强化学习循环中使用 vLLM

RL 循环需要生成 rollout。我们的配置是：一张 GPU 放 Hugging Face 模型和优化器用于训练，另一张 GPU 放 vLLM，包括模型和 KV 缓存。因此，除前述初始化和生成接口外，每次推理前还需要在两张卡之间同步权重。以下接口位于 `cs336_alignment/vllm_utils.py`：

```python
@dataclass
class VLLMServer:
    gpu: int = 1  # GPU 0 训练，GPU 1 推理

    # 在训练 GPU 和 vLLM 之间创建 NCCL 权重传输组
    def init_weight_sync(self, policy_device: str): ...

    # 将当前 Hugging Face 策略权重复制到 vLLM，
    # 并重置依赖旧权重的 vLLM 缓存
    def sync_policy_weights(self, policy: torch.nn.Module) -> None: ...

    # 使用 vLLM 当前权重生成 rollout
    def generate_completions(
        self,
        prompts: list[str],
        sampling_params: dict,
        batch_size: int | None = None,
    ) -> list[VLLMCompletion]: ...
```

与提示实验一样，使用温度 1.0、top-p 1.0、最大生成长度 512。提示要求以 `</answer>` 结束，因此可以设置：

```python
# 参考 Dr. GRPO：在模型完成答案时停止
# https://github.com/sail-sg/understand-r1-zero/blob/
#   c18804602b85da9e88b4aeeb6c43e2f08c594fbc/train_zero_math.py#L167
sampling_params['stop'] = ["</answer>"]
sampling_params['include_stop_str_in_output'] = True
```

#### 4.2.3 GRPO 组件

下面实现 GRPO loss 的组成部分。后面还会实现不同优势归一化器、不同重要性加权方式等变体。为使变体间切换时尽量少重复代码，接口包含选择变体的参数。本阶段只需要实现标准 GRPO。

首先实现计算回答奖励的辅助函数。

#### 题目（compute_rollout_rewards）：计算 rollout 奖励（1 分）

**交付内容：** 实现 `compute_rollout_rewards`，计算每个生成回答的原始奖励。

```python
def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
```

功能：为回答列表计算奖励，并返回奖励分量的元数据。

参数：

- `reward_fn`：对照标准答案给回答评分，返回包含 `"reward"`、`"format_reward"`、`"answer_reward"` 的字典。
- `rollout_responses: list[str]`：策略生成的回答，长度为 `rollout_batch_size = n_prompts_per_rollout_batch * group_size`。
- `repeated_ground_truths: list[str]`：每题标准答案重复 `group_size` 次后的列表，长度为 `rollout_batch_size`。

返回 `(raw_rewards, metadata)`：

- `raw_rewards`：形状 `(rollout_batch_size,)`，每个回答未经归一化的奖励。
- `metadata`：用于日志的奖励统计，至少包含该批次总奖励和格式奖励的均值。

实现 `adapters.run_compute_rollout_rewards`，运行并通过：

```bash
uv run pytest -k compute_rollout_rewards
```

下一步实现 GRPO 的核心数学操作，将奖励归一化为优势。上个函数输出的奖励是展平的，因此需要组大小将其恢复为分组形式。测试要求支持 `baseline="mean"`、`advantage_normalizer="std"`，并在组标准差上加 `advantage_eps`，避免除零。

#### 题目（compute_group_normalized_rewards_grpo）：组归一化（1 分）

**交付内容：** 实现 `compute_group_normalized_rewards`，在组内归一化原始奖励，返回归一化结果和你认为有用的元数据。

目前只需支持 `baseline="mean"`、`advantage_normalizer="std"`。对其他输入可以抛出 `NotImplementedError`，后面会实现其他选项并进行消融。记得给归一化分母加上 `advantage_eps`。

```python
def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
):
```

功能：在每个组内应用指定基线和归一化策略，计算优势。

参数：

- `raw_rewards`：形状 `(rollout_batch_size,)` 的原始奖励，其中 `rollout_batch_size = n_prompts_per_rollout_batch * group_size`。
- `group_size: int`：每题／每组的回答数。
- `baseline`：本题支持 `"mean"`，即减去组平均奖励；后面的 `"none"` 表示不减基线。
- `advantage_eps: float`：防止归一化时除零的小常数。
- `advantage_normalizer`：本题支持 `"std"`，即除以组标准差；后面的 `"none"` 表示不归一化，`"mean"` 表示除以组平均奖励。

返回 `tuple[torch.Tensor, dict[str, float]]`：

- `advantages`：形状 `(rollout_batch_size,)`，每个回答经过组归一化的奖励。
- `metadata`：自选日志统计，如奖励均值、标准差、最大／最小值。

实现 `adapters.run_compute_group_normalized_rewards`，运行并通过：

```bash
uv run pytest -k compute_group_normalized_rewards_grpo
```

接下来计算逐 token 策略梯度 loss。算法 1 写的是梯度估计器，本题要求写一个求导后产生 GRPO 各项梯度的逐 token loss，再在 token 间聚合。它并非传统意义上预期随训练不断下降的损失，只是求导后能得到策略梯度的表达式。

应实现的目标为：

$$
J_\theta^{\mathrm{GRPO\text{-}on\text{-}policy}}
=\frac1{BG}\sum_{i=1}^B\sum_{j=1}^G\frac1{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\frac{r(y^{(i,j)}\mid x^{(i)})-\mu_i}{\mathrm{std}_i}
\log\pi_\theta(y_t^{(i,j)}\mid x^{(i)},y_{<t}^{(i,j)}). \tag{32}
$$

可以检查它的梯度就是算法 1 中的梯度。PyTorch 优化器执行梯度下降，所以实现应返回这个目标的负值。`compute_policy_gradient_loss` 计算每个序列、每个 token 的 loss，随后由聚合函数在 token 和序列维度取平均。

#### 题目（compute_policy_gradient_loss_on_policy）：同策略策略梯度（1 分）

**交付内容：** 实现 `compute_policy_gradient_loss`，给定原始奖励或预先计算的优势，计算逐 token 策略梯度 loss。

目前假定所有 rollout 都是同策略的，只需支持 `importance_reweighting_method="none"`，可忽略旧对数概率和裁剪参数。其他输入可抛出 `NotImplementedError`，裁剪将在后面实现。

```python
def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```

参数：

- `raw_rewards_or_advantages`：形状 `(batch_size,)` 或 `(batch_size, 1)`，每个回答的标量奖励或已归一化优势。
- `policy_log_probs`：形状 `(batch_size, sequence_length)`，每个 token 的对数概率。
- `importance_reweighting_method`：`"none"` 不使用重要性加权；`"noclip"` 加权但不裁剪；`"grpo"` 使用 PPO/GRPO 风格的 token 级加权与裁剪；`"gspo"` 使用 GSPO 风格的序列级加权与裁剪。
- `old_log_probs`：形状 `(batch_size, sequence_length)`；除 `"none"` 外均需要。
- `cliprange`：裁剪参数 $\varepsilon$，`"grpo"` 和 `"gspo"` 需要。
- `response_mask`：可选，形状 `(batch_size, sequence_length)` 的回答 token 掩码。GSPO 若仅在回答 token 上平均序列对数比值，则需要它。

返回 `(per_token_policy_gradient_loss, metadata)`：

- loss 形状为 `(batch_size, sequence_length)`，训练循环中将沿批次和序列维度聚合。
- `metadata` 为底层 loss 计算的统计量，例如裁剪比例的组成项。

实现 `adapters.run_compute_policy_gradient_loss`，运行并通过：

```bash
uv run pytest -k test_compute_policy_gradient_loss_on_policy
```

最后实现跨 token 和序列的聚合。标准 GRPO 先在每个序列内平均，再对序列平均。后面会实现除以常数的替代方案，它更忠实于最初希望估计的策略梯度。

#### 题目（aggregate_loss_across_microbatch_sequence）：跨 token 和序列聚合 loss（0.5 分）

**交付内容：** 实现 `aggregate_loss_across_microbatch`，输入逐 token loss 和回答掩码，输出平均 loss。本阶段使用标准 GRPO 聚合，先在每个序列内平均，再对序列平均，即可假定 `loss_normalization="sequence"`。其他输入可抛出 `NotImplementedError`。

```python
def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
```

功能：根据回答掩码和 loss 归一化策略聚合逐 token 策略梯度 loss。

参数：

- `per_token_policy_gradient_loss`：形状 `(batch_size, sequence_length)`。
- `mask`：同样形状，标记哪些位置应计入 loss。
- `loss_normalization`：`"sequence"` 先序列内平均、再跨序列平均；`"constant"` 将总 loss 除以常数。
- `normalization_constant`：总 loss 的除数，`loss_normalization="constant"` 时必须提供。

返回 `loss: torch.Tensor`，包含平均 loss 的标量，必须保留后续调用 `backward` 的能力。

实现 `adapters.run_aggregate_loss_across_microbatch`，运行并通过：

```bash
uv run pytest -k test_aggregate_loss_across_microbatch_sequence
```

#### 4.2.4 GRPO 训练步骤

现在把这些组件组合成完整训练步骤。

**梯度累积。** 为充分利用推理资源，需要较大的批次。同策略 RL 中训练批次大小等于推理批次大小，因此 GPU 显存无法一次容纳整个批次的梯度计算。我们需要拆成多个微批次，累积梯度。关键是正确处理归一化，确保累积结果等价于整个批次一次计算的梯度。

PyTorch 中梯度累积很直接。每个权重张量的 `.grad` 属性保存梯度；第一次调用 `loss.backward()` 前通常为 `None`，调用后包含梯度。通常先做优化器更新，再调用 `optimizer.zero_grad()` 重置权重张量的 `.grad`：

```python
# 前向传播
logits = model(inputs)
loss = loss_fn(logits, labels)
# 反向传播
loss.backward()
# 更新权重
optimizer.step()
# 清空梯度，为下一轮准备
optimizer.zero_grad()
```

梯度累积把批次分成 $k$ 个微批次，只调用一次 `optimizer.step()` 和一次 `optimizer.zero_grad()`。若使用先序列内平均、再跨序列平均的归一化，计算出微批次平均 loss 后，还要按序列数量重新加权：

```python
gradient_accumulation_steps = 4
microbatch_size = len(inputs) // gradient_accumulation_steps
for i in range(0, len(inputs), microbatch_size):
    inputs_microbatch = inputs[i:i+microbatch_size]
    labels_microbatch = labels[i:i+microbatch_size]
    # 前向传播
    logits = model(inputs_microbatch)
    loss = loss_fn(logits, labels_microbatch) * (len(inputs_microbatch) / len(inputs))
    # 反向传播
    loss.backward()
# 整个批次只更新一次权重
optimizer.step()
# 整个批次只清空一次梯度
optimizer.zero_grad()
```

后面改用常数归一化时，微批次代码也要相应调整。

**实现训练步骤。** 给定一批 rollout，函数接收模型、tokenizer、优化器、奖励函数、提示、回答和超参数，累积梯度并执行一次优化器更新。为避免显存不足，必须实现前述梯度累积。在更新前，还应将梯度范数裁剪到 `max_grad_norm`。返回批次训练 loss 和元数据以供日志记录。至少记录：

- loss。
- 梯度范数。
- token 熵。
- 训练奖励，包括总奖励和格式奖励。

也欢迎记录其他指标，例如 $k$ 从 1 到 `group_size` 的 pass@k，即前 $k$ 个 rollout 中至少有一个正确回答的问题比例。

#### 题目（grpo_train_step_standard_on_policy）：GRPO 训练步骤（5 分）

**交付内容：** 给定模型、tokenizer 和 rollout，实现一次批次策略梯度更新。本阶段只要求标准同策略 GRPO：`baseline="mean"`、`advantage_normalizer="std"`、`importance_reweighting_method="none"`、`loss_normalization="sequence"`。其他输入可抛出 `NotImplementedError`。

```python
def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # 奖励归一化
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # 重要性加权与裁剪
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # loss 归一化
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
```

功能：通过 `gradient_accumulation_steps` 个微批次执行前向和反向传播。

参数：

- `model: PreTrainedModel`：待训练的 Hugging Face 模型。
- `tokenizer: PreTrainedTokenizer`：分词器。
- `optimizer: Optimizer`：模型优化器。
- `gradient_accumulation_steps: int`：一次优化器更新包含的微批次数。
- `max_grad_norm: float | None`：若非 `None`，在 `optimizer.step()` 前将梯度范数裁剪到此值。
- `reward_fn`：对照标准答案给 rollout 评分，返回 `"reward"`、`"format_reward"`、`"answer_reward"`。
- `repeated_prompts: list[str]`：每题提示重复 `group_size` 次，列表长度为 `rollout_batch_size`。
- `rollout_responses: list[str]`：策略回答，长度为 `rollout_batch_size = n_prompts_per_rollout_batch * group_size`。
- `repeated_ground_truths: list[str]`：每题标准答案重复 `group_size` 次，列表长度为 `rollout_batch_size`。
- `group_size: int`：每题／每组的回答数。
- `baseline`：`"mean"` 减去组平均奖励；`"none"` 不做处理。
- `advantage_eps: float`：避免归一化除零的小常数。
- `advantage_normalizer`：`"std"` 除以组标准差；`"none"` 不归一化；`"mean"` 除以组平均奖励。
- `importance_reweighting_method`：`"none"` 不加权；`"noclip"` 重要性加权但不裁剪；`"grpo"` 使用 PPO/GRPO token 级加权与裁剪；`"gspo"` 使用 GSPO 序列级加权与裁剪。
- `old_log_probs`：形状 `(batch_size, sequence_length)`，除 `"none"` 外均需要。
- `cliprange`：裁剪参数 $\varepsilon$，`"grpo"` 或 `"gspo"` 时需要。
- `loss_normalization`：`"sequence"` 先在每个序列内平均再跨序列平均；`"constant"` 将总 loss 除以整个训练期间固定的常数。
- `normalization_constant`：总 loss 的除数，常数归一化时必须提供。

返回值：

- `loss`：标量张量，经过梯度累积调整的批次 loss，供日志使用。
- `metadata`：包含底层 loss 计算的元数据、裁剪前梯度范数和其他希望记录的统计量。

实现 `adapters.run_grpo_train_step`，运行并通过：

```bash
uv run pytest -k test_grpo_train_step_standard_on_policy
```

### 4.3 实验

现在可以构建完整 GRPO 训练循环，包括加载模型与数据集，初始化日志（例如 Wandb）、vLLM 服务器和优化器，以及运行 RL 训练循环。

以下是建议超参数。若脚本正确，在多数随机种子下应看到验证奖励随训练增加。

```python
n_train_examples = 6400
n_val_examples = 1024
num_rollout_steps = 200
learning_rate = 1e-5
rollout_batch_size = train_batch_size = 256
group_size = 8
gradient_accumulation_steps = 32
sampling_temperature = 1.0
sampling_max_tokens = 512
max_grad_norm = 1.0
optimizer = torch.optim.AdamW(
    policy.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0
)
```

注意，`rollout_batch_size` 和 `train_batch_size` 统计的是回答数，不是提示数。因此 256 表示 32 个提示，每个提示生成 8 个 rollout。

合理的默认设置是每 10 个 rollout 批次评估一次验证集。由于 CoT/RL 评估噪声较大，验证样本数至少应为 1024。除上一节指标外，记录 rollout 也有助于定性理解模型行为；可默认每 40 个 rollout 批次记录当前批次的训练回答。可以使用 `data/gsm8k/train.jsonl` 训练，使用 `data/gsm8k/test.jsonl` 验证。

策略梯度估计器可能具有高方差，RL 又有自我强化的特性，因此训练轨迹的差异通常很大。确认脚本正确后，请使用 4 个随机种子运行 RL。固定种子后无法完全复现也没关系，也可以直接将实验运行 4 次。图中要体现运行间波动，不要只画均值。合理方式包括：

- 画 4 次运行的均值，同时展示各次实际曲线。为了清晰，可以将单次曲线另画一张图，或使用低透明度、细线、虚线。
- 画均值和阴影置信区间。每一步计算样本均值、样本标准差，使用 $\mu\pm1.96\sigma/\sqrt n$，其中 $n$ 是种子数。
- 画均值，以及每一步的最小值和最大值。

#### 题目（grpo_experiments_standard_on_policy）：用 GRPO 提高 OLMo-2-0425-1B 在 GSM8K 上的表现（2 B200 小时，10 分）

**（a）** 编写脚本，给定模型名、提示、训练和验证数据路径、采样超参数、训练超参数，运行 GRPO 训练循环。先初始化 vLLM、wandb 日志、数据集、模型和优化器，再重复：向 vLLM 同步权重、生成训练 rollout、在训练批次上执行策略梯度更新，并定期在验证集检查表现，同时记录生成结果。

**交付内容：** 在 GSM8K 和 OLMo-2-0425-1B 上运行标准同策略 GRPO 的脚本。

**（b）** 运行约 50 步，确认验证奖励改善，且 rollout 随时间表现合理。初始奖励接近零是正常的；根据随机种子不同，可能需要若干步才采样到非零奖励回答并开始提升。

**交付内容：** 令你相信脚本正确的证据，例如验证奖励改善、生成回答合理。

**（c）** 确认正确后，采用上述超参数和 4 个随机种子，在 GSM8K 上训练 OLMo-2-0425-1B，使用零样本 `r1_zero`。若代码正确，应看到验证奖励随训练改善。

记录并绘制以下指标随时间的变化，体现各次运行的方差，例如均值／标准差或最小值／最大值：

- loss。
- 梯度范数。
- token 熵。
- 训练奖励：总奖励、格式奖励。
- 验证奖励：总奖励、格式奖励。
- 验证回答平均长度。
- 其他有助于调试的指标。

还应定期记录并观察 rollout：训练是否改善了回答？

**交付内容：** 每个指标的一张图和几句话分析，描述训练中的变化及运行间波动；若干训练前后生成样例；一个在各随机种子上平均最终验证准确率达到至少 **25%** 的训练方案。

脚本可用后，开始调参和消融。先调整通常影响最大的超参数：学习率。

#### 题目（grpo_learning_rate）：调整学习率（4 B200 小时，3 分）

以建议超参数为起点，扫描学习率，至少包含一个小于默认值和一个大于默认值的学习率。报告最终验证奖励；若优化器发散，注明发散。根据前一部分观察到的方差，决定各实验使用多少随机种子。后续作业可使用调优后的学习率替代默认值。

**交付内容：** 最终验证奖励随学习率变化的图，以及几句话分析。

提示也是影响 RL 动态的选择：RL 强化模型生成的回答，因此提示会影响训练中探索哪些 rollout。下面探索提示的影响。

#### 题目（grpo_prompt_ablation）：提示消融（4 B200 小时，3 分）

使用 `question_only` 和 `r1_zero_three_shot`，各选几个随机种子运行 GRPO。与前面的零样本 `r1_zero` 相比，它们表现如何？哪种平均奖励最高、方差最低？其他记录指标是否存在系统性差异？根据运行间波动，你对结论有多大把握？

**交付内容：** 分析及相关指标图。

## 5 强化学习算法变体

GRPO 是语言模型 RL 的常用选择，但多篇论文对其算法设计提出了争论。本节学习这些选择背后的理论论据，并通过受控实验亲自判断哪种算法更好，至少对于本作业的模型和数据集组合如此。本节讨论同策略，下一节讨论异策略。

### 5.1 Dr. GRPO

推导过程中有两项设计，使 GRPO 估计器不再具有“正确”的期望 $\nabla_\theta J_\theta$，其中 $J_\theta$ 是期望奖励：按标准差归一化优势，以及按序列长度归一化。

首先考虑 Dr. GRPO 论文 [Z. Liu 等，2025] 主张的消融：撤销这两项修改，移除标准差归一化，并将总 loss 除以一个常数，而非先按各序列长度归一化。将该常数记为 $Z$，则：

$$
\hat g\leftarrow\frac1Z\sum_{i=1}^B\sum_{j=1}^G
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
(r(y^{(i,j)}\mid x^{(i)})-\mu_i)
\nabla_\theta\log\pi_\theta(y_t^{(i,j)}\mid x^{(i)},y_{<t}^{(i,j)}). \tag{33}
$$

$Z$ 通常取训练批次大小乘以最大生成长度，即 $Z=BGL$，其中 $L$ 是最大生成长度，本作业为 512。

#### 题目（think_about_length_normalization）：思考长度归一化（1 分）

实验前，思考按每个序列自身长度归一化，与所有序列使用同一常数归一化有何区别。各有什么优缺点？是否存在某种具体设置或例子，使一种方法显得更好？

**交付内容：** 几句话讨论。

#### 题目（compute_group_normalized_rewards_drgrpo）：Dr. GRPO 组归一化（0.5 分）

**交付内容：** 扩展 `compute_group_normalized_rewards`，支持 `advantage_normalizer="none"`。虽然 Dr. GRPO 使用 `baseline="mean"`，本题也请支持 `baseline="none"`。

接入 `adapters.run_compute_group_normalized_rewards`，运行并通过：

```bash
uv run pytest -k compute_group_normalized_rewards_drgrpo
```

#### 题目（aggregate_loss_across_microbatch_constant）：Dr. GRPO loss 聚合（0.5 分）

**交付内容：** 扩展 `aggregate_loss_across_microbatch`，支持 `loss_normalization="constant"`。

接入 `adapters.run_aggregate_loss_across_microbatch`，运行并通过：

```bash
uv run pytest -k test_aggregate_loss_across_microbatch_constant
```

### 5.2 拒绝采样微调

下一种消融是更简单的算法，通常称为拒绝采样微调（rejection fine tuning，RFT）或专家迭代（expert iteration，EI）。顾名思义，它采样一批 rollout，保留正确回答，再对正确回答做监督微调，即最小化下一词预测的对数损失。梯度为：

$$
\hat g\leftarrow\nabla_\theta\left[
\frac1Z\sum_{i=1}^B\sum_{j=1}^G
\mathbf1\{r(y^{(i,j)}\mid x^{(i)})=1\}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\log\pi_\theta(y_t^{(i,j)}\mid x^{(i)},y_{<t}^{(i,j)})\right]. \tag{34}
$$

$Z$ 与 Dr. GRPO 一样是常数，指示函数仅保留正确回答。下一题思考 RFT 梯度是否真的是策略梯度，即是否优化期望奖励 $J_\theta$，以及它与 GRPO 的关系。实现 PyTorch loss 时仍应返回目标的负值，使最小化 loss 对应梯度上升。

#### 题目（think_about_rft）：思考 RFT（2 分）

使用常数归一化的 RFT 目标为：

$$
J_\theta=\frac1Z\sum_x\sum_{j=1}^G
\mathbf1\{r(y^{(j)}\mid x)=1\}\log\pi_\theta(y^{(j)}\mid x). \tag{35}
$$

$x$ 为提示，每个 $y^{(j)}$ 独立采样自 $\pi_\theta(\cdot\mid x)$，$G$ 是生成数，$Z$ 为常数归一化因子，$r$ 为奖励函数。RFT 梯度为 $\nabla_\theta J_\theta$。

相比之下，同策略 Dr. GRPO，即常数归一化且不使用标准差归一化的 GRPO，其策略梯度估计器为：

$$
\frac1Z\sum_x\sum_{j=1}^G(r(y^{(j)}\mid x)-\mu)
\nabla_\theta\log\pi_\theta(y^{(j)}\mid x), \tag{36}
$$

其中 $\mu=\frac1G\sum_{j=1}^G r(y^{(j)}\mid x)$ 为组均值。假设奖励是二元值，比较这两个目标：期望是否相同？哪一个预计方差更低？直观上，什么情况下会偏好其中一种？

**交付内容：** 几句话讨论。

### 5.3 MaxRL

最后研究近期方法“最大似然强化学习”（Maximum Likelihood Reinforcement Learning，MaxRL）[F. Tajwar 等，2026]。它既不像 GRPO 那样除以组标准差，也不像 Dr. GRPO 那样不归一化，而是除以组均值：

$$
\hat g\leftarrow\frac1Z\sum_{i=1}^B\sum_{j=1}^G
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\frac{r(y^{(i,j)}\mid x^{(i)})-\mu_i}{\mu_i}
\nabla_\theta\log\pi_\theta(y_t^{(i,j)}\mid x^{(i)},y_{<t}^{(i,j)}). \tag{37}
$$

原始 MaxRL 论文并未使用常数 $Z$，而是除以批次 token 总数 $\sum_{i=1}^B\sum_{j=1}^G\operatorname{len}(y^{(i,j)})$，这个量随批次变化。本作业改用常数，减少与其他基线之间同时变化的因素，从而更容易单独研究用 $\mu_i$ 替代 $\mathrm{std}_i$ 的影响。

下面用基本数学分析不同优势归一化器的效果。直观上，除以平均奖励 $\mu_i$ 会提高困难提示的平均权重，因为它们通常比简单提示具有更低的 $\mu_i$。可将此形式化为重新加权的期望奖励：

$$
J_{\theta,w}=\mathbb E_{x\sim\rho}
\left[w(x,\operatorname{stopgrad}(\pi_\theta))
\mathbb E_{y\sim\pi_\theta}r(y\mid x)\right]. \tag{38}
$$

$w(x,\operatorname{stopgrad}(\pi_\theta))$ 对提示重新加权，例如根据当前策略在提示上的期望奖励所体现的难度 $\eta(x)=\mathbb E_{y\sim\pi_\theta}r(y\mid x)$。`stopgrad` 表示不对加权函数传播梯度。下一题推导各归一化器隐含的难度加权。

#### 题目（derive_difficulty_reweightings）：推导不同优势归一化器带来的难度加权（6 分）

跨提示聚合的标准策略梯度为：

$$
\nabla_\theta J_\theta=\nabla_\theta\mathbb E_{x\sim\rho}
[\mathbb E_{y\sim\pi_\theta}r(y\mid x)]. \tag{39}
$$

$\rho$ 是提示分布，$\pi_\theta$ 是策略，$r(y\mid x)$ 表示回答是否正确。本题推导 GRPO 各变体优化的代理目标，其形式为：

$$
\nabla_\theta J_{\theta,w}=\nabla_\theta\mathbb E_{x\sim\rho}
[w(x,\operatorname{stopgrad}(\pi_\theta))\mathbb E_{y\sim\pi_\theta}r(y\mid x)], \tag{40}
$$

其中 $w$ 为提示加权函数，求导不经过 $w$。

**（a）** Dr. GRPO 的策略梯度估计器为：

$$
\mathbb E_{x\sim\rho}\left[\frac1Z\sum_{j=1}^G
(r(y^{(j)}\mid x)-\mu)\nabla_\theta\log\pi_\theta(y^{(j)}\mid x)\right]. \tag{41}
$$

$\mu=\frac1G\sum_{j=1}^G r(y^{(j)}\mid x)$。令 $Z=G$，并取组大小 $G\to\infty$。什么加权函数 $w$ 使该估计器等价于优化代理目标 $J_{\theta,w}$？

**交付内容：** 由题目参数的某个子集表示的表达式，以及几句话论证。

**（b）** 在常数归一化下，GRPO 比 Dr. GRPO 多除以一个组标准差：

$$
\mathbb E_{x\sim\rho}\left[\frac1Z\sum_{j=1}^G
\frac{r(y^{(j)}\mid x)-\mu}{\mathrm{std}}
\nabla_\theta\log\pi_\theta(y^{(j)}\mid x)\right]. \tag{42}
$$

令 $Z=G$ 且 $G\to\infty$，对应的 $w$ 是什么？

**交付内容：** 由题目参数的某个子集表示的表达式，以及几句话论证。

**（c）** MaxRL 除以组均值：

$$
\mathbb E_{x\sim\rho}\left[\frac1Z\sum_{j=1}^G
\frac{r(y^{(j)}\mid x)-\mu}{\mu}
\nabla_\theta\log\pi_\theta(y^{(j)}\mid x)\right]. \tag{43}
$$

令 $Z=G$ 且 $G\to\infty$，对应的 $w$ 是什么？

**交付内容：** 由题目参数的某个子集表示的表达式，以及几句话论证。

#### 题目（think_about_advantage_normalization）：思考优势归一化（2 分）

实验前，思考用组标准差、组均值归一化优势，以及完全不归一化的差异。各有什么优缺点？什么具体设置或例子下，一种方法可能更好？

**交付内容：** 几句话讨论。

#### 题目（compute_group_normalized_rewards_maxrl）：MaxRL 组归一化（0.5 分）

**交付内容：** 扩展 `compute_group_normalized_rewards`，支持 `advantage_normalizer="mean"`。与标准差归一化一样，要在分母上加 `advantage_eps`，避免除零。

接入 `adapters.run_compute_group_normalized_rewards`，运行并通过：

```bash
uv run pytest -k compute_group_normalized_rewards_maxrl
```

### 5.4 实验

现在可以实验比较这些策略梯度估计器。首先扩展 `grpo_train_step`，支持已实现组件对应的变体，设置如下：

| 变体 | `baseline` | `advantage_normalizer` | `loss_normalization` |
|---|---|---|---|
| GRPO_constant | `"mean"` | `"std"` | `"constant"` |
| Dr_GRPO | `"mean"` | `"none"` | `"constant"` |
| RFT | `"none"` | `"none"` | `"constant"` |
| MaxRL（常数归一化） | `"mean"` | `"mean"` | `"constant"` |

为了加快训练，计算出归一化优势后，优势为零的序列不必送入模型，因为其梯度贡献为零。二元奖励下，有两种情况：

- `baseline="mean"` 时，一组内所有回答奖励相同，则整组优势为零。
- `baseline="none"` 时，奖励为零的序列优势为零。

删去零优势序列后，也可同比例减少 `gradient_accumulation_steps`，保持微批次大小不变。例如一半序列优势为零，可将 $k$ 次累积减少为 $k/2$ 次。这对有大量零奖励序列的 RFT 尤其有用。但必须仔细处理数学和实现，保证剪枝前后的梯度相同。

#### 题目（grpo_train_step_variants_on_policy）：GRPO 训练步骤变体（2.5 分）

**交付内容：** 扩展 `grpo_train_step`，支持全部同策略变体，仍使用 `importance_reweighting_method="none"`。需要支持 `baseline: Literal["mean", "none"]`、`advantage_normalizer: Literal["std", "none", "mean"]` 和 `loss_normalization: Literal["sequence", "constant"]`。还应避免将零优势序列送入模型，以加速训练。

特别地，`baseline="none"` 时，错误样本奖励为零，在 loss 中权重也为零，因此无须经过模型，可以据此优化实现。

接入 `adapters.run_grpo_train_step`，运行并通过：

```bash
uv run pytest -k test_grpo_train_step_variants_on_policy
```

实现完成后，运行实验，观察各方法是否优于标准 GRPO。为使结果可比，应使用与此前实验相同的超参数。

**关于超参数调优。** 判断方法优劣最严谨的实验方式，是分别为每个方法调参。但调参昂贵，本作业计算资源有限。更省算力的选择是只为基线调参，再把同一组超参数用于新方法，例如把标准 GRPO 调好的学习率用于以下实验。此时，若新方法优于基线，仍可认为它更好，因为非最优学习率下的结果是它充分调参后性能的下界；反之，若它更差，不能就此认定方法更差，因为没有为它充分调参。

#### 题目（grpo_experiments_variants_on_policy）：比较不同强化学习算法（8 B200 小时，10 分）

保持与标准 GRPO 相同的超参数，学习率可选用你调好的值。使用零样本 `r1_zero`，每个变体运行 4 个随机种子：

- **GRPO_constant**：标准 GRPO，但 loss 聚合使用常数归一化，而不是序列归一化。
- **Dr_GRPO**：在 GRPO_constant 基础上设置 `advantage_normalizer="none"`。
- **RFT**：在 GRPO_constant 基础上设置 `advantage_normalizer="none"`、`baseline="none"`。错误样本不必经过训练模型，应能提速。
- **MaxRL**：在 GRPO_constant 基础上设置 `advantage_normalizer="mean"`。原始 MaxRL 按微批次 token 数归一化 loss，本作业使用常数。

与标准 GRPO 相比，各方法如何？哪种平均表现最好、方差最低？哪些可能从进一步调参中受益？根据运行间方差，你对结论有多大把握？

**交付内容：** 分析，并用所引用指标的图支持。

## 6 异策略强化学习

此前所有实验和算法都是完全同策略的：每个梯度估计器使用的样本直接来自当前正在更新的模型，即每个推理批次只做一次训练更新，且训练批次大小等于推理批次大小。本节研究异策略 RL：每个推理批次做多次更新，有望加快训练，但可能降低稳定性、增加算法复杂度。

### 6.1 重要性加权

普通策略梯度先采样一批回答，再对整批回答做一次大批次梯度更新。我们可能希望把这一批分成多个小批次，每个小批次单独更新一次，从而加快学习。这就是异策略 RL，与每个推理批次只更新一次的同策略 RL 相对。

第一次小批次更新后，当前策略已不同于推理时的策略，样本因而成为“异策略”或“过时”样本，不再来自希望计算梯度的当前策略。用 $\pi_0$ 表示推理策略，$\pi_\theta$ 表示当前策略，则：

$$
\mathbb E_{x\sim\rho}\mathbb E_{y\sim\pi_0(y\mid x)}
[r(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)]
\ne\mathbb E_{x\sim\rho}\mathbb E_{y\sim\pi_\theta(y\mid x)}
[r(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)]. \tag{44}
$$

左侧是“朴素”异策略估计器：假装回答来自当前策略，机械地使用相同计算步骤。这个等式说明它的期望不正确。

一种纠偏方式是重要性加权：

$$
\mathbb E_{x\sim\rho}\mathbb E_{y\sim\pi_0(y\mid x)}
\left[\frac{\pi_\theta(y\mid x)}{\pi_0(y\mid x)}r(y\mid x)
\nabla_\theta\log\pi_\theta(y\mid x)\right]. \tag{45}
$$

权重 $\pi_\theta(y\mid x)/\pi_0(y\mid x)$ 提高对当前策略仍有相关性的回答的权重，降低当前策略赋予低概率的“过时”回答的权重。修改后具有正确的期望：

$$
\mathbb E_{x\sim\rho}\mathbb E_{y\sim\pi_0(y\mid x)}
\left[\frac{\pi_\theta(y\mid x)}{\pi_0(y\mid x)}r(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)\right]
=\mathbb E_{x\sim\rho}\mathbb E_{y\sim\pi_\theta(y\mid x)}
[r(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)]. \tag{46}
$$

缺点是方差增加：若重要性权重可大到 $C$，方差也可能放大 $C$ 倍。语言模型的权重是 $\operatorname{len}(y)$ 项的乘积，随回答长度呈指数级变化，使朴素重要性加权不适用：

$$
\frac{\pi_\theta(y\mid x)}{\pi_0(y\mid x)}
=\prod_{t=1}^{\operatorname{len}(y)}
\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_0(y_t\mid x,y_{<t})}. \tag{47}
$$

因此，朴素不加权的异策略估计器偏差高，但没有加权引入的额外方差；加权估计器无偏，却方差高。以下方法处于二者之间，用不同启发式方法权衡偏差和方差。

### 6.2 PPO／GRPO 风格的重要性加权与裁剪

#### 6.2.1 token 级加权

PPO [J. Schulman 等，2017] 首先提出、GRPO [Z. Shao 等，2024] 随后采用的标准做法，是用 token 级重要性加权替代序列级加权。未裁剪的 token 级估计器为：

$$
\mathbb E_{x\sim\rho}\mathbb E_{y\sim\pi_0(y\mid x)}
\left[r(y\mid x)\sum_{t=1}^{\operatorname{len}(y)}
\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_0(y_t\mid x,y_{<t})}
\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})\right]. \tag{48}
$$

它不同于此前有严格依据的序列级估计器，后者展开时间步后为：

$$
\mathbb E_{x\sim\rho}\mathbb E_{y\sim\pi_0(y\mid x)}
\left[r(y\mid x)
\left(\prod_{t=1}^{\operatorname{len}(y)}\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_0(y_t\mid x,y_{<t})}\right)
\sum_{t=1}^{\operatorname{len}(y)}\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})\right]. \tag{49}
$$

token 级加权项不再随回答长度指数变化，方差因而大幅降低。

为理解它，像之前一样推导其优化的代理目标 [T. Degris 等，2013]。定义代理策略 $\tilde\pi_t$：除了第 $t$ 步从当前策略 $\pi_\theta$ 采样，其他所有时间步都从旧策略 $\pi_0$ 采样：

$$
\tilde\pi_t(y\mid x)=
\left(\prod_{s=1}^{t-1}\pi_0(y_s\mid x,y_{<s})\right)
\pi_\theta(y_t\mid x,y_{<t})
\left(\prod_{s=t+1}^{\operatorname{len}(y)}\pi_0(y_s\mid x,y_{<s})\right). \tag{50}
$$

假定所有回答长度均为 $L$，token 级加权优化的是各个代理策略下的期望奖励之和：

$$
J_\theta^{\mathrm{token}}=\mathbb E_x
\left[\sum_{t=1}^L\mathbb E_{y\sim\tilde\pi_t(y\mid x)}[r(y\mid x)]\right]. \tag{51}
$$

将代理目标用重要性权重改写即可证明：

$$
\nabla_\theta J_\theta^{\mathrm{token}}
=\nabla_\theta\mathbb E_x\left[\sum_{t=1}^L
\mathbb E_{y\sim\tilde\pi_t(y\mid x)}[r(y\mid x)]\right] \tag{52}
$$
$$
=\nabla_\theta\mathbb E_x\left[\sum_{t=1}^L\mathbb E_{y\sim\pi_0(y\mid x)}
\left[\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_0(y_t\mid x,y_{<t})}r(y\mid x)\right]\right] \tag{53}
$$
$$
=\mathbb E_x\left[\mathbb E_{y\sim\pi_0(y\mid x)}\left[
\sum_{t=1}^L\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_0(y_t\mid x,y_{<t})}
r(y\mid x)\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})\right]\right]. \tag{54}
$$

最后一步再次使用对数导数技巧。

这个代理目标直观揭示了 token 级加权的偏差。第一，第 $t$ 项的前缀 $y_{<t}$ 来自旧策略，而非当前策略，未必代表当前模型的前缀。第二，第 $t$ 个 token 虽来自当前策略，后缀 $y_{>t}$ 却仍来自旧策略。也就是说，目标衡量动作对未来的影响时，依据的是旧策略行为，而不是当前策略行为。序列级加权会同时校正前缀和后缀，代价是方差增大。当前策略离旧策略越远，这两种偏差通常越严重。

#### 题目（derive_surrogate_objectives）：推导重要性加权方法的代理目标（2 分）

前面看到，token 级加权优化的代理策略仅在一个位置使用当前策略，其他位置都使用旧策略。现在考虑“成对”的重要性加权，其策略梯度估计器为：

$$
\sum_{t=1}^{L/2}
\frac{\pi_\theta(y_{2t-1}\mid x,y_{<2t-1})\pi_\theta(y_{2t}\mid x,y_{<2t})}
{\pi_0(y_{2t-1}\mid x,y_{<2t-1})\pi_0(y_{2t}\mid x,y_{<2t})}
r(y\mid x)\nabla_\theta\left[
\log\left(\pi_\theta(y_{2t-1}\mid x,y_{<2t-1})\pi_\theta(y_{2t}\mid x,y_{<2t})\right)\right]. \tag{55}
$$

$\pi_\theta$ 为当前策略，$\pi_0$ 为旧采样策略，$y\sim\pi_0(y\mid x)$。该估计器优化什么代理目标？

**交付内容：** 由题目参数的某个子集表示的表达式及推导。

#### 6.2.2 裁剪

PPO 和 GRPO 的另一项启发式是重要性权重裁剪。当前策略越偏离旧策略，重要性加权的方差越严重；代理目标也说明 token 级加权的偏差会增大。因此，为保持异策略 RL 稳定，需要让 $\pi_\theta$ 接近 $\pi_0$。

最简单的方法是减少每个推理批次的训练更新次数，但有些批次能支持更多次更新。为了充分利用算法，希望每批尽可能多更新。为此，一类方法会裁剪重要性权重过大或过小的项。

近期论文对裁剪细节存在分歧，本作业实现从 PPO 延续到 GRPO、目前似乎最常用的方法。先把 token 级加权与标准 GRPO 结合：

$$
J_\theta^{\mathrm{GRPO\text{-}off\text{-}policy\text{-}noclip}}
=\frac1{BG}\sum_{i=1}^B\sum_{j=1}^G
\frac1{\operatorname{len}(y^{(i,j)})}
\frac{r(y^{(i,j)}\mid x^{(i)})-\mu_i}{\mathrm{std}_i}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_0(y_t\mid x,y_{<t})}. \tag{56}
$$

样本来自旧推理策略 $\pi_0$。对其求导，得到前述 token 加权估计器，再加上 GRPO 优势和序列归一化。

令回答 $j$ 在提示 $i$ 上的优势为 $A^{(i,j)}=(r(y^{(i,j)}\mid x^{(i)})-\mu_i)/\mathrm{std}_i$，第 $t$ 个重要性权重为 $w_t^{(i,j)}=\pi_\theta(y_t\mid x,y_{<t})/\pi_0(y_t\mid x,y_{<t})$。裁剪目标为：

$$
J_\theta^{\mathrm{GRPO\text{-}off\text{-}policy\text{-}clip}}
=\frac1{BG}\sum_{i=1}^B\sum_{j=1}^G\frac1{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\min\left(A^{(i,j)}w_t^{(i,j)},
A^{(i,j)}\operatorname{clip}(w_t^{(i,j)},[1-\varepsilon,1+\varepsilon])\right). \tag{57}
$$

$\operatorname{clip}(w,[1-\varepsilon,1+\varepsilon])=\min(\max(w,1-\varepsilon),1+\varepsilon)$ 将权重限制在该区间。更直观的写法为 [J. Achiam，2018]：

$$
J_\theta^{\mathrm{GRPO\text{-}off\text{-}policy\text{-}clip}}
=\frac1{BG}\sum_{i=1}^B\sum_{j=1}^G\frac1{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\begin{cases}
\min(w_t^{(i,j)},1+\varepsilon)A^{(i,j)},&A^{(i,j)}\ge0,\\
\max(w_t^{(i,j)},1-\varepsilon)A^{(i,j)},&A^{(i,j)}<0.
\end{cases} \tag{58}
$$

对正优势动作，模型受到激励提高其概率，直到达到旧策略概率的 $1+\varepsilon$ 倍。此时 `min` 选择常数分支，梯度变成零。对负优势动作，模型受到激励降低其概率，直到达到旧策略概率的 $1-\varepsilon$ 倍，随后该动作的梯度也变成零。因此求导得到：

$$
\nabla_\theta J_\theta^{\mathrm{GRPO\text{-}off\text{-}policy\text{-}clip}}
=\frac1{BG}\sum_{i=1}^B\sum_{j=1}^G\frac1{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\mathrm{mask}_t^{(i,j)}w_t^{(i,j)}A^{(i,j)}
\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t}), \tag{59}
$$

掩码函数为：

$$
\mathrm{mask}_t^{(i,j)}=
\begin{cases}
\mathbf1\{w_t^{(i,j)}<1+\varepsilon\},&A^{(i,j)}\ge0,\\
\mathbf1\{w_t^{(i,j)}>1-\varepsilon\},&A^{(i,j)}<0.
\end{cases} \tag{60}
$$

它屏蔽正优势中重要性权重过大的项，以及负优势中权重过小的项。

现在具备实现 `importance_reweighting_method="noclip"` 和 `"grpo"` 的数学基础，分别对应未裁剪和裁剪目标。与同策略情况一样，loss 应返回目标负值，使梯度下降实现所需的梯度上升。

#### 题目（compute_policy_gradient_loss_off_policy）：带 token 级加权的异策略策略梯度（1 分）

**交付内容：** 扩展 `compute_policy_gradient_loss`，支持 `importance_reweighting_method="noclip"` 或 `"grpo"`，并使用 `old_log_probs`、`cliprange`。前者表示生成 rollout 时模型的逐 token 对数概率，需要在训练脚本中添加代码计算；后者表示裁剪强度 $\varepsilon$。

接入 `adapters.run_compute_policy_gradient_loss`，运行并通过：

```bash
uv run pytest -k test_compute_policy_gradient_loss_off_policy
```

注意：PPO 用裁剪限制新旧策略偏离，但它也可理解为通过压低过大的权重项来降低方差。如果不那么担心偏离旧策略，想采用更激进的更新，一个更简单直接的方差控制方式，是直接对梯度估计器中的重要性权重设置上界：

$$
\hat g\leftarrow\frac1{BG}\sum_{i=1}^B\sum_{j=1}^G\frac1{\operatorname{len}(y^{(i,j)})}
\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\min(w_t^{(i,j)},1+\varepsilon)A^{(i,j)}
\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t}). \tag{61}
$$

与 PPO/GRPO 不同，即使动作概率已提高超过 $1+\varepsilon$ 倍，这里仍有非零梯度。这种更简单的裁剪来自 CISPO [MiniMax 等，2025]。其原方法还按组内 token 数归一化，而非序列长度；这里为简洁仍使用序列归一化。你可以选择尝试 CISPO，并与其他裁剪方法比较。

### 6.3 GSPO

token 级加权忽略前缀和后缀的重新加权，因而有偏。GSPO 论文 [C. Zheng 等，2025] 的作者发现 GRPO 不稳定，主张改用序列级加权。为避免序列权重是 $L$ 项乘积带来的问题，他们将其取 $1/L$ 次方，即使用序列 token 重要性权重的几何均值，而非乘积。目标为：

$$
J_\theta^{\mathrm{GRPO\text{-}off\text{-}policy\text{-}gspo}}
=\frac1{BG}\sum_{i=1}^B\sum_{j=1}^G
\min\left(A^{(i,j)}s^{(i,j)},
A^{(i,j)}\operatorname{clip}(s^{(i,j)},[1-\varepsilon,1+\varepsilon])\right). \tag{62}
$$

序列级权重在各时间步共享：

$$
s^{(i,j)}=
\left(\prod_{t=1}^{\operatorname{len}(y^{(i,j)})}
\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_0(y_t\mid x,y_{<t})}\right)^{1/\operatorname{len}(y^{(i,j)})}. \tag{63}
$$

若去掉 $1/\operatorname{len}(y^{(i,j)})$ 次方和裁剪，就回到序列级重要性加权策略梯度 loss。几何均值和裁剪以引入偏差为代价降低方差。

另一个特点是，几何均值的指数使求导时自然出现序列长度归一化。为简洁忽略裁剪，有：

$$
\nabla_\theta J_\theta^{\mathrm{GRPO\text{-}off\text{-}policy\text{-}gspo}}
=\frac1{BG}\sum_{i=1}^B\sum_{j=1}^G A^{(i,j)}\nabla_\theta s^{(i,j)} \tag{64}
$$
$$
=\frac1{BG}\sum_{i=1}^B\sum_{j=1}^G A^{(i,j)}s^{(i,j)}
\frac1{\operatorname{len}(y^{(i,j)})}\sum_{t=1}^{\operatorname{len}(y^{(i,j)})}
\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t}). \tag{65}
$$

因此，若想将 GSPO 用于你喜欢的常数归一化 RL 算法（可选），就需要改变序列级重要性权重的指数。

#### 题目（think_about_importance_reweighting）：思考重要性加权（2 分）

考虑异策略 RL 的三种策略：（a）不做重要性加权；（b）PPO/GRPO 风格的带裁剪 token 级加权；（c）GSPO 风格、采用几何均值的带裁剪序列级加权。从偏差—方差权衡看，各方法处于什么位置？能否想到某些场景，使某种方法直观上优于其他方法？

**交付内容：** 几句话分析。

下面实现 GSPO loss，记得以数值稳定的方式计算几何均值。

#### 题目（compute_policy_gradient_loss_off_policy_gspo）：带序列级加权的异策略策略梯度（1 分）

**交付内容：** 扩展 `compute_policy_gradient_loss`，支持 `importance_reweighting_method="gspo"`，使用 `old_log_probs` 和 `cliprange`。前者是生成 rollout 时模型的逐 token 对数概率，需在训练脚本中添加计算；后者为裁剪强度 $\varepsilon$。仍返回目标负值，使最小化 loss 对应梯度上升。

接入 `adapters.run_compute_policy_gradient_loss`，运行并通过：

```bash
uv run pytest -k test_compute_policy_gradient_loss_off_policy_gspo
```

### 6.4 实验

现在可以将异策略组件接入 `grpo_train_step`。

#### 题目（grpo_train_step_off_policy）：异策略 GRPO 训练步骤（2.5 分）

**交付内容：** 扩展 `grpo_train_step`，支持 `importance_reweighting_method`、`old_log_probs`、`cliprange` 等异策略参数。此时函数应支持全部参数组合。

接入 `adapters.run_grpo_train_step`，运行并通过：

```bash
uv run pytest -k test_grpo_train_step_off_policy
```

异策略训练实际会提速吗？稳定性会损失多少？哪种加权／裁剪方式最好？下面实验研究这些问题。除改为每个推理批次进行 32 次更新，即“32 倍异策略”外，保持此前超参数：

```python
rollout_batch_size = 256
train_batch_size = 8
gradient_accumulation_steps = 1
```

裁剪范围可默认使用原论文中的值，也可自行调优：

- `offpolicy_clip`：`cliprange = 0.2`。
- `offpolicy_gspo`：`cliprange = 3e-4`。

还需在训练脚本中计算 `old_log_probs`，传入加权 loss。

#### 题目（grpo_experiments_off_policy）：比较不同异策略算法（8 B200 小时，10 分）

保持标准 GRPO 的超参数，学习率可选用你调好的值，使用零样本 `r1_zero`，每个变体运行 4 个随机种子。也可以用上一节中你喜欢的 RL 变体替代标准 GRPO：

- **offpolicy_naive**：训练和推理批次不再均为 256，将训练批次减至推理批次的 $1/32$，即 $256/32=8$。梯度累积步数也要同比例减少，以继续充分利用 GPU。保持 `importance_reweighting_method="none"`。
- **offpolicy_noclip**：在上一方法基础上设置 `importance_reweighting_method="noclip"`。
- **offpolicy_clip**：在朴素方法基础上设置 `importance_reweighting_method="grpo"`。
- **offpolicy_gspo**：在朴素方法基础上设置 `importance_reweighting_method="gspo"`。

除了已有指标，还必须记录裁剪比例（clip fraction）。

与此前完全同策略的 GRPO 相比，各方法表现如何？异策略如何影响训练稳定性和运行间方差？两种裁剪方法的裁剪比例有什么差异，哪种看起来更稳定？哪些方法可能从进一步调参中受益？根据运行间方差，你对结论有多大把握？

**交付内容：** 分析及相关指标图。

## 7 尝试自己的策略梯度估计器

#### 题目（try_your_own）：尝试自己的策略梯度估计器（10 分）

了解了文献中的多种估计器及其理论后，现在请提出自己的方法。可以考虑：

- 不同的优势估计器，从而改变提示之间的权重。
- 不同的重要性加权策略。
- 优势估计器中不同的奖励基线。

提出自己的策略梯度估计器，在相同的 OLMo-2-0425-1B／GSM8K RL 设置上运行，与此前方法比较。使用多个随机种子，固定问题设置以保证可比性。相对已尝试的方法，**只改变一个因素**：本作业介绍的方法中，至少有一个与你的方法仅存在一项区别。

修改背后的理论依据或直觉是什么？你的修改能否胜过基线？

**交付内容：** 分析及相关指标图；如果有助于论证，可以附数学推导。

## 参考文献

为便于检索，以下保留论文原名，并附中文释义。

1. J. Li 等，*DataComp-LM: In search of the next generation of training sets for language models*（DataComp-LM：寻找下一代语言模型训练集）。[论文](https://arxiv.org/abs/2406.11794)。
2. T. OLMo 等，*2 OLMo 2 Furious*（OLMo 2 技术报告）。[论文](https://arxiv.org/abs/2501.00656)。
3. K. Cobbe 等，*Training Verifiers to Solve Math Word Problems*（训练验证器解决数学应用题）。[论文](https://arxiv.org/abs/2110.14168)。
4. DeepSeek-AI 等，*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*（通过强化学习激发大语言模型的推理能力）。[论文](https://arxiv.org/abs/2501.12948)。
5. W. Kwon 等，*Efficient Memory Management for Large Language Model Serving with PagedAttention*（使用 PagedAttention 高效管理大语言模型服务的内存），2023。
6. Z. Liu 等，*Understanding R1-Zero-Like Training: A Critical Perspective*（理解类 R1-Zero 训练：批判性视角）。[论文](https://arxiv.org/abs/2503.20783)。
7. OpenAI 等，*OpenAI o1 System Card*（OpenAI o1 系统卡）。[论文](https://arxiv.org/abs/2412.16720)。
8. Z. Shao 等，*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*（推进开放语言模型的数学推理极限）。[论文](https://arxiv.org/abs/2402.03300)。
9. J. Achiam，*Spinning Up in Deep Reinforcement Learning*（深度强化学习入门），2018。
10. N. Lambert，*Reinforcement Learning from Human Feedback*（基于人类反馈的强化学习）。[在线书籍](https://rlhfbook.com/)。
11. R. J. Williams，*Simple statistical gradient-following algorithms for connectionist reinforcement learning*（用于联结主义强化学习的简单统计梯度跟随算法），*Machine Learning*，8 卷，3–4 期，229–256 页，1992。[DOI](https://doi.org/10.1007/BF00992696)。
12. F. Tajwar 等，*Maximum Likelihood Reinforcement Learning*（最大似然强化学习）。[论文](https://arxiv.org/abs/2602.02710)。
13. J. Schulman、F. Wolski、P. Dhariwal、A. Radford、O. Klimov，*Proximal Policy Optimization Algorithms*（近端策略优化算法）。[论文](https://arxiv.org/abs/1707.06347)。
14. T. Degris、M. White、R. S. Sutton，*Off-Policy Actor-Critic*（异策略 Actor-Critic）。[论文](https://arxiv.org/abs/1205.4839)。
15. J. Achiam，*Simplified PPO-Clip Objective*（简化的 PPO-Clip 目标）。[文档](https://drive.google.com/file/d/1PDzn9RPvaXjJFZkGeapMHbHGiWWW20Ey/view)。
16. MiniMax 等，*MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention*（借助 Lightning Attention 高效扩展测试时计算）。[论文](https://arxiv.org/abs/2506.13585)。
17. C. Zheng 等，*Group Sequence Policy Optimization*（组序列策略优化）。[论文](https://arxiv.org/abs/2507.18071)。
