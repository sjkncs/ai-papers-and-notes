下面用“**乐高积木**”来解读 AlphaCFG 的核心思想：

> **AlphaCFG = 用 CFG 规定哪些乐高可以拼、怎么拼；再用 MCTS 像聪明玩家一样试拼，寻找最赚钱的 alpha 因子。**

---

# 一、为什么传统遗传编程 GP 效果不好？

传统 alpha 挖掘中，遗传编程 GP 是常见方法：随机生成表达式，然后通过交叉、变异、选择来进化因子。

例如一个因子：

\[
\text{Rank}(\text{TsMean}(close, 20) / \text{TsStd}(volume, 10))
\]

GP 会随机组合算子和变量，像这样：

```text
Rank( TsMean(close, 20) / TsStd(volume, 10) )
```

## 1. 搜索空间巨大，而且很多表达式无意义

GP 的问题是它像“闭着眼睛乱拼乐高”。

假设你有这些积木：

- 原始数据：open, high, low, close, volume, vwap
- 时间序列算子：TsMean, TsStd, TsRank, Delay, Delta
- 截面算子：Rank, ZScore, Neutralize
- 算术算子：+, -, *, /, log, abs

理论上可以组合出无数表达式。

但很多表达式是非法或无意义的，例如：

```text
TsMean(Rank(close), 20)
```

这里 `Rank(close)` 是截面排序结果，再拿它做时间序列均值未必符合预期。

再比如：

```text
log(volume - TsMean(volume, 20))
```

可能出现负数，导致数值错误。

又比如：

```text
TsCorr(close, sector, 20)
```

`sector` 是分类变量，不能直接做时间序列相关。

GP 不理解这些约束，只会不断产生“坏积木组合”。

---

## 2. 语法不受控，容易产生垃圾因子

GP 的交叉和变异可能破坏原本合理的表达式。

例如两个父代：

```text
Rank(TsMean(close, 20))
TsCorr(volume, return, 10)
```

交叉后可能变成：

```text
Rank(TsCorr(volume))
```

参数数量不对。

或者：

```text
TsMean(close, return)
```

窗口参数应该是整数，但被换成了 return 序列。

这就像把乐高轮胎插到屋顶上，不是不能插，而是没有实际意义。

---

## 3. 进化效率低，计算成本高

GP 通常需要：

1. 生成大量随机因子；
2. 回测每个因子；
3. 选出表现好的；
4. 做交叉、变异；
5. 重复很多代。

问题是，回测 alpha 因子非常昂贵。

如果 90% 的候选因子都是无效或低质量的，计算资源就被严重浪费。

---

## 4. 容易过拟合

GP 倾向于生成复杂表达式。

复杂因子可能在训练集上 IC 很高，但样本外失效。

例如：

```text
Rank(
    TsCorr(
        TsMean(close / Delay(open, 3), 7),
        TsStd(volume - Delay(volume, 11), 13),
        17
    )
)
```

这种因子看起来很“高级”，但可能只是拟合噪声。

在量化中，一个简单稳定的因子往往比复杂脆弱的因子更有价值。

---

# 二、CFG 如何约束搜索空间？

CFG，全称 Context-Free Grammar，中文是“上下文无关文法”。

它本质上是一套生成规则：

> 什么东西可以放在哪里，什么表达式是合法的。

在 AlphaCFG 中，CFG 用来定义 alpha 因子的合法语法空间。

乐高类比：

- **原始行情数据**：基础积木；
- **算子**：连接件；
- **CFG 规则**：说明哪些积木能接在哪些位置；
- **合法 alpha 表达式**：拼好的模型。

---

# 1. 一个简单的因子语法示例

假设我们定义 alpha 表达式为 `<Alpha>`：

```text
<Alpha> ::= <CrossSectionOp>(<TimeSeriesFeature>)
          | <ArithmeticExpr>

<CrossSectionOp> ::= Rank | ZScore

<TimeSeriesFeature> ::= <TSOp>(<RawField>, <Window>)
                      | <BinaryTSOp>(<RawField>, <RawField>, <Window>)
                      | <RawField>

<TSOp> ::= TsMean | TsStd | TsRank | Delta | Delay

<BinaryTSOp> ::= TsCorr | TsCov

<ArithmeticExpr> ::= <Alpha> + <Alpha>
                   | <Alpha> - <Alpha>
                   | <Alpha> * <Alpha>
                   | SafeDiv(<Alpha>, <Alpha>)

<RawField> ::= open | high | low | close | volume | vwap | returns

<Window> ::= 5 | 10 | 20 | 60
```

这个文法可以生成：

```text
Rank(TsMean(close, 20))
```

也可以生成：

```text
ZScore(TsCorr(volume, returns, 10))
```

还可以生成：

```text
SafeDiv(Rank(TsMean(close, 20)), ZScore(TsStd(volume, 10)))
```

但它不会生成：

```text
TsMean(close, returns)
```

因为 `<Window>` 只能是整数集合 `{5, 10, 20, 60}`。

也不会生成：

```text
TsCorr(close, sector, 20)
```

因为 `sector` 没有被放进 `<RawField>` 或相关算子允许的输入类型中。

---

# 2. 更贴近量化的类型约束

真实因子搜索中，仅靠普通 CFG 可能还不够，通常还会加入类型系统。

比如把表达式分成：

```text
<Scalar>
<Series>
<CrossSectionalSeries>
<TimeSeries>
<Boolean>
<Category>
```

例如：

```text
<Alpha> ::= Rank(<Series>)
          | ZScore(<Series>)
          | Neutralize(<Series>, <Category>)

<Series> ::= TsMean(<Series>, <Window>)
           | TsStd(<Series>, <Window>)
           | Delta(<Series>, <Window>)
           | SafeDiv(<Series>, <Series>)
           | <Price>
           | <Volume>

<Price> ::= open | high | low | close | vwap

<Volume> ::= volume | amount

<Category> ::= industry | sector | country
```

这样可以合法生成：

```text
Neutralize(Rank(TsMean(close, 20)), industry)
```

含义是：

1. 计算 close 的 20 日均值；
2. 做截面排名；
3. 对行业中性化。

但不会生成：

```text
TsMean(industry, 20)
```

因为 `industry` 是分类变量，不是数值时间序列。

---

# 3. CFG 的核心价值

CFG 的作用不是帮你直接找到好因子，而是：

> 把“所有可能表达式”限制为“语法合法、经济含义合理、可计算、可回测”的表达式集合。

这相当于在一开始就砍掉大量无效搜索空间。

对比：

| 方法 | 搜索方式 | 问题 |
|---|---|---|
| 遗传编程 GP | 随机拼接表达式 | 容易产生非法、冗余、过拟合表达式 |
| CFG | 按规则生成表达式 | 保证合法性和结构可控 |
| AlphaCFG | CFG + MCTS | 在合法空间中高效搜索 |

---

# 三、MCTS 的四个步骤如何用于因子搜索？

MCTS 是蒙特卡洛树搜索，常见于围棋 AlphaGo。

在 AlphaCFG 中，一棵搜索树表示“因子表达式的逐步生成过程”。

---

# 1. 树节点是什么？

每个节点表示一个“部分生成的表达式”。

例如从起始符号开始：

```text
<Alpha>
```

展开一步：

```text
Rank(<Series>)
```

再展开：

```text
Rank(TsMean(<Price>, <Window>))
```

再展开：

```text
Rank(TsMean(close, <Window>))
```

最后：

```text
Rank(TsMean(close, 20))
```

这就是一个完整 alpha 因子。

乐高类比：

- 根节点：空地板；
- 中间节点：拼了一半的乐高结构；
- 叶子节点：完整乐高模型，也就是完整因子。

---

# 2. MCTS 的四个步骤

## Step 1：Selection，选择

从根节点开始，根据某种策略选择最值得继续探索的分支。

常用是 UCB/UCT：

\[
UCT = \bar{R}_i + c \sqrt{\frac{\log N}{n_i}}
\]

其中：

- \(\bar{R}_i\)：该分支过去生成因子的平均收益；
- \(N\)：父节点访问次数；
- \(n_i\)：子节点访问次数；
- \(c\)：探索强度。

含义：

- 如果某个分支过去表现好，继续利用；
- 如果某个分支访问少，也给它机会探索。

例如在根节点：

```text
<Alpha>
```

可以展开为：

```text
Rank(<Series>)
ZScore(<Series>)
Neutralize(<Series>, industry)
SafeDiv(<Series>, <Series>)
```

如果历史发现 `Rank(...)` 结构经常产生高 IC 因子，那么 MCTS 会更多探索它。

但它也不会完全放弃 `ZScore(...)`，因为可能隐藏好因子。

这就是 exploration vs exploitation。

---

## Step 2：Expansion，扩展

当走到一个还没有完全展开的节点时，选择一个 CFG 规则继续展开。

例如当前节点：

```text
Rank(<Series>)
```

可以把 `<Series>` 展开为：

```text
TsMean(<Series>, <Window>)
TsStd(<Series>, <Window>)
Delta(<Series>, <Window>)
SafeDiv(<Series>, <Series>)
close
volume
returns
```

选择其中一个：

```text
Rank(TsMean(<Series>, <Window>))
```

这相当于继续往乐高上加一块合法积木。

注意：因为扩展严格遵守 CFG，所以不会出现非法表达式。

---

## Step 3：Simulation / Rollout，模拟

从当前“半成品表达式”开始，随机或启发式地补全，直到得到完整因子。

例如当前节点：

```text
Rank(TsMean(<Series>, <Window>))
```

rollout 可能补成：

```text
Rank(TsMean(close, 20))
```

也可能补成：

```text
Rank(TsMean(volume, 10))
```

也可能补成：

```text
Rank(TsMean(returns, 5))
```

然后对完整因子进行评价。

评价指标可以是：

- IC 均值；
- RankIC；
- ICIR；
- 多空组合收益；
- Sharpe；
- turnover-adjusted return；
- 样本外表现；
- 与已有因子的相关性惩罚；
- 因子复杂度惩罚。

例如 reward 可以设计为：

\[
Reward = ICIR - \lambda_1 \cdot Turnover - \lambda_2 \cdot CorrPenalty - \lambda_3 \cdot Complexity
\]

这样搜索不会只追求样本内 IC，而会偏向稳定、低换手、低相关、简单的因子。

---

## Step 4：Backpropagation，回传

将这个完整因子的评分回传到路径上所有节点。

例如生成了：

```text
Rank(TsMean(close, 20))
```

其样本内 ICIR = 0.8。

那么以下节点都获得一次更新：

```text
<Alpha>
Rank(<Series>)
Rank(TsMean(<Series>, <Window>))
Rank(TsMean(close, <Window>))
Rank(TsMean(close, 20))
```

如果多次发现 `Rank(TsMean(price, window))` 结构表现好，那么未来 MCTS 会更偏向这类结构。

乐高类比：

> 如果你发现“底盘 + 轮子 + 车身”经常拼出好模型，下次你就更愿意先选底盘和轮子，而不是随机乱拼。

---

# 四、AlphaCFG 的整体流程

可以概括为：

```text
1. 设计 CFG：定义合法 alpha 表达式空间
2. 从 <Alpha> 根节点开始搜索
3. MCTS 逐步扩展表达式
4. Rollout 得到完整因子
5. 回测并计算 reward
6. reward 回传更新搜索树
7. 重复多轮
8. 输出高分 alpha 因子
9. 做样本外验证和组合构建
```

---

# 五、量化实战中如何使用？

AlphaCFG 更适合作为“因子工厂”，不是直接替代研究员。

实际流程可以这样做。

---

## 1. 先设计一个干净的语法空间

不要一开始就把所有算子都放进去。

建议按层次构建：

### 第一层：基础价量因子

```text
<Price> ::= open | high | low | close | vwap
<Volume> ::= volume | amount
```

### 第二层：常用时间序列变换

```text
<TS> ::= TsMean(<X>, <W>)
       | TsStd(<X>, <W>)
       | TsRank(<X>, <W>)
       | Delta(<X>, <W>)
       | Delay(<X>, <W>)
```

### 第三层：截面处理

```text
<Alpha> ::= Rank(<TS>)
          | ZScore(<TS>)
          | Neutralize(Rank(<TS>), industry)
```

### 第四层：组合结构

```text
<Alpha> ::= SafeDiv(<Alpha>, <Alpha>)
          | <Alpha> - <Alpha>
          | DecayLinear(<Alpha>, <W>)
```

窗口集合建议离散化：

```text
<W> ::= 3 | 5 | 10 | 20 | 60
```

这样搜索空间更可控。

---

## 2. Reward 设计非常关键

一个实用 reward 不能只看样本内 IC。

建议：

\[
Reward =
ICIR_{train}
+ a \cdot ICIR_{valid}
- b \cdot Turnover
- c \cdot Correlation
- d \cdot Complexity
- e \cdot Drawdown
\]

其中：

- `ICIR_train`：训练期 IC 稳定性；
- `ICIR_valid`：验证期泛化能力；
- `Turnover`：因子换手；
- `Correlation`：与已有因子库相关性；
- `Complexity`：表达式复杂度；
- `Drawdown`：多空组合回撤。

实战中，我会更重视：

1. 样本外 RankIC；
2. 分年份稳定性；
3. 与现有因子低相关；
4. 交易成本后收益；
5. 因子逻辑可解释。

---

## 3. 必须做严格的防过拟合

AlphaCFG 搜索能力强，也就更容易“挖到噪声”。

建议：

### 时间切分

```text
Train: 2016-2020
Validation: 2021-2022
Test: 2023-2024
Live paper trading: 2025
```

### 横截面切分

例如按股票池分组：

```text
Universe A: 沪深300
Universe B: 中证500
Universe C: 中证1000
```

一个好因子不应该只在某个股票池有效。

### 年度稳定性

不要只看总 IC。

要看每年：

```text
2018 IC = 0.04
2019 IC = 0.03
2020 IC = 0.05
2021 IC = -0.01
2022 IC = 0.04
```

如果某一年崩掉，要理解原因。

---

## 4. 输出因子后还要做人类审查

例如 AlphaCFG 找到：

```text
Rank(TsMean(volume, 20) / TsStd(close, 10))
```

不能只因为 IC 高就上线。

研究员要问：

1. 这个因子的经济含义是什么？
2. 是否只是微盘、市值、行业暴露？
3. 是否依赖停牌、涨跌停、流动性异常？
4. 是否存在未来函数？
5. 交易成本后还有效吗？
6. 与已有动量、反转、波动、流动性因子的相关性如何？

AlphaCFG 是强力搜索器，但不是最终裁判。

---

# 六、乐高积木类比总结

可以把整个方法想象成拼乐高赛车。

## 传统 GP

传统遗传编程像是：

> 把所有乐高积木倒在地上，然后随机拼接。

它可能拼出赛车，也可能拼出一个轮子插在屋顶、方向盘插在底盘下面的怪东西。

问题是：

- 很多拼法不合法；
- 很多拼法没意义；
- 浪费大量时间；
- 容易拼出看似复杂但不稳定的模型。

---

## CFG

CFG 像是一本乐高说明书，但不是指定唯一模型，而是规定：

- 轮子只能接到底盘；
- 车窗只能接到车身；
- 发动机不能接到车顶；
- 每辆车必须有底盘、轮子、车身。

对应到因子：

- 时间窗口必须是整数；
- Rank 作用于截面数值；
- TsMean 作用于时间序列；
- Neutralize 的第二个参数必须是行业等分类变量；
- 除法要用 SafeDiv；
- log 输入要保证正数或使用 safe log。

CFG 的作用是：

> 只允许拼出合法的 alpha 因子。

---

## MCTS

MCTS 像一个聪明的乐高玩家。

它不是随机拼，也不是死板照说明书拼。

它会：

1. 先看哪些结构过去容易拼出好赛车；
2. 在这些方向上多试；
3. 同时保留探索新结构的机会；
4. 每次拼完一辆车后测试速度；
5. 把测试结果反馈给前面的拼装决策。

对应 alpha：

1. 选择 promising 的表达式结构；
2. 扩展一个合法语法规则；
3. 补全成完整因子；
4. 回测计算 reward；
5. 把好坏反馈给搜索树。

---

# 七、一个完整例子

假设起始符号：

```text
<Alpha>
```

CFG 允许：

```text
<Alpha> ::= Rank(<Feature>)
<Feature> ::= TsMean(<Raw>, <W>) | TsStd(<Raw>, <W>) | Delta(<Raw>, <W>)
<Raw> ::= close | volume | returns
<W> ::= 5 | 10 | 20
```

MCTS 第一次生成：

```text
Rank(TsMean(close, 20))
```

回测发现 ICIR = 0.4。

第二次生成：

```text
Rank(TsStd(returns, 10))
```

ICIR = 0.9。

那么树会知道：

```text
Rank(TsStd(returns, <W>))
```

这个结构比较有希望。

于是后续会继续探索：

```text
Rank(TsStd(returns, 5))
Rank(TsStd(returns, 20))
Rank(TsStd(close, 10))
```

如果发现短期收益波动率高的股票未来表现差，那么可能得到一个有经济意义的低波动/反转类因子：

```text
-Rank(TsStd(returns, 10))
```

这比随机搜索更高效。

---

# 八、实战定位

AlphaCFG 的价值在于：

> 用结构化语法减少无效搜索，用 MCTS 提高搜索效率，用回测 reward 引导发现高质量 alpha。

它特别适合：

- 系统化挖掘价量因子；
- 扩展已有 WorldQuant 风格因子库；
- 发现非线性算子组合；
- 构建大量低相关候选 alpha；
- 作为研究员的自动化因子生成工具。

但它不能替代：

- 数据清洗；
- 防未来函数；
- 交易成本建模；
- 风险中性化；
- 样本外检验；
- 经济逻辑审查。

---

# 一句话总结

**AlphaCFG 就是：先用 CFG 制定“乐高拼装规则”，保证只拼合法 alpha；再用 MCTS 像 AlphaGo 一样在合法拼法中聪明探索，把回测表现好的结构不断强化，最终高效找到稳定、可解释、低相关的量化因子。**