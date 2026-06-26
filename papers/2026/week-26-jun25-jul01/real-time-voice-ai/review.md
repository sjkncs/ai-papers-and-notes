# Real-Time Voice AI Hears but Does Not Listen

**Authors / 作者：** Martijn Bartelds, Federico Bianchi, James Zou  
**arXiv / 链接：** [https://arxiv.org/abs/2606.26083](https://arxiv.org/abs/2606.26083)  
**Published / 发表时间：** 2026-06-24 (arXiv announcement ~2026-06-25)  
**Venue / 会议：** TBA

---

## 摘要 / Abstract（原文 verbatim）

> Speech conveys information through both words and vocal delivery. We evaluate four leading production realtime voice systems—OpenAI's GPT Realtime 2, Google's Gemini 3.1 Flash Live, and Alibaba's Qwen3.5 Omni Plus and Omni Flash—on tasks where the words and the delivery patterns both convey meaningful information. Across three consequential scenarios, all four systems act on the words rather than the voice. They end calls with crying callers who insist nothing is wrong, approve wire transfers authorized in frightened voices, and enroll callers whose agreement is clearly sarcastic. Surprisingly, this is often not a failure of perception. When asked directly, three of the four systems reliably identify the distress, fear, or sarcasm they later ignore when making decisions. We observe a similar pattern when these realtime voice systems estimate accent and age, as their responses frequently follow the biases of the words rather than the acoustic properties of the speaker. We term this disconnect between perception and action the emotional intelligence gap of voice AI. Prompting systems to explicitly attend to vocal delivery improves performance only partially and inconsistently. Our findings show that current realtime voice AI systems often behave as if speech had been reduced to a transcript, suggesting that they should be used with caution in settings where the tone and emotion of delivery convey important information.

---

## Q1：它真正想解决的问题是什么？ / What problem does it really want to solve?

**中文：** 实时语音 AI（real-time voice AI）已经广泛应用于客服、金融、医疗、紧急服务等高风险场景。传统评测多关注 ASR 准确率、响应延迟、自然度等指标，却忽略了一个更根本的问题：**系统是否能将“听到的情绪/语气”真正转化为“行动的考量”**？

本文发现，主流实时语音系统虽然在直接提问时能识别情绪（哭泣、恐惧、讽刺），但在做决策时却像只读了转录文本一样忽略这些信息。作者把这种“能感知但不行动”的断裂命名为**语音 AI 的情感智能鸿沟（emotional intelligence gap）**。论文要解决的正是：如何诊断、量化并缩小这一鸿沟，以避免在高风险场景中造成实际伤害。

**English:** Real-time voice AI is already deployed in high-stakes settings such as customer service, finance, healthcare, and emergency response. Traditional evaluations focus on ASR accuracy, latency, and naturalness, but overlook a more fundamental question: **can the system translate heard emotion/prosody into action?**

The paper finds that leading real-time voice systems can identify emotions (distress, fear, sarcasm) when asked directly, yet ignore that same information when making decisions—as if they had only read a transcript. The authors name this "perception without action" disconnect the **emotional intelligence gap of voice AI**. The paper aims to diagnose, quantify, and narrow this gap to prevent real-world harm in consequential scenarios.

---

## Q2：它声称的贡献是什么？ / What does it claim to contribute?

**中文：**

1. **现象发现与概念命名**：首次系统性地揭示主流实时语音 AI 的“情感智能鸿沟”——能感知情绪，却不依据情绪行动。
2. **多系统、多场景评估**：覆盖 OpenAI、Google、阿里巴巴四款 production 级实时语音系统，在三个高风险场景（自杀/危机干预、银行电汇授权、讽刺性同意）中展示一致缺陷。
3. **跨维度验证**：不仅在情绪决策任务上，还在口音、年龄估计等任务中发现系统输出受语言内容偏见主导，而非声学特征。
4. **提示工程边界**：显示简单提示“请关注语气”只能部分且不稳定地改善问题，说明该鸿沟并非仅靠 prompt 就能根除。
5. **安全警示**：明确提出在高风险场景中应谨慎使用当前实时语音 AI，因为系统倾向于把语音还原为文本。

**English:**

1. **Phenomenon and terminology**: First systematic demonstration of the "emotional intelligence gap" in mainstream real-time voice AI—systems perceive emotion but do not act on it.
2. **Multi-system, multi-scenario evaluation**: Evaluates four production real-time voice systems from OpenAI, Google, and Alibaba across three consequential scenarios (crisis intervention, bank wire authorization, sarcastic consent).
3. **Cross-domain validation**: Shows the same pattern extends to accent and age estimation, where model responses follow the biases of words rather than acoustic properties.
4. **Limits of prompting**: Shows that simply prompting systems to attend to vocal delivery improves performance only partially and inconsistently, implying the gap is not easily promptable away.
5. **Safety warning**: Explicitly cautions against deploying current real-time voice AI in settings where tone and emotion convey important information.

---

## Q3：最可能被 reviewer 攻击的地方在哪里？ / Where are reviewers most likely to attack?

**中文：**

1. **黑盒系统的可控性有限**：四款系统都是闭源 API，作者无法访问内部架构、训练数据、系统提示。Reviewer 会质疑：观察到的“情感智能鸿沟”究竟是模型本身的缺陷，还是厂商系统提示 / 安全过滤 / 后处理的结果？
2. **任务设计的生态效度（ecological validity）**：论文中的“哭泣坚持没事”“恐惧中授权转账”“讽刺同意”是人为构造的极端场景。真实世界中的语音交互更复杂，reviewer 可能质疑这些场景在多大程度上能推广到实际部署。
3. **基线与对比不足**：没有与专门的语音情感识别系统、人类表现、或经过语音情感微调的模型做系统对比。如果专用模型表现更好，那么问题可能不是“语音 AI 没情感智能”，而是“通用语音 LLM 没针对情感决策优化”。
4. **“能感知但不行动”的因果解释较弱**：论文展示了三款系统在被直接问时能识别情绪，但做决策时忽略。这种 disconnect 可能源于训练目标（RLHF 更关注文本奖励）、上下文长度、系统提示优先级等多种因素，论文对这些机制的剖析可能不够深入。
5. **提示策略不够穷尽**：作者尝试“请关注意气”但未穷尽其他方法（如 few-shot 示例、链式思考、情绪摘要前置）。Reviewer 可能认为改进空间未被充分探索。

**English:**

1. **Limited controllability of black-box systems**: The four systems are closed-source APIs. Reviewers will ask whether the observed emotional intelligence gap is a property of the models themselves or a consequence of vendor system prompts, safety filters, or post-processing.
2. **Ecological validity of task design**: The scenarios (crying caller insisting nothing is wrong, frightened wire authorization, sarcastic consent) are deliberately extreme. Reviewers may question how well they generalize to real-world deployment.
3. **Insufficient baselines and comparisons**: The paper does not systematically compare against dedicated speech-emotion recognition systems, human performance, or models fine-tuned on emotional speech. If specialized models perform better, the problem may be "general voice LLMs are not optimized for emotional decision-making" rather than "voice AI lacks emotional intelligence."
4. **Weak causal explanation for the perception-action disconnect**: The paper shows that three systems identify emotion when asked directly but ignore it when deciding. This disconnect could stem from training objectives, context length, system-prompt priorities, or other factors, and the paper's mechanistic analysis may be shallow.
5. **Prompting strategies may not be exhaustive**: The authors test "explicitly attend to vocal delivery" but do not explore alternatives such as few-shot examples, chain-of-thought, or prepending emotion summaries. Reviewers may argue the improvement space is underexplored.

---

## Q4：同方向博士生应精读哪些、跳过哪些？ / What should PhD students in this direction read carefully vs. skip?

**中文：**

**必读（精读）：**
- **实验设计与场景构建**：本文如何把抽象的“情感智能”转化为可量化、可复现的语音任务，是安全评估类论文的范本。
- **结果呈现方式**：尤其是“直接提问时能识别情绪 / 做决策时忽略情绪”的对比，是本文最具冲击力的发现，学习如何用数据讲一个清晰的故事。
- **Related Work 中关于多模态对齐（multimodal alignment）的梳理**：理解为何文本模态会“压倒”声学模态。

**延伸阅读（按兴趣）：**
- **语音情感识别（SER）与副语言（paralinguistic）理解**：如 wav2vec 2.0 / HuBERT 在情感任务上的微调研究。
- **多模态融合与模态偏见（modality bias）**：如 MLLM 中视觉被语言先验主导的类似研究（可与本文形成类比）。
- **AI 安全与 red-teaming 方法论**：学习如何对 production 黑盒 API 进行系统性审计。

**可以跳过：**
- 如果你从事**底层语音表示学习**，本文没有提出新的模型或训练方法，重点在评测与现象揭示，方法细节不是你的重点。
- 如果你需要**可立即部署的解决方案**，本文只指出问题并给出有限的提示工程结果，没有提供可工程化的修复方案。

**English:**

**Must-read (close reading):**
- **Experimental design and scenario construction**: The paper shows how to turn the abstract notion of "emotional intelligence" into quantifiable, reproducible speech tasks—a model for safety-evaluation papers.
- **Result presentation**: The contrast between "identifies emotion when asked directly" and "ignores emotion when deciding" is the paper's most striking finding; learn how to tell a clear story with data.
- **Related work on multimodal alignment**: Understand why the text modality tends to dominate the acoustic modality.

**Further reading (by interest):**
- **Speech emotion recognition (SER) and paralinguistic understanding**: e.g., wav2vec 2.0 / HuBERT fine-tuning for emotion tasks.
- **Multimodal fusion and modality bias**: e.g., studies showing vision being dominated by language priors in MLLMs (analogous to this paper).
- **AI safety and red-teaming methodology**: Learn how to systematically audit production black-box APIs.

**Can skip:**
- If you work on **low-level speech representation learning**, this paper proposes no new model or training method; it focuses on evaluation and phenomenon disclosure, so method details are not your priority.
- If you need **an immediately deployable fix**, the paper only identifies the problem and provides limited prompting results; it does not offer an engineering-ready solution.
