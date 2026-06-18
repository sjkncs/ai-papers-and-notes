"""补全剩余4篇论文解读 (每篇单独调用, 避免超时)"""
import json, os, sys, requests, time

API_URL = "https://lanyiapi.com/v1"
API_KEY = "sk-vpBVC7bc3t9dDAVMPprGsxW4fEgQtuzn2lkMgrGW7SpQGRel"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
OUT = os.path.dirname(os.path.abspath(__file__))

def ask(prompt, fname):
    fpath = os.path.join(OUT, fname)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 100:
        print(f"  [SKIP] {fname} already exists")
        return
    print(f"  [CALL] {fname} ...")
    resp = requests.post(f"{API_URL}/chat/completions", headers=HEADERS, json={
        "model": "gpt-5.5-openai-compact",
        "messages": [
            {"role": "system", "content": "You are a senior quant researcher. Explain papers in Chinese with intuition and examples. Be concise but deep."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3, "max_tokens": 2000
    }, timeout=120)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  [OK] {fname} ({len(text)} chars)")

PAPERS = [
    ("explanation_alphacfg.md",
     "请用中文深入解读 AlphaCFG 论文 (arXiv:2601.22119)。\n\n"
     "核心方法: 用上下文无关文法(CFG)定义合法alpha因子的语法空间, "
     "再用蒙特卡洛树搜索(MCTS)在这个空间中搜索最优因子。\n\n"
     "请解释:\n"
     "1. 为什么传统方法(遗传编程)效果不好?\n"
     "2. CFG 如何约束搜索空间? 给出具体因子语法示例。\n"
     "3. MCTS 的四个步骤如何应用在因子搜索中?\n"
     "4. 量化实战中如何使用?\n\n"
     "请用'乐高积木'的类比来解释整个方法。"),

    ("explanation_spo_portfolio.md",
     "# SPO Portfolio 深度解读\n\n"
     "**arXiv**: [2601.04062](https://arxiv.org/abs/2601.04062)\n\n"
     "请用中文解读 Smart Predict-then-Optimize (SPO) 论文。\n\n"
     "核心方法: 传统量化是两阶段(先预测收益->再优化组合), "
     "SPO把学习目标直接对齐到组合决策质量(端到端)。\n\n"
     "请解释:\n"
     "1. 两阶段方法的根本缺陷? (预测准≠决策好)\n"
     "2. SPO损失函数如何设计? 如何嵌入交易成本?\n"
     "3. 为什么市场摩擦大时SPO优势更明显?\n"
     "4. 2020疫情期间为何表现更好?\n\n"
     "用GPS导航类比(预测路况 vs 直接优化路线)。"),

    ("explanation_deep_g_pricing.md",
     "# Deep g-Pricing 深度解读\n\n"
     "**arXiv**: [2601.18804](https://arxiv.org/abs/2601.18804)\n\n"
     "请用中文解读 Deep g-Pricing 论文。\n\n"
     "核心: 用深度学习替代BS, 融合波动率轨迹+市场情绪为CSI300期权定价, MAE降32.2%。\n\n"
     "请解释:\n"
     "1. BS在中国市场的局限?\n"
     "2. g-生成器的数学直觉?\n"
     "3. 为何Call主要受情绪驱动, Put受波动+情绪双重驱动?\n"
     "4. MAE降32%实际意味着什么?\n\n"
     "用天气预报(物理模型 vs AI模型)类比。"),

    ("explanation_financial_ner.md",
     "# Financial NER + LoRA 深度解读\n\n"
     "**arXiv**: [2601.10043](https://arxiv.org/abs/2601.10043)\n\n"
     "请用中文解读 Financial NER 论文。\n\n"
     "核心: Instruction Tuning + LoRA 适配 LLaMA-3-8B 做金融NER, 1693句达0.894 F1。\n\n"
     "请解释:\n"
     "1. 金融NER比通用NER难在哪?\n"
     "2. Instruction Tuning作用? prompt如何设计?\n"
     "3. LoRA原理? 为何只需0.05%参数?\n"
     "4. 量化中金融NER能做什么?\n\n"
     "用'给新员工做入职培训'类比。")
]

for fname, prompt in PAPERS:
    try:
        ask(prompt, fname)
    except Exception as e:
        print(f"  [ERR] {fname}: {e}")
    time.sleep(2)

print("\nDone!")
