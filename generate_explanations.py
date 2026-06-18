"""
用 GPT-5.5 + gpt-image-2 为论文生成解读和配图
"""

import json
import base64
import os
import requests
import sys

API_URL = "https://lanyiapi.com/v1"
API_KEY = "sk-vpBVC7bc3t9dDAVMPprGsxW4fEgQtuzn2lkMgrGW7SpQGRel"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1. GPT-5.5 论文解读
# ============================================================

def ask_gpt55(prompt: str) -> str:
    """调用 GPT-5.5 生成论文解读"""
    resp = requests.post(
        f"{API_URL}/chat/completions",
        headers=HEADERS,
        json={
            "model": "gpt-5.5-openai-compact",
            "messages": [
                {"role": "system", "content": (
                    "You are a senior quantitative researcher. "
                    "Explain AI papers clearly with intuition and examples. "
                    "Use Chinese for explanations. Be concise but insightful."
                )},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1500
        },
        timeout=120
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ============================================================
# 2. gpt-image-2 配图生成
# ============================================================

def generate_diagram(prompt: str, filename: str) -> str:
    """用 gpt-image-2 生成论文架构图"""
    resp = requests.post(
        f"{API_URL}/images/generations",
        headers=HEADERS,
        json={
            "model": "gpt-image-2",
            "prompt": prompt,
            "size": "1536x1024",
            "n": 1
        },
        timeout=180
    )
    resp.raise_for_status()
    data = resp.json()
    
    b64 = data["data"][0]["b64_json"]
    img_bytes = base64.b64decode(b64)
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(img_bytes)
    
    print(f"  [OK] Saved: {filename} ({len(img_bytes)//1024}KB)")
    return filepath


# ============================================================
# 3. 论文配图 Prompt 定义
# ============================================================

PAPERS = [
    {
        "name": "AlphaCFG",
        "arxiv": "2601.22119",
        "gpt55_prompt": (
            "请用中文深入解读 AlphaCFG 论文 (arXiv:2601.22119)。\n\n"
            "核心方法: 用上下文无关文法(CFG)定义合法alpha因子的语法空间, "
            "再用蒙特卡洛树搜索(MCTS)在这个空间中搜索最优因子。\n\n"
            "请解释:\n"
            "1. 为什么传统方法(遗传编程)效果不好?\n"
            "2. CFG 如何约束搜索空间? 给出具体因子语法示例。\n"
            "3. MCTS 的四个步骤(Selection/Expansion/Simulation/Backpropagation)如何应用在因子搜索中?\n"
            "4. 量化实战中如何使用这个框架?\n\n"
            "请用一个直觉化的类比来解释整个方法。"
        ),
        "image_prompt": (
            "A clean technical architecture diagram for AlphaCFG system. "
            "The diagram shows a pipeline from left to right:\n\n"
            "LEFT SECTION labeled 'Context-Free Grammar': "
            "A tree structure showing grammar rules like EXPR -> FUNC(FEATURE, WINDOW), "
            "with example factor expressions like 'ts_mean(close, 20)' and 'rank(ts_delta(high, 5))'.\n\n"
            "MIDDLE SECTION labeled 'MCTS Search': "
            "A Monte Carlo tree with nodes showing factor candidates, "
            "with arrows showing Selection (UCB1), Expansion (grammar rules), "
            "Simulation (IC evaluation), Backpropagation (reward update). "
            "Highlight the best path in gold color.\n\n"
            "RIGHT SECTION labeled 'Alpha Factor Library': "
            "Top-K discovered factors ranked by IC score, "
            "with bar chart showing IC values.\n\n"
            "Style: Professional technical diagram, dark blue background, "
            "white and cyan text, clean boxes and arrows, "
            "all labels in English. No Chinese characters."
        ),
        "image_file": "diagram_alphacfg.png"
    },
    {
        "name": "Hybrid Transformer GNN",
        "arxiv": "2601.04602",
        "gpt55_prompt": (
            "请用中文解读 Hybrid Transformer GNN 论文 (arXiv:2601.04602)。\n\n"
            "核心方法: 用 Transformer 捕捉时序非平稳性 + 图注意力网络建模股票间关系, "
            "在 Fisher-z 空间预测相关性残差。\n\n"
            "请解释:\n"
            "1. 为什么预测相关性比预测收益更难? 为什么重要?\n"
            "2. Fisher-z 变换的直觉是什么?\n"
            "3. 为什么用 GNN 而不是简单的 MLP? 图结构如何编码行业信息?\n"
            "4. 这个模型如何改善组合构建?\n\n"
            "用一个天气预测的类比来解释。"
        ),
        "image_prompt": (
            "A professional architecture diagram for a Hybrid Transformer-GNN model "
            "for stock correlation forecasting. "
            "The diagram flows from bottom to top:\n\n"
            "BOTTOM LAYER labeled 'Input Features': "
            "Multiple parallel time series showing daily returns, technical indicators, "
            "sector labels, and historical correlations for S&P 500 stocks. "
            "Small sparkline charts for each feature.\n\n"
            "MIDDLE-LEFT labeled 'Temporal Encoder (Transformer)': "
            "A standard transformer block with self-attention mechanism, "
            "showing how temporal dependencies are captured across time steps. "
            "Color: blue gradient.\n\n"
            "MIDDLE-RIGHT labeled 'Graph Attention Network': "
            "A graph where nodes are stocks connected by sector relationships, "
            "with attention weights shown as edge thickness. "
            "Tech stocks clustered together, finance stocks in another cluster. "
            "Color: green gradient.\n\n"
            "TOP LAYER labeled 'Fisher-z Residual Prediction': "
            "A correlation matrix heatmap showing predicted vs actual correlations, "
            "with an arrow pointing to 'Portfolio Construction'.\n\n"
            "Style: Clean technical diagram, dark navy background, "
            "neon blue and green accents, all text in English."
        ),
        "image_file": "diagram_correlation_gnn.png"
    },
    {
        "name": "SPO Portfolio",
        "arxiv": "2601.04062",
        "gpt55_prompt": (
            "请用中文解读 Smart Predict-then-Optimize (SPO) 论文 (arXiv:2601.04062)。\n\n"
            "核心方法: 传统量化是两阶段(先预测收益->再优化组合), "
            "SPO把学习目标直接对齐到组合决策质量(端到端)。\n\n"
            "请解释:\n"
            "1. 两阶段方法的根本缺陷是什么? (预测准≠决策好)\n"
            "2. SPO的损失函数如何设计? 如何嵌入交易成本?\n"
            "3. 为什么在市场摩擦大的环境中SPO优势更明显?\n"
            "4. 2020疫情期间SPO为什么表现更好?\n\n"
            "用GPS导航的类比来解释(预测路况 vs 直接优化路线)。"
        ),
        "image_prompt": (
            "A comparison diagram showing two approaches side by side: "
            "'Traditional Two-Stage' vs 'Smart Predict-then-Optimize (SPO)'. "
            "Split the image vertically in the middle.\n\n"
            "LEFT SIDE labeled 'Traditional (Two-Stage)':\n"
            "Step 1 box: 'Return Prediction' with MSE loss arrow pointing to it, "
            "showing a neural network predicting future returns.\n"
            "Step 2 box: 'Portfolio Optimization' with mean-variance formula.\n"
            "A red warning icon between steps showing 'Prediction gap: "
            "accurate forecast ≠ good portfolio'.\n"
            "Arrow showing transaction costs eating into returns.\n\n"
            "RIGHT SIDE labeled 'SPO (End-to-End)':\n"
            "Single unified pipeline where prediction parameters "
            "are trained directly on portfolio Sharpe ratio.\n"
            "Loss function box showing 'Loss = -Sharpe + Transaction Cost'.\n"
            "Green checkmark showing 'Objective aligned with decision quality'.\n\n"
            "BOTTOM: Performance comparison bar chart showing "
            "Sharpe ratio and Max Drawdown for both methods, "
            "with SPO outperforming.\n\n"
            "Style: Professional infographic, white background with subtle grid, "
            "red accents for traditional method, green for SPO. "
            "All text in English."
        ),
        "image_file": "diagram_spo_portfolio.png"
    },
    {
        "name": "Deep g-Pricing",
        "arxiv": "2601.18804",
        "gpt55_prompt": (
            "请用中文解读 Deep g-Pricing 论文 (arXiv:2601.18804)。\n\n"
            "核心方法: 用深度学习替代 Black-Scholes, "
            "融合波动率轨迹+市场情绪(融资融券、北向资金)为CSI 300期权定价。\n\n"
            "请解释:\n"
            "1. Black-Scholes 在中国市场的局限是什么?\n"
            "2. g-生成器的数学直觉是什么?\n"
            "3. 为什么Call期权主要受情绪驱动, Put受波动+情绪双重驱动?\n"
            "4. MAE降低32.2%在实际交易中意味着什么?\n\n"
            "用天气预报(传统物理模型 vs AI模型)的类比来解释。"
        ),
        "image_prompt": (
            "A technical architecture diagram for Deep g-Pricing model "
            "for CSI 300 index options. "
            "The diagram has three horizontal layers:\n\n"
            "TOP INPUT LAYER labeled 'Market Signals':\n"
            "Three input boxes side by side:\n"
            "1. 'Option Parameters' showing Strike(K), Maturity(T), Type(Call/Put), "
            "Moneyness(K/S) with small option payoff diagrams\n"
            "2. 'Volatility Trajectory' showing a volatility surface with smile curve "
            "and term structure\n"
            "3. 'Sentiment Signals' showing margin trading balance change chart, "
            "northbound capital flow bar chart, and sentiment index gauge (0-100)\n\n"
            "MIDDLE LAYER labeled 'Deep g-Generator Network':\n"
            "A neural network with 3 hidden layers (64->32->16 neurons), "
            "showing nonlinear activation functions. "
            "Side annotation: 'g-function: nonlinear drift + diffusion'. "
            "Comparison with Black-Scholes formula shown faded in background.\n\n"
            "BOTTOM OUTPUT LAYER labeled 'Option Price':\n"
            "Two outputs side by side: "
            "'Black-Scholes Price' (with MAE=15.2 in red) vs "
            "'Deep g-Price' (with MAE=10.3 in green, showing 32.2% improvement).\n\n"
            "Style: Dark theme with blue and gold accents, "
            "professional financial tech aesthetic. All text in English."
        ),
        "image_file": "diagram_deep_g_pricing.png"
    },
    {
        "name": "Financial NER + LoRA",
        "arxiv": "2601.10043",
        "gpt55_prompt": (
            "请用中文解读 Financial NER 论文 (arXiv:2601.10043)。\n\n"
            "核心方法: 用 Instruction Fine-tuning + LoRA 适配 LLaMA-3-8B 做金融实体识别, "
            "在1693句金融语料上达到 0.894 micro-F1。\n\n"
            "请解释:\n"
            "1. 为什么金融NER比通用NER更难? 有哪些特殊挑战?\n"
            "2. Instruction Tuning 的作用是什么? prompt如何设计?\n"
            "3. LoRA 的原理是什么? 为什么只需要0.05%参数?\n"
            "4. 在量化研究中, 金融NER可以做什么? (事件驱动、知识图谱)\n\n"
            "用'给新员工做入职培训'的类比来解释 Instruction Tuning + LoRA。"
        ),
        "image_prompt": (
            "A pipeline diagram showing Financial NER with LLaMA-3-8B and LoRA. "
            "The diagram flows from left to right in 4 stages:\n\n"
            "STAGE 1 labeled 'Base Model':\n"
            "A large neural network icon representing LLaMA-3-8B (8 billion parameters), "
            "shown as a frozen block (blue/gray, lock icon). "
            "Text: '8B parameters, frozen'.\n\n"
            "STAGE 2 labeled 'Instruction Tuning':\n"
            "A document showing the prompt format:\n"
            "'Instruction: Identify financial entities...'\n"
            "'Input: Apple reported revenue of $85B...'\n"
            "'Output: B-COMPANY O O B-METRIC...'\n"
            "Arrows showing training data flowing in.\n\n"
            "STAGE 3 labeled 'LoRA Adaptation':\n"
            "A small matrix pair (A and B low-rank matrices, rank=16) "
            "being injected into the attention layers (q_proj, k_proj, v_proj, o_proj). "
            "Size comparison: tiny LoRA (0.8M params) vs huge base model (8B). "
            "Show the math: W' = W + BA, where B is d×r and A is r×d.\n\n"
            "STAGE 4 labeled 'Results':\n"
            "A financial sentence with color-coded entity highlights:\n"
            "'Apple'(blue=COMPANY) 'revenue'(green=METRIC) '$85B'(gold=MONEY).\n"
            "Performance badge: 'Micro-F1: 0.894'.\n\n"
            "Style: Clean white background, colorful entity highlights, "
            "modern tech infographic style. All text in English."
        ),
        "image_file": "diagram_financial_ner.png"
    }
]


# ============================================================
# 4. 主流程
# ============================================================

def main():
    print("=" * 60)
    print("GPT-5.5 + gpt-image-2: Paper Explanations & Diagrams")
    print("=" * 60)
    
    for i, paper in enumerate(PAPERS, 1):
        name = paper["name"]
        arxiv = paper["arxiv"]
        
        print(f"\n{'='*60}")
        print(f"[{i}/{len(PAPERS)}] {name} (arXiv:{arxiv})")
        print(f"{'='*60}")
        
        # --- GPT-5.5 解读 ---
        print(f"\n  [GPT-5.5] 生成论文解读...")
        try:
            explanation = ask_gpt55(paper["gpt55_prompt"])
            
            # 保存解读
            md_file = os.path.join(OUTPUT_DIR, f"explanation_{name.lower().replace(' ', '_')}.md")
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(f"# {name} 深度解读\n\n")
                f.write(f"**arXiv**: [{arxiv}](https://arxiv.org/abs/{arxiv})\n\n")
                f.write(explanation)
            print(f"  [OK] 解读已保存: explanation_{name.lower().replace(' ', '_')}.md")
            
        except Exception as e:
            print(f"  [ERR] GPT-5.5 调用失败: {e}")
            explanation = None
        
        # --- gpt-image-2 配图 ---
        print(f"\n  [gpt-image-2] 生成架构图...")
        try:
            filepath = generate_diagram(
                paper["image_prompt"],
                paper["image_file"]
            )
        except Exception as e:
            print(f"  [ERR] 图片生成失败: {e}")
    
    print(f"\n{'='*60}")
    print("All done! Files saved to project directory.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
