# 1️⃣ 食品安全 Agent 框架 / Food-Safety Agent Frameworks

> 专门用于食品行业（HACCP、合规、检测、召回）的 AI Agent 框架与仓库。

## 概览 / Overview

| 仓库 | ⭐ Stars | 语言 | 核心框架 | 场景 |
|---|---:|---|---|---|
| [Sharma-Aditya7/multiagent-food-compliance](#1-safeguard-ai) | 5 | Python | CrewAI + Groq | HACCP 多 Agent 审计 |
| [ovitrac/SFPPy](#2-sfppy) | 4 | Python | LLM-ready API | 食品接触合规 + 迁移建模 |
| [AtharvaVavhal/FoodSafe](#3-foodsafe) | 1 | Python+React | YOLOv8 + IndicBERT + Prophet | 印度家庭掺假检测 |
| [phanben110/KAD-FoodHazard](#4-kad-foodhazard) | 5 | Python | RAG + LLM | SemEval-2025 Top-2 |
| [BioinfoMachineLearning/SERSFormer](#5-sersformer) | 3 | Python | Transformer | SERS 农药残留检测 |
| [afrexai-food-safety](#6-afrexai-food-safety) | 1933★ | Skill | Industry LLM | HACCP/FDA/USDA Skill |

---

## 1. SafeGuard AI

- **Repo**: https://github.com/Sharma-Aditya7/multiagent-food-compliance
- **Stars**: 5 | **License**: MIT
- **Stack**: Python · CrewAI · Gradio · Groq Mixtral-8x7b

### 核心特性 / Core Features
- 多 Agent 协作执行食品合规审计
- 实时风险识别（生物/化学/物理危害）
- 自动生成 HACCP 审计跟踪文档

### 架构 / Architecture
```text
SafeGuard AI
├── Risk Identification Agent    # 风险识别
├── HACCP Compliance Agent       # 合规检查
├── Documentation Agent          # 文档生成
└── Gradio UI                    # 用户交互
```

### 实战价值 / Practical Value
- 减少人工审计误差
- 降低合规成本
- 可作为中型食品企业 HACCP 数字化起点

---

## 2. SFPPy (Safe Food Packaging Python)

- **Repo**: https://github.com/ovitrac/SFPPy
- **Stars**: 4 | **License**: MIT
- **Homepage**: https://ovitrac.github.io/SFPPy/

### 核心特性
- 食品接触材料合规测试 (FCM)
- 回收塑料安全评估
- 化学迁移建模 (M0–M3 tiers)
- **agentic SAG** (Simulation-Augmented Generation)

### 关键能力
- 标准化 Jupyter Notebook 工作流
- AI-Ready 设计 (Symbolic + structured data)
- 自定义 GPT 已训练 SFPPy 知识库

---

## 3. FoodSafe

- **Repo**: https://github.com/AtharvaVavhal/FoodSafe
- **Stars**: 1 | **Stack**: React · FastAPI · YOLOv8 · IndicBERT/MuRIL · Prophet
- **Live**: https://food-safe-lsca.vercel.app

### 核心特性
- YOLOv8 实时摄像头掺假检测
- Hindi/Marathi 区域 NLP (IndicBERT/MuRIL)
- Prophet 时序预测季节性掺假峰值
- FSSAI 违规数据集成
- 个性化毒素暴露评分

### 架构
```text
FoodSafe
├── Frontend (React + Tailwind + Leaflet)
├── Backend (FastAPI + PostgreSQL + Redis + Celery)
├── ML (Claude API + YOLOv8 + IndicBERT + Prophet)
└── Cost: ₹0 (free tier)
```

---

## 4. KAD-FoodHazard (MyMy @ SemEval-2025 Task 9)

- **Repo**: https://github.com/phanben110/KAD-FoodHazard
- **Stars**: 5 | **License**: GPL-3.0
- **Result**: SemEval-2025 Task 9 Subtask 1 & 2 Top-2

### 核心方法 / Methodology
- **KAD** (Knowledge-Augmented Data) 管线
- RAG + PubMed API 领域知识
- LLM 数据增强 (Llama 3.1 / Mixtral)
- 验证过滤保证数据质量
- 微调 PubMedBERT / Gemini Flash
- 集成策略 (Ensemble)

### Pipeline
```text
PubMed API → Document Retrieval
       ↓
   LLM Generation (Llama 3.1 / Mixtral)
       ↓
   Validation Filter (scoring)
       ↓
   Fine-tune (PubMedBERT / Gemini Flash)
       ↓
   Ensemble Prediction
```

---

## 5. SERSFormer

- **Repo**: https://github.com/BioinfoMachineLearning/SERSFormer
- **Stars**: 3 | **License**: MIT
- **Application**: 食品表面增强拉曼光谱 (SERS) 农药残留检测

### 核心特性
- 多任务权重共享 Transformer
- 同时执行 **分类 + 回归**
- 数据集：5 种农药 (thiabendazole / phosmet / coumaphos / carbophenothion / oxamyl)
- 浓度范围 0–10 ppm

### 使用 / Usage
```bash
git clone https://github.com/BioinfoMachineLearning/SERSFormer.git
python SERSFormer_Training.py --attn_head 4 --encoder_layers 6 --save_dir SERSFormer_log
```

---

## 6. afrexai-food-safety

- **Marketplace**: https://level8.bg/tools/skills/afrexai-food-safety
- **Stats**: 1933 stars · 367 forks (skill 生态)
- **Vendor**: AfrexAI — Food & Manufacturing Context Pack

### 监管框架 / Regulatory Framework
- **FDA**: FSMA Preventive Controls (21 CFR 117), Seafood HACCP (21 CFR 123), Juice HACCP (21 CFR 120), Acidified Foods (21 CFR 114), Produce Safety Rule
- **USDA-FSIS**: 9 CFR Parts 417, 430 — HACCP + Listeria Rule + Salmonella Performance Standards

### Agent 角色
- Hazard Analysis (Biological/Chemical/Physical)
- CCPs + Critical Limits
- Monitoring + Corrective Actions
- Verification + Record-keeping

### 安装 / Install
```bash
npx clawhub@latest install afrexai-food-safety
```

---

## 🔗 相关商业 SaaS / Commercial SaaS References

- **IONI** (https://ioni.ai/) — AI agents for Food & Beverage Compliance
  - 自动构建 HACCP 计划
  - 实时 CCP 监控
  - 审计报告一键导出
  - 支持 SQF / BRCGS / FSMA / CFIA / FSSC 22000
