# 🎯 QUICK REFERENCE - Autoencoder & Data Pipeline

## 📊 **Data Flow in 60 Seconds**

```
100K CSV → Clean (99K) → Engineer → Encode → Scale → Balance → Train (137K) → Test (20K)
                                                                      ↓
                                                             Autoencoder (9,942 params)
                                                                      ↓
                                                            Risk Score (0-100) → Decision
```

---

## 🧠 **Autoencoder Explained Simply**

**What it does:**
"Learns to copy normal loan applications. Can't copy unusual ones well."

**How it works:**
1. Show it 68,827 **good loans** (no defaults)
2. It learns: "This is what normal looks like"
3. When it sees a **risky loan**, it can't reconstruct it well
4. High error = High risk = Reject

**Architecture:**
```
INPUT (30 numbers) → COMPRESS (to 8 numbers) → DECOMPRESS (back to 30) → OUTPUT

If OUTPUT ≈ INPUT → Normal (low error)
If OUTPUT ≠ INPUT → Unusual (high error) → DEFAULT RISK
```

---

## 📈 **Key Numbers**

| Metric | Value | Meaning |
|--------|-------|---------|
| **Input samples** | 100,000 | Original loan applications |
| **After cleaning** | 99,326 | Removed bad data |
| **Features** | 30 | Age, income, loan amount, etc. |
| **Training samples** | 137,654 | Balanced with SMOTE |
| **Model parameters** | 9,942 | Autoencoder size |
| **Training time** | 5 minutes | On CPU |
| **Accuracy** | 94.14% | Test set performance |
| **ROC-AUC** | 0.6748 | Better than random (0.5) |

---

## 🔧 **Preprocessing Steps**

1. **Clean:** Remove duplicates, outliers (674 removed)
2. **Engineer:** Create 7 new features (debt-to-income, etc.)
3. **Encode:** Convert text to numbers (19 features)
4. **Impute:** Fill missing values (median strategy)
5. **Scale:** Normalize to similar ranges (RobustScaler)
6. **Balance:** SMOTE-Tomek for training only (50/50 split)
7. **Split:** 70% train / 10% val / 20% test

---

## 🎯 **Risk Score Examples**

**Score 5/100 (LOW)** ✅
- Good income, low debt
- Strong credit history
- **Decision:** APPROVE

**Score 40/100 (MEDIUM)** ⚠️
- Moderate debt-to-income
- Some late payments
- **Decision:** APPROVE WITH CONDITIONS

**Score 85/100 (HIGH)** ❌
- High debt, low income
- Multiple delinquencies
- **Decision:** REJECT

---

## 🚀 **Next: LLM Agent**

**Current System:**
```
Customer → Autoencoder → Risk Score → Decision
```

**With LLM Agent:**
```
Customer → Autoencoder → Risk Score
                           ↓
                    Natural Language Query
                           ↓
           LLM Agent (Llama 3.2) + RAG
                           ↓
              "Rejected because debt-to-income
               ratio (11.2x) exceeds policy limit
               (8x). Try with co-signer..."
```

---

## 📁 **File Structure**

```
credit-risk-agent/
├── data/
│   ├── raw/Bank_data.csv              ← Your 100K loans
│   └── processed/*.npy                 ← Ready for training
│
├── models/
│   ├── autoencoder/                    ← Trained model (40KB)
│   └── preprocessor/                   ← Fitted transformers
│
├── src/
│   ├── data_preprocessing.py           ← Step 1: Clean data
│   ├── autoencoder_training.py         ← Step 2: Train model
│   ├── risk_scoring.py                 ← Step 3: Make decisions
│   └── agent/                          ← Step 4: LLM (next)
│
└── COMPLETE_TECHNICAL_OVERVIEW.md     ← Full details (30+ pages)
```

---

## 💡 **Key Insights**

1. **Why autoencoder?**
   - Works with imbalanced data (99:1)
   - No need for many default examples
   - Detects "unusual" patterns

2. **Why train on normal only?**
   - Defaults are too rare (1%)
   - Model learns "what's normal"
   - Defaults appear as anomalies

3. **Why conservative scoring?**
   - Banking prefers safe rejections
   - Better to lose 5% good loans
   - Than approve 1% bad loans

4. **Why add LLM?**
   - Autoencoder gives number (85/100)
   - LLM explains "why" in plain English
   - RAG adds policy compliance

---

## 🎓 **For Your Understanding**

Think of the system as:
- **Autoencoder** = Security guard (spots suspicious behavior)
- **Preprocessor** = Data cleaner (prepares info)
- **Risk Scorer** = Decision maker (approve/reject)
- **LLM Agent** = Friendly explainer (tells you why)

The autoencoder doesn't understand "loans" or "defaults" - it just learns patterns in 30 numbers and flags unusual patterns!

---

**Read COMPLETE_TECHNICAL_OVERVIEW.md for full details!**
