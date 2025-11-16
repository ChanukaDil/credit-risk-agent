# 🏦 Credit Risk Assessment System

AI-powered credit risk assessment system combining **supervised machine learning (XGBoost)**, **advanced feature engineering**, **SHAP explainability**, and **LLM agent** with RAG for intelligent loan default prediction.

## 🎯 Key Features

### Production ML System

- **🎯 XGBoost Classifier**: 72.5% recall (catches 145/200 defaults) - **2,800% improvement** over autoencoder
- **📊 Advanced Feature Engineering**: 30 base features → 80+ engineered features
- **📈 SHAP Explainability**: Understand why model makes predictions
- **🚀 Production REST API**: FastAPI with real-time & batch predictions
- **📉 Class Imbalance Handling**: SMOTE + optimized threshold tuning

### AI Agent System

- **🤖 RAG System**: Policy retrieval from knowledge base using FAISS
- **💬 LLM Agent**: Natural language interface with Llama 3.2 3B
- **🔍 Multi-stage Validation**: ML Model → RAG → LLM → Decision pipeline
- **💻 Interactive CLI**: Beautiful command-line interface

### Performance

- ✅ **Recall**: 72.5% (vs 2.5% autoencoder)
- ✅ **Precision**: 45.7%
- ✅ **ROC AUC**: 0.965
- ✅ **PR AUC**: 0.722
- ✅ **False Positive Rate**: 0.87% (only 172 false alarms out of 19,666)

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 8GB RAM minimum (16GB recommended for LLM agent)
- 5GB free disk space

### 📚 **Read Documentation First!**

**🌟 START HERE:** [PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md) - Complete production guide

- Complete file list (what to keep vs remove)
- Final folder structure
- Step-by-step execution from start to finish

**📊 Visual Guide:** [EXECUTION_FLOWCHART.md](EXECUTION_FLOWCHART.md) - See the flow visually

**⚡ Quick Reference:** [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) - Common commands

**📖 All Docs:** [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Complete documentation index

---

### Installation & Setup

1. **Clone the repository:**

```bash
git clone https://github.com/ChanukaDil/credit-risk-agent.git
cd credit-risk-agent
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
pip install -r requirements_api.txt
pip install shap  # For explainability
```

3. **Clean up deprecated files (autoencoder):**

```bash
# Windows
cleanup_deprecated_files.bat

# Linux/Mac
bash cleanup_deprecated_files.sh
```

---

### Production ML Pipeline (5 Steps)

#### **Step 1: Data Preprocessing**

```bash
python src/run_preprocessing.py
```

Output: `data/processed/*.npy` (137,654 train, 19,866 test samples)

#### **Step 2: Feature Engineering (Optional but Recommended)**

```bash
python src/feature_engineering.py
```

Output: 80+ engineered features, `models/improved/feature_engineer.pkl`

#### **Step 3: Model Training (MAIN)**

```bash
python src/final_improved_models.py
```

Output: `models/improved/rank1_xgboost.pkl` (72.5% recall)

#### **Step 4: Generate Monitoring Dashboard**

```bash
python src/model_monitoring.py
```

Output: SHAP plots in `results/monitoring/`

#### **Step 5: Deploy Production API**

```bash
uvicorn api.deployment_api:app --host 0.0.0.0 --port 8000 --reload
```

API running at: http://localhost:8000

---

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "NET_RENTAL": 15000,
    "NO_OF_RENTAL": 60,
    "PAID_RENTALS": 12,
    "CB_ARREARS_AGE": 0,
    "INCOME": 75000,
    "EXPENSE": 40000
  }'
```

---

### Optional: LLM Agent System

```bash
# Download LLM (one-time setup)
python scripts/download_models.py

# Initialize knowledge base
python scripts/setup_vector_db.py

# Run agent CLI
python scripts/run_agent_cli.py
```

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  PRODUCTION ML PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Raw Data → Preprocessing → Feature Engineering → SMOTE     │
│                                 ↓                            │
│           XGBoost Training ← Balanced Data                   │
│                                 ↓                            │
│           Trained Model (72.5% recall)                       │
│                    ↓           ↓                             │
│         SHAP Explainability  FastAPI                         │
│                                 ↓                            │
│                    Real-time Predictions                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  OPTIONAL: LLM AGENT                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Customer Query → ML Model → RAG (Policy) → LLM → Explanation│
└─────────────────────────────────────────────────────────────┘
```

**See full architecture:** [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)

## 🛠️ Technology Stack

### Production ML System

- **ML Framework**: XGBoost, scikit-learn, LightGBM
- **Class Balancing**: imbalanced-learn (SMOTE)
- **Explainability**: SHAP
- **API**: FastAPI, Uvicorn
- **Data Processing**: Pandas, NumPy

### AI Agent (Optional)

- **LLM**: Llama 3.2 3B (HuggingFace)
- **RAG**: LangChain, FAISS, sentence-transformers
- **Vector DB**: FAISS
- **Embeddings**: all-MiniLM-L6-v2

## 📁 Project Structure

```
credit-risk-agent/
├── 📚 DOCUMENTATION
│   ├── PRODUCTION_DEPLOYMENT_GUIDE.md    ⭐ Complete guide (START HERE)
│   ├── EXECUTION_FLOWCHART.md            Visual workflow
│   ├── QUICK_START_GUIDE.md              Quick commands
│   ├── ARCHITECTURE_OVERVIEW.md          Full architecture
│   ├── MODEL_IMPROVEMENT_SUMMARY.md      Model comparison
│   └── DOCUMENTATION_INDEX.md            All docs index
│
├── 💻 SOURCE CODE (Production)
│   ├── src/
│   │   ├── run_preprocessing.py          ✅ Step 1: Preprocessing
│   │   ├── feature_engineering.py        ✅ Step 2: Feature eng
│   │   ├── final_improved_models.py      ✅ Step 3: Training ⭐
│   │   ├── model_monitoring.py           ✅ Step 4: Monitoring
│   │   └── agent/                        🤖 LLM agent (optional)
│   └── api/
│       └── deployment_api.py             ✅ Step 5: API ⭐
│
├── 🧠 MODELS
│   └── models/improved/
│       ├── rank1_xgboost.pkl             Best model (72.5% recall)
│       ├── rank2_random_forest.pkl       Backup model
│       └── model_results.json            Performance metrics
│
├── 💾 DATA
│   ├── data/raw/Bank_data.csv            167,452 loan applications
│   └── data/processed/*.npy              Preprocessed data
│
├── 🧹 CLEANUP SCRIPTS
│   ├── cleanup_deprecated_files.bat      Windows cleanup
│   └── cleanup_deprecated_files.sh       Linux/Mac cleanup
│
└── 📊 RESULTS
    ├── results/monitoring/               SHAP plots
    └── results/logs/                     API logs
```

**See complete structure:** [PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md#final-folder-structure)

## 📖 Documentation

| Document                                                                | Purpose                                          | Read Time |
| ----------------------------------------------------------------------- | ------------------------------------------------ | --------- |
| **[PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md)** ⭐ | Complete guide (files to keep/remove, execution) | 20 min    |
| **[EXECUTION_FLOWCHART.md](EXECUTION_FLOWCHART.md)**                    | Visual workflow & flowchart                      | 10 min    |
| **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)**                        | Quick commands & troubleshooting                 | 5 min     |
| **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)**                    | Complete documentation index                     | 5 min     |
| **[ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)**                | Full system architecture                         | 30 min    |
| **[MODEL_IMPROVEMENT_SUMMARY.md](MODEL_IMPROVEMENT_SUMMARY.md)**        | Why we replaced autoencoder                      | 15 min    |

## 🧪 Testing

```bash
# Quick test
python tests/quick_test.py

# Complete system test
python tests/test_complete_system.py

# Test model samples
python tests/test_model_samples.py
```

## � Performance Metrics

### Production Model (XGBoost)

| Metric                  | Value | Meaning                          |
| ----------------------- | ----- | -------------------------------- |
| **Recall**              | 72.5% | Catches 145/200 defaults ✅      |
| **Precision**           | 45.7% | 145 correct out of 317 flagged   |
| **F1-Score**            | 0.561 | Good balance                     |
| **ROC AUC**             | 0.965 | Excellent discrimination ⭐      |
| **PR AUC**              | 0.722 | Excellent for imbalanced data ⭐ |
| **False Positive Rate** | 0.87% | Only 172 false alarms/19,666     |

### Business Impact

- ✅ **Defaults caught:** 145/200 (72.5%)
- ⚠️ **Defaults missed:** 55/200 (27.5%)
- ✅ **False alarms:** 172/19,666 (0.87%)
- ✅ **Correct approvals:** 19,494/19,666 (99.13%)

### Comparison: Old vs New

| System                   | Recall | Improvement |
| ------------------------ | ------ | ----------- |
| **Autoencoder (OLD)** ❌ | 2.5%   | Baseline    |
| **XGBoost (NEW)** ✅     | 72.5%  | **+2,800%** |

**See detailed comparison:** [MODEL_IMPROVEMENT_SUMMARY.md](MODEL_IMPROVEMENT_SUMMARY.md)

## 🚀 Deployment

### Local Development

```bash
uvicorn api.deployment_api:app --reload --port 8000
```

### Production

```bash
uvicorn api.deployment_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Coming Soon)

```bash
docker-compose up -d
```

## �️ Deprecated Components

**⚠️ These files should be removed (use cleanup script):**

- ❌ `src/autoencoder_training.py` (2.5% recall - failed)
- ❌ `src/improved_models.py` (intermediate attempt)
- ❌ `scripts/evaluate_autoencoder.py` (obsolete)
- ❌ `models/autoencoder/` (entire directory)

**Run cleanup:**

```bash
cleanup_deprecated_files.bat  # Windows
bash cleanup_deprecated_files.sh  # Linux/Mac
```

## 🤝 Contributing

Contributions welcome! Please:

1. Read [PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md) first
2. Follow existing code structure
3. Add tests for new features
4. Update documentation

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Dataset:** Bank loan application data (167,452 records)
- **XGBoost:** Gradient boosting framework
- **SHAP:** Model explainability
- **FastAPI:** Modern web framework
- **LangChain:** LLM agent framework
- **HuggingFace:** Llama 3.2 3B model

## 📞 Support

- **Documentation:** Start with [PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md)
- **Quick Help:** [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)
- **All Docs:** [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)
- **Issues:** GitHub Issues
- **Questions:** GitHub Discussions

---

**⭐ Star this repo if you find it useful!**

**Made with ❤️ for better credit risk assessment**
