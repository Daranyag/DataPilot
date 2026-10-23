# 🚀 DataPilot: Autonomous Data Analyst & AutoML Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-purple)](https://github.com/langchain-ai/langgraph)
[![Gemini](https://img.shields.io/badge/Powered%20by-Google%20Gemini-orange)](https://ai.google.dev/)

**DataPilot** is an agentic AI system for data science and automated machine learning. It pairs a **conversational Natural Language Data Analyst** with a robust **AutoML engine**, enabling users to inspect data, execute pandas queries safely, and automatically identify the highest-performing machine learning models for their datasets.

---

## 🌟 Key Features

### 1. 💬 Conversational Data Analyst Agent
- **Natural Language to Code**: Ask questions about your CSV in plain English (e.g., *"What is the distribution of sales by region?"* or *"Show top 5 customers by revenue"*).
- **LangGraph Self-Correction Loop**: The agent generates Python/pandas code, executes it in a secure environment, and if errors occur, autonomously reads the traceback and refines its code.
- **Visual Plots & Tables**: Directly outputs styled tables and dynamic data visualizations.

### 2. 🤖 End-to-End AutoML Engine
- **Automated Task Detection**: Intelligently identifies whether your dataset problem is **Regression**, **Classification**, or **Clustering** based on target column characteristics.
- **Smart Data Preparation**: Handles missing values (imputation), categorical encodings, and numerical feature scaling (Standard / MinMax).
- **5-Fold Cross-Validation**: Validates algorithms rigorously across stratified/standard folds.
- **Mathematically Sound Ranking**: Compares models using standardized metrics (RMSE, $R^2$, F1-Macro, Accuracy, Silhouette Score) and breaks ties by prioritizing faster, more interpretable models.

### 3. 🛡️ Secure AST-Based Execution Sandbox
- Validates all generated Python code against an AST (Abstract Syntax Tree) policy before execution.
- Blocks hazardous standard libraries (`os`, `sys`, `subprocess`, `shutil`) and dangerous built-ins (`eval`, `exec`, `open`, `__builtins__`).

### 4. 🎨 Interactive Streamlit Interface
- Modern, intuitive web dashboard for uploading datasets, inspecting data profiles, running queries, and viewing AutoML leaderboards with a single click.

---

## 📂 Project Architecture

```text
DataPilot/
├── agent/            # LangGraph workflow, prompt engineering, and dataset profiler
├── app/              # Streamlit frontend application (main.py)
├── automl/           # Task detection, model training pipelines, caching, and metric evaluations
├── evaluation/       # Benchmark datasets and QA / AutoML evaluation scripts
├── sandbox/          # AST security validator and safe code execution engine
├── tests/            # Test suite for unit and integration testing
├── requirements.txt  # Project dependencies
└── README.md         # Project documentation
```

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Daranyag/DataPilot.git
cd DataPilot
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```
*(Get your free API key at [Google AI Studio](https://aistudio.google.com/)).*

### 4. Run the Streamlit App
```bash
streamlit run app/main.py
```
Open your browser at `http://localhost:8501`.

---

## 🚀 How to Deploy on Streamlit Community Cloud

1. Fork or push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"** and select:
   - **Repository**: `Daranyag/DataPilot`
   - **Branch**: `main`
   - **Main file path**: `app/main.py`
4. Under **Advanced Settings** -> **Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```
5. Click **Deploy**!

---

## 🧪 Testing

Run the test suite using `pytest`:
```bash
pytest tests/
```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
