# Project Directory Structure Reference

## 📂 Current Project Layout

```
google_agents_intensive_capstone_project/
│
├── README.md                     ← Project documentation (shows on GitHub)
├── requirements.txt              ← Python dependencies
├── .gitignore                    ← Files Git should ignore
├── .env                          ← Secret API keys (NEVER commit!)
│
├── data/                         ← Raw data (gitignored)
│   └── creditcard.csv            ← 284K fraud transactions
│
├── models/                       ← Trained models (gitignored)
│   └── xgboost_fraud_model.pkl   ← Your 0.9886 AUC model
│
├── notebooks/                    ← Jupyter notebooks for exploration
│   └── demo_v1_clean.ipynb       ← Original research notebook
│
├── src/                          ← Source code (main modules)
│   ├── __init__.py               ← Makes src a Python package
│   ├── model.py                  ← Model training & evaluation
│   ├── fraud_agent.py            ← LLM agent + fraud scoring
│   └── utils.py                  ← Helper functions
│
├── tests/                        ← Unit tests
│   ├── __init__.py
│   ├── conftest.py               ← Test fixtures & mocks
│   ├── test_model.py             ← Model tests
│   ├── test_fraud_agent.py       ← Agent tests
│   └── test_utils.py             ← Utils tests
│
└── streamlit_app/                ← Interactive web demo
    └── app.py                    ← Streamlit fraud detector app
```

---

## 🔧 Commands to Generate This Anytime

### **Windows PowerShell:**
```powershell
# Show full tree
tree /F /A

# Show only folders (no files)
tree

# Save to file
tree /F /A > structure.txt

# Show specific folder
tree /F src
```

### **Navigate with Commands:**
```powershell
# List files in current directory
ls
# or
dir

# List files in src
ls src

# List everything recursively
ls -R
```

---

## 📝 Directory Path Notation Explained

### **Slash `/` Notation**
| Path | Meaning |
|------|---------|
| `src/` | A folder named "src" |
| `src/model.py` | File "model.py" inside "src" folder |
| `./src/` | "src" in current directory (`.` = here) |
| `../src/` | "src" in parent directory (`..` = up one level) |
| `/src/` | "src" at root level (absolute path from C:\ drive) |

### **Special Symbols**
| Symbol | Meaning | Example |
|--------|---------|---------|
| `/` | Directory separator | `src/model.py` |
| `.` | Current directory | `./src/` = src in current folder |
| `..` | Parent directory | `../data/` = data one level up |
| `~` | Home directory | `~/Documents/` (Mac/Linux) |
| `*` | Wildcard (any) | `*.py` = all Python files |

---

## 📁 Common Folder Name Abbreviations

| Folder | Full Name | Purpose |
|--------|-----------|---------|
| `src/` | source | Your main source code |
| `docs/` | documents | Documentation |
| `tests/` | tests | Unit/integration tests |
| `bin/` | binary | Executable programs |
| `lib/` | library | External libraries |
| `tmp/` | temporary | Temporary files |
| `env/` | environment | Virtual environment |
| `dist/` | distribution | Built/packaged code |
| `data/` | data | Data files |
| `models/` | models | Saved ML models |

---

## 🎯 Quick Navigation Tips

```powershell
# Where am I?
pwd
# or
cd

# Go to project root
cd C:\Users\chris\google_agents_intensive_capstone_project

# Go to src folder
cd src

# Go up one level
cd ..

# Go to specific folder
cd tests

# List what's here
ls
```

---

## 🚀 Useful Commands for Your Project

```powershell
# Show project structure
tree /F /A

# Run tests
pytest tests/ -v

# Test a module
python src/fraud_agent.py

# Launch Streamlit
streamlit run streamlit_app/app.py

# Check Git status
git status

# View file contents
cat README.md
```

---

## 📊 Your Project Stats

- **Total Test Files**: 4 (conftest.py + 3 test modules)
- **Total Tests**: 38 tests
- **Test Pass Rate**: 92% (35/38)
- **Source Modules**: 3 (model.py, fraud_agent.py, utils.py)
- **Model Performance**: 0.9886 AUC
- **Lines of Code**: ~1,000+ lines of production code

---

*Generated: November 2025*  
*Author: Cristian Perera*