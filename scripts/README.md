# Scripts Directory - Organized

## 📁 **Script Categories**

### 🔧 **Model Management**
- **`download_qwen3_4b.py`** - Download and convert Qwen3-4B model from HuggingFace
- **`create_test_model.py`** - Create test models for validation
- **`export.py`** - Convert HuggingFace models to qwen3.c format
- **`model.py`** - Model utilities and helper functions

### 📊 **Benchmarking & Performance**
- **`final_performance_report.py`** - ⭐ **MAIN**: Generate comprehensive performance analysis
- **`comprehensive_benchmark.py`** - Complete Metal vs CPU benchmark suite
- **`actual_response_test.py`** - Test real LLM responses with timing measurements
- **`setup_prompt_test.py`** - Set up standardized prompt-test benchmarking

### 🧪 **Testing & Validation**
- **`check_api_accuracy.py`** - Validate OpenAI API compatibility

## 🚀 **Quick Usage**

### **Get Performance Results:**
```bash
python3 scripts/final_performance_report.py
```

### **Test Real Responses:**
```bash
python3 scripts/actual_response_test.py
```

### **Full Benchmark Suite:**
```bash
python3 scripts/comprehensive_benchmark.py
```

### **Download Model (if needed):**
```bash
python3 scripts/download_qwen3_4b.py --output-dir models
```

## ✅ **Cleanup Status**
- Removed duplicate/unused Python files from root directory
- Organized scripts by purpose and functionality
- Updated all path references in documentation
- Maintained only essential, working scripts

**Total Scripts: 9 (down from 25+ scattered files)**