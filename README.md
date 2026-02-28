# Assignment 2: Neural Sequence Models

This assignment extends your Assignment 1 baseline work to neural sequence models. You will implement two mandatory neural architectures and demonstrate whether added complexity yields meaningful performance gains on your specific task.

## 1. Continuity Requirement
**CRITICAL:** You must use the same dataset and same evaluation metrics from Assignment 1. This ensures direct comparison and demonstrates whether neural models justify their computational costs.

- Keep the same train/validation/test splits
- Report the same primary metric (e.g., F1-Macro, entity-level F1, BLEU, ROUGE)
- Compare directly against your Assignment 1 baseline

## 2. Task-Specific Implementations
Your implementation depends on your task type:

- **Text Classification:** Sequence encoder (LSTM/Transformer) + pooling + classification head
- **Sequence Labeling (NER, POS):** Token-level predictions with sequence-to-sequence architecture
- **Generation (Summarization, Translation):** Encoder-decoder with attention

## 3. Required Models — Both Are Mandatory for Everyone
**⚠ MANDATORY FOR ALL STUDENTS, REGARDLESS OF DATASET SIZE:**

- One recurrent model: **LSTM or GRU**, trained from scratch (bidirectional recommended)
- One **Transformer** model: Either trained from scratch or a fine-tuned pre-trained model (e.g., BERT, RoBERTa, DistilBERT)

Both models must be compared against your Assignment 1 baseline. There are no exceptions based on dataset size.

### Choosing Your Transformer Variant
- **Large datasets (>10K examples):** Transformer from scratch is feasible and recommended
- **Small datasets (<10K examples):** Use a fine-tuned pre-trained model (BERT, RoBERTa, DistilBERT) as your Transformer — you must still implement an LSTM from scratch alongside it

### Handling Small Datasets for the LSTM
If your dataset is small, train the LSTM from scratch regardless — this comparison is precisely what makes the experiment valuable. For the Transformer component, fine-tuning a pre-trained model is the recommended approach. You may additionally use:
- **Transfer Learning:** Fine-tune pre-trained models for the Transformer requirement (recommended for small datasets)
- **Data Augmentation:** Back-translation, paraphrasing, or synonym replacement to supplement training data

## 4. Required Experiments
You must conduct the following systematic investigations:

### Experiment 1: Architecture Comparison
- Compare all three models (Assignment 1 baseline, LSTM/GRU, Transformer/BERT) using your task's metric
- Report training time, GPU memory, inference speed, and parameter count

### Experiment 2: Learning Curve Analysis
- Train models on 25%, 50%, 75%, and 100% of training data
- Plot performance vs. training size
- Analyze: At what size do neural models surpass baselines? Does the Transformer benefit more from additional data than the LSTM?

### Experiment 3: Ablation Studies
Conduct at least two ablations on your best-performing model. Choose appropriate ablations for your model type:
- **LSTM:** Unidirectional vs. Bidirectional, number of layers, hidden dimensions
- **Transformer (from scratch):** Attention heads, layers, pooling strategies
- **Pre-trained models:** Frozen vs. unfrozen layers, learning rates, fine-tuning epochs, different model variants (BERT vs. RoBERTa vs. DistilBERT)

### Experiment 4: Error Analysis
- Find 5+ examples where neural models fixed baseline errors
- Find 5+ examples where neural models introduced new errors
- Visualize attention weights (if using Transformers)

### Experiment 5: Computational Cost Analysis
- Compare training time and inference speed across all three models
- Discuss practical deployment considerations

## 5. Training Requirements
Ensure your models are properly trained:

- **Early Stopping:** Monitor validation metric with patience (5-7 epochs)
- **Gradient Clipping:** Max norm of 1.0 for RNNs/LSTMs
- **Learning Rate Scheduling:** Use warmup for Transformers
- **Fixed Random Seeds:** Set `random_state=42` everywhere for reproducibility

## 6. Deliverables
Submit via MiAulario:

- **Code Repository:** Well-organized with structure:
  - `models/` - PyTorch model implementations (must include both LSTM and Transformer)
  - `notebooks/` - Training and evaluation notebooks
  - `results/` - Training curves, confusion matrices, attention visualizations
  - `README.md` - Setup and execution instructions
  - `requirements.txt` - All dependencies with versions
- **Report (PDF):** Maximum 4 pages (excluding visualizations):
  - **Introduction (0.5 pages):** Recap Assignment 1 baseline
  - **Methodology (1 page):** Both architectures, training setup, data handling
  - **Results (1.5 pages):** All 5 required experiments
  - **Analysis (1 page):** Error analysis, practical recommendations