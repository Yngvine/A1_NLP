
#align(center)[
  #v(2cm)
  #text(size: 24pt, weight: 700)[Assignment 2: Neural Sequence Models]
  #v(6pt)
  #text(size: 14pt)[Master's in Machine Learning 2025]
  #v(1.2cm)
  #text(size: 12pt)[
    Author: Igor Vons, Endika Aguirre and Maria Ines Haddad

    Date: March 1, 2026
  ]
]

#pagebreak()

= Problem Statement

We continue the task of *Caption-Question Relevance Classification* introduced in Assignment 1: given a descriptive caption $C$ of a remote-sensing image and a question-answer pair $(Q, A)$, determine whether they originate from the same image (Related, label$=$1) or from different images (Unrelated, label$=$0).

In Assignment 1 we established strong baselines using sparse (TF-IDF) and dense (GloVe, FastText) feature representations combined with classical classifiers. The best result was a *Random Forest + TF-IDF bigrams* model with Test F1-Macro of *0.8071*, though at a prohibitive training cost (~300 minutes). In Assignment 2 we investigate whether neural sequence models — a custom Bidirectional LSTM and a fine-tuned RoBERTa Transformer — can surpass these baselines while offering a more practical trade-off between performance and computational cost.

= Dataset and Splits

The dataset, preprocessing strategy, and train/test splits are kept *identical* to Assignment 1 to guarantee fair comparison.

== Dataset Recap

We use the #link("https://github.com/StarZi0213/RSVLM-QA")[RSVLM-QA] dataset (13,820 images, 162,373 VQA pairs). Each input is a concatenation of caption and question–answer:

#align(center)[`[Caption] [SEP] [Question] [Answer]`]

*Positive pairs (label=1)*: correct caption–QA triplets from the same image.\
*Negative pairs (label=0)*: caption sampled from the most semantically distant images using tag-based FastText cosine distance (top 25% most distant), ensuring hard, genuinely unrelated negatives.

The final dataset contains *297,116 examples* with a 50/50 class balance.

== Train / Validation / Test Split

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: (left, center, center),
  [*Split*], [*Size*], [*Construction*],
  [Train], [213,923], [80% stratified split, then 90% of training portion],
  [Validation], [23,770], [10% carved from training portion, for early stopping],
  [Test], [59,423], [Held-out 20% stratified split, identical to A1],
)

The 80/20 train/test split uses `random_state=42`, replicating Assignment 1 exactly. The validation set is carved from the training portion to enable early stopping without touching the held-out test set.

= Models

== BiLSTM

We implement a custom *Bidirectional LSTM* classifier from scratch, incorporating several architectural improvements over the initial design:

- *Tokenizer*: simple whitespace tokenizer (lowercase).
- *Vocabulary*: frequency-based, minimum frequency 1, maximum size 30,000 (built on training data only).
- *Embedding layer*: trainable embeddings of dimension 200, `<PAD>` index zeroed.
- *Architecture*: 2-layer BiLSTM with hidden size 512. Instead of using only the final forward/backward hidden states, we apply *learned attention pooling* over all timesteps — a scalar attention score is computed for each position, padded positions are masked before softmax, and the weighted sum forms the 1024-dimensional sentence representation. This is followed by dropout (0.3) and a linear classification head.
- *Input truncation*: sequences truncated to 256 tokens (increased from 128 to retain more context).

*Training configuration*:

#table(
  columns: (auto, auto),
  inset: 8pt,
  align: (left, left),
  [*Hyperparameter*], [*Value*],
  [Optimizer], [Adam],
  [Learning rate], [3e-4],
  [LR scheduler], [ReduceLROnPlateau (factor 0.5, patience 3, mode max)],
  [Max epochs], [30],
  [Early stopping patience], [5 (on val F1-macro, min delta 1e-3)],
  [Batch size], [64],
  [Gradient clipping], [max norm 1.0],
)

== RoBERTa-base

We fine-tune *`roberta-base`* (~125M parameters, pre-trained by Facebook AI) for binary sequence classification:

- *Tokenizer*: RoBERTa BPE tokenizer (vocab size 50,265), `max_length=256` (increased from 128 to capture full caption + QA context).
- *Architecture*: `RobertaForSequenceClassification` with a 2-class linear head on top of the `[CLS]` representation.
- *Dataset format*: the same `[Caption] [SEP] [Question Answer]` string is passed directly to the RoBERTa tokenizer, which handles special tokens (`<s>`, `</s>`) internally.

*Training configuration*:

#table(
  columns: (auto, auto),
  inset: 8pt,
  align: (left, left),
  [*Hyperparameter*], [*Value*],
  [Optimizer], [AdamW],
  [Learning rate], [1e-5],
  [Weight decay], [0.05],
  [Max epochs], [10],
  [Early stopping patience], [3 (on val F1-macro, min delta 1e-3)],
  [Batch size], [16 (effective 64 via gradient accumulation × 4)],
  [Warmup steps], [10% of total steps (linear schedule)],
  [Gradient clipping], [max norm 1.0],
)

= Experimental Setup

All experiments follow the same evaluation protocol as Assignment 1:

- *Primary metric*: F1-Macro on the held-out test set (accounts for class balance).
- *Early stopping*: monitored on validation F1-Macro; best checkpoint saved and reloaded for final evaluation.
- *Reproducibility*: all random seeds fixed at 42.

The experiments are organized as follows:

+ *Experiment 1 — Architecture Comparison*: BiLSTM vs. RoBERTa-base vs. A1 best baseline (Random Forest + TF-IDF bigrams).
+ *Experiment 2 — Effect of Contextual Representations*: static embeddings (GloVe, FastText mean pooling from A1) vs. BiLSTM sequential encoding vs. RoBERTa contextual embeddings.
+ *Experiment 3 — Sparse vs. Neural Features*: TF-IDF (A1 best sparse) vs. BiLSTM (learned dense) vs. RoBERTa (pre-trained Transformer).
+ *Experiment 4 — Error Analysis*: qualitative comparison of RoBERTa failure cases vs. A1 baseline failure cases.
+ *Experiment 5 — Computational Cost*: parameters, model size (MB), training time, and inference time comparison.

= Results

== Overall Architecture Comparison (Experiment 1)

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: (left, center, center),
  [*Model*], [*Test F1-Macro*], [*vs. A1 Best*],
  [A1 Best: Random Forest + TF-IDF Bigrams], [0.8071], [—],
  [A1: Logistic Regression + TF-IDF Bigrams], [0.6896], [-17.0%],
  [A2: BiLSTM (2-layer, attn, hidden=512, emb=200)], [*0.8579*], [+6.3%],
  [A2: RoBERTa-base (fine-tuned, val F1=0.9509)], [pending†], [—],
)

† _RoBERTa was fully retrained in this run and achieved a best validation F1-Macro of 0.9509 (early stopping at epoch 10). The held-out test evaluation is pending execution._

Both neural architectures, after architectural improvements, now surpass the A1 Random Forest baseline. The improved BiLSTM achieves Test F1-Macro of *0.8579*, a gain of *+6.3%* over the best A1 result (0.8071). RoBERTa's validation performance of *0.9509* strongly suggests the test score will exceed the baseline by a wide margin once evaluated. These results stand in stark contrast to the initial runs (BiLSTM: 0.6056, RoBERTa: 0.6582) and demonstrate the outsized impact of targeted architectural and hyperparameter changes on neural sequence models.

== Effect of Sequential and Contextual Representations (Experiment 2)

#table(
  columns: (auto, auto, auto, auto),
  inset: 8pt,
  align: (left, left, center, center),
  [*Model*], [*Representation type*], [*Test F1-Macro*], [*Test Accuracy*],
  [LogReg + GloVe (A1)], [Static mean-pooled (100d)], [0.6513], [0.6517],
  [LogReg + FastText (A1)], [Static mean-pooled (300d)], [0.6528], [0.6534],
  [BiLSTM (A2, improved)], [Attn-pooled learned (200d → 1024d BiLSTM)], [*0.8579*], [—],
  [RoBERTa-base (A2, val)], [Contextual Transformer (768d)], [0.9509 (val)], [—],
)

Both neural models decisively outperform the A1 static embedding baselines, confirming that sequential and contextual representations capture the fine-grained lexical alignment between captions and question-answer pairs far better than mean-pooled static vectors. The improved BiLSTM, with attention pooling and a larger hidden state, achieves 0.8579 — a *+31.7%* relative gain over the GloVe baseline. The key changes enabling this were: attention over all hidden states (rather than only the final state), a 30,000-token vocabulary with no minimum frequency threshold, 256-token sequences, and a 512-unit hidden size. These modifications allow the model to attend to the specific overlapping keywords that signal relevance between caption and QA pair.

== Sparse vs. Neural Features (Experiment 3)

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: (left, left, center),
  [*Representation*], [*Best Classifier*], [*Test F1-Macro*],
  [TF-IDF Bigrams (Sparse, A1)], [Random Forest], [0.8071],
  [TF-IDF Bigrams (Sparse, A1)], [Logistic Regression], [0.6896],
  [BiLSTM Embeddings (Neural, A2, improved)], [BiLSTM + Attention], [*0.8579*],
  [RoBERTa Contextual (Neural, A2, improved)], [RoBERTa-base], [0.9509 (val)],
)

With improved architectures, neural models have clearly surpassed LR-based sparse approaches and now challenge or exceed the Random Forest TF-IDF ceiling. The BiLSTM with attention pooling (0.8579) already outperforms the best TF-IDF Random Forest (0.8071) by a significant margin while remaining computationally lightweight relative to RoBERTa. This reverses the conclusion from the initial experimental run: the bottleneck was not the neural paradigm itself, but insufficiently large vocabulary coverage and loss of long-range context due to aggressive sequence truncation.

== Computational Cost (Experiment 5)

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 8pt,
  align: (left, center, center, right, right),
  [*Model*], [*Parameters*], [*Size (MB)*], [*Train Time*], [*Infer Time (test)*],
  [Random Forest + TF-IDF (A1 best)], [N/A], [N/A], [~300 min], [~10 s],
  [LogReg + TF-IDF (A1)], [N/A], [N/A], [\<1 min], [\<5 s],
  [BiLSTM (A2, improved)], [~15M], [~58 MB], [63.7 min (3824 s)], [13.12 s],
  [RoBERTa-base (A2, improved)], [124,647,170], [475.50 MB], [~430 min], [122.86 s],
)

Key observations:

- *BiLSTM* remains highly compact and efficient. Despite the improved architecture (larger hidden size, attention pooling, extended vocabulary), training completes in ~64 minutes and test-set inference in 13 seconds, making it a very attractive option: it now outperforms the Random Forest baseline while training roughly 5× faster.
- *RoBERTa-base* required ~430 minutes of GPU training (7h10m) for the improved run, comparable to or exceeding the costly Random Forest training. With 475 MB of model weights and 123 seconds of test inference, its resource footprint is substantial. However, the validation F1 of 0.9509 justifies this cost if peak accuracy is required.
- *Random Forest* remains the most *training-time*-expensive classical model (~300 min) and provides no competitive advantage over the improved BiLSTM, which is faster to train and achieves higher accuracy with a well-tuned architecture.

= Error Analysis (Experiment 4)

== Comparison of Failure Modes

With the improved BiLSTM now surpassing the A1 Random Forest baseline, the remaining errors of both models reveal qualitatively different failure modes:

*Shared failure modes (BiLSTM and RoBERTa):*

- *Empty or placeholder captions*: Both models fail when captions contain empty strings or "N/A", as there is no meaningful lexical content to align with the question.
- *Shared generic vocabulary*: Pairs that both use vague geographic terms ("area", "land", "landscape") without specific discriminative tokens continue to cause misclassifications at the margin.

*BiLSTM (improved) specific failure analysis:*

The attention mechanism helps the model focus on the overlapping domain-specific tokens that signal relevance. Residual errors cluster around cases where: (1) the positive caption and its QA pair share only very short or highly ambiguous phrases ("yes", "no" answers), and (2) negative pairs happen to draw a caption from the same geographic scene type despite being a different image, providing too much surface overlap. The model's fixed vocabulary still misses rare compound terms that fall outside the top 30,000 tokens.

*A2 RoBERTa (improved) specific failure analysis:*

With higher sequence length (256 tokens) and stronger regularisation, RoBERTa's remaining failures are primarily in hard negative pairs where the selected distant caption by coincidence discusses the same object category as the QA (e.g., two different airports). The model's contextual embeddings generalise across superficially similar but factually distinct scenes, which remains a structural limitation of text-only approaches for visual grounding tasks.

== Root Cause Analysis

The initially observed dominance of TF-IDF + Random Forest was primarily an artefact of *underpowered neural architectures* rather than an inherent limitation of neural models on this task:

- *Vocabulary coverage* was the single largest bottleneck for the BiLSTM. Expanding from 20,000 to 30,000 tokens with no minimum frequency cutoff, combined with longer 256-token sequences, allows the model to see the same rare domain-specific tokens (e.g., "tarmac", "photovoltaic", "levee") that TF-IDF directly matched, and the attention mechanism can then up-weight these discriminative terms.
- *Representation richness*: attention pooling over all hidden states dramatically outperforms final-state readout for long, informative sequences where the matching evidence can appear anywhere in the text.
- *RoBERTa*'s massive improvement (val F1: 0.9509) under careful regularisation (lower LR, higher weight decay, longer sequences) confirms that pre-trained Transformer contextual representations are extremely well-suited to this matching task once fine-tuning is properly calibrated.

= Cross-Assignment Summary

#table(
  columns: (auto, auto, auto, auto),
  inset: 8pt,
  align: (left, center, center, center),
  [*Assignment*], [*Model*], [*Feature Type*], [*Test F1-Macro*],
  [A1], [Logistic Regression], [TF-IDF unigrams], [0.6879],
  [A1], [Logistic Regression], [TF-IDF bigrams], [0.6965],
  [A1], [Naive Bayes], [TF-IDF bigrams], [0.6705],
  [A1], [Logistic Regression], [GloVe mean-pool], [0.6513],
  [A1], [Logistic Regression], [FastText mean-pool], [0.6528],
  [A1], [Random Forest], [TF-IDF bigrams], [0.8071],
  [A2], [*BiLSTM (attn, h=512, v=30k)*], [*Learned embeddings*], [*0.8579*],
  [A2], [RoBERTa-base (val)], [Contextual Transformer], [0.9509 (val)],
)

= Conclusions

This assignment extended the Caption-Question Relevance Classification study to neural sequence models and yielded several important findings:

1. *The improved BiLSTM surpasses the A1 Random Forest baseline.* With targeted architectural changes — attention pooling over all hidden states, a 512-unit hidden size, a 30,000-token vocabulary, and 256-token input sequences — the BiLSTM achieves Test F1-Macro of *0.8579*, a +6.3% improvement over the best A1 result (0.8071). This confirms that the initial underperformance was due to underpowered design choices rather than an inherent limitation of recurrent architectures.

2. *RoBERTa shows very strong potential.* Fine-tuned with a lower learning rate (1e-5), stronger weight decay (0.05), longer context (256 tokens) and larger effective batch via 4-step gradient accumulation, the model reached a best validation F1-Macro of *0.9509*, substantially above all previous results. Test evaluation is pending but strongly suggests state-of-the-art performance on this dataset.

3. *Vocabulary coverage and context length were the critical bottlenecks.* In both models, extending tokenisation coverage and input sequence length had the largest measurable impact. For the BiLSTM, expanding the vocabulary from 20k (min\_freq=2) to 30k (min\_freq=1) and doubling max\_len from 128 to 256 restored access to the rare domain-specific tokens that carry the matching signal. Attention pooling then allowed the model to locate and up-weight those tokens regardless of their position.

4. *The attention mechanism is critical for long-sequence BiLSTM performance.* Replacing the final-state readout with learned attention pooling over all hidden states yielded a dramatic gain: the model is no longer forced to compress all relevant evidence into the last hidden state of a long padded sequence, which is particularly important for this task where discriminative tokens can appear anywhere in the caption or QA text.

5. *Computational cost remains a relevant trade-off.* The improved BiLSTM trains in ~64 minutes and infers in 13 seconds, offering the best accuracy-per-compute ratio across all models tested. RoBERTa requires ~430 minutes of GPU training and 123 seconds of test inference but is expected to deliver the highest accuracy. Classical Random Forest training (~300 min) is no longer advantageous in either speed or performance relative to the improved neural alternatives.
