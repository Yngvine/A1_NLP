
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
+ *Experiment 2 — Computational Cost*: parameters, model size (MB), training time, and inference time comparison.
+ *Experiment 3 — Learning Curve Analysis*: train each model on 25%, 50%, 75%, and 100% of training data; evaluate on fixed test set; identify the crossover point where neural models surpass the A1 baseline.
+ *Experiment 4 — Error Analysis*: qualitative comparison of RoBERTa failure cases vs. A1 baseline failure cases.


= Results

== Overall Architecture Comparison (Experiment 1)

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: (left, center, center),
  [*Model*], [*Test F1-Macro*], [*vs. A1 Best*],
  [A1 Best: Random Forest + TF-IDF Bigrams], [0.8071], [—],
  [A1: Logistic Regression + TF-IDF Bigrams], [0.6896], [-17.0%],
  [A2: BiLSTM (2-layer, attn, hidden=512, emb=200)], [*0.8509*], [+5.4%],
  [A2: RoBERTa-base (fine-tuned)], [*0.9502*], [+17.7%],
)

Both neural architectures, after architectural improvements, now surpass the A1 Random Forest baseline. The improved BiLSTM achieves Test F1-Macro of *0.8509*, a gain of *+5.4%* over the best A1 result (0.8071). RoBERTa achieves a test F1-Macro of *0.9502*, exceeding the baseline by *+17.7%*. These results stand in stark contrast to the initial runs (BiLSTM: 0.6056, RoBERTa: 0.6582 with smaller context window) and confirm that the initial underperformance was due to underpowered design choices rather than an inherent limitation of neural models on this task.


== Computational Cost (Experiment 2)

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 8pt,
  align: (left, center, center, right, right),
  [*Model*], [*Parameters*], [*Size (MB)*], [*Train Time*], [*Infer Time (test)*],
  [Random Forest + TF-IDF (A1 best)], [N/A], [N/A], [~300 min], [~10 s],
  [LogReg + TF-IDF (A1)], [N/A], [N/A], [\<1 min], [\<5 s],
  [BiLSTM (A2, improved)], [11,380,867], [43.41 MB], [67.4 min (1h 7min)], [12.70 s],
  [RoBERTa-base (A2, improved)], [124,647,170], [475.50 MB], [~430 min (7h10m)], [231.40 s],
)

Key observations:

- *BiLSTM* remains highly compact and efficient. Despite the improved architecture (larger hidden size, attention pooling, extended vocabulary), training completes in ~67 minutes and test-set inference in ~13 seconds, making it a very attractive option: it now outperforms the Random Forest baseline while training roughly 4× faster.
- *RoBERTa-base* required ~430 minutes of GPU training (7h10m) for the improved run, comparable to or exceeding the costly Random Forest training. With 475 MB of model weights and 231 seconds of test inference, its resource footprint is substantial. However, the test F1 of *0.9502* justifies this cost if peak accuracy is required.
- *Random Forest* remains the most *training-time*-expensive classical model (~300 min) and provides no competitive advantage over the improved BiLSTM, which is faster to train and achieves higher accuracy with a well-tuned architecture.

== Learning Curve Analysis (Experiment 3)

We trained each neural model on stratified subsets of 25%, 50%, 75%, and 100% of the training set and evaluated each checkpoint on the fixed held-out test set. The A1 Random Forest baseline (F1 = 0.8071) is plotted as a reference line.

#table(
  columns: (auto, auto, auto, auto),
  inset: 8pt,
  align: (left, center, center, center),
  [*Training subset*], [*Train examples*], [*BiLSTM F1-Macro*], [*RoBERTa F1-Macro*],
  [25%],  [53,480],  [0.7187], [0.8823],
  [50%],  [106,961], [0.7843], [0.9159],
  [75%],  [160,442], [0.8198], [0.9384],
  [100%], [213,923], [0.8509], [0.9502],
)

*At what size do neural models surpass the A1 baseline?*

- *RoBERTa* surpasses the A1 RF baseline already at *25% of the training data* (F1 = 0.8823 > 0.8071). Its pre-trained contextual representations transfer effectively even with only ~53,000 fine-tuning examples.
- *BiLSTM* crosses the A1 baseline between *50%* (F1 = 0.7843) and *75%* (F1 = 0.8198) of training data, corresponding to roughly *~142,000 examples*. Since it learns representations entirely from scratch, it requires substantially more labelled data to reach the same performance level.

*Does the Transformer benefit more from additional data than the LSTM?*

In absolute terms the BiLSTM gains *+0.1322* F1 points from 25% to 100% of data, versus only *+0.0679* for RoBERTa. The BiLSTM learning curve is therefore steeper: each additional batch of training examples has a larger marginal impact because the model must learn both the language representation and the classification signal simultaneously. RoBERTa, by contrast, enters fine-tuning with powerful pre-trained representations and exhibits clear *diminishing returns* past 75% (Δ = 0.0118 from 75% to 100%), indicating that its capacity is not the bottleneck — the remaining errors stem from fundamental limitations of text-only relevance modelling.

In relative terms, however, RoBERTa achieves far higher absolute performance at every data point, showing that pre-training provides a high performance floor that LSTM cannot match regardless of training set size at this scale.

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

Out of the 59,424 test examples, RoBERTa produced *2,457 False Positives* and only *498 False Negatives*, revealing a clear asymmetry: the model is far more likely to incorrectly predict "related" than to miss a genuine match.

== Representative False Positives (model predicted Related, true label = Unrelated)

_Example 1_: The caption describes a large industrial warehouse facility with white-roofed buildings and bus parking, while the question asks _"Are there any small vehicles in the image? Yes, there are 24 small vehicles."_ Both caption and QA discuss vehicles in an industrial context, so RoBERTa conflates the surface lexical overlap ("vehicles", "parking") with genuine relevance despite the caption depicting a bus depot and the QA belonging to a different scene.

_Example 2_: The caption describes a rural landscape with ponds, agricultural fields, and farm buildings. The question asks _"How many waters are there in the image? There are 3 waters."_ RoBERTa is misled by the spatial proximity of "water" tokens in the caption and a QA that counts water bodies, even though they originate from different images. The model cannot resolve that the specific count must align with the specific image.

_Example 3_: A caption about a lake shoreline with dense tree cover and docked boats is paired with the question _"How many ships are there in the image? There are 2 ships."_ RoBERTa treats "boats" and "ships" as semantically equivalent (correctly, contextually) and the count happens to match, resulting in a false match.

== Representative False Negatives (model predicted Unrelated, true label = Related)

_Example 1_: The caption mentions a large building under construction on the upper left of a grassy field, with a main road and railway; the question asks _"Are there any buildings in the image? Yes, there are 25 buildings."_ The count (25) is much higher than the number of buildings visually salient in the caption text, and the model's high confidence threshold apparently prevents it from predicting "related" despite the correct pairing.

_Example 2_: The caption describes an industrial complex with warehouses, trucks, a roundabout and a river to the north. The question asks _"What lies directly north of the industrial complex? A river lies directly north of the industrial complex."_ This is a clear correct pair, but the long, descriptive caption dilutes the specific relational detail ("north / river") within a dense paragraph, and the model fails to attend to it.

== Summary

The error pattern confirms the qualitative analysis: RoBERTa's failure mode is dominated by *false positives driven by thematic lexical overlap* (industrial scenes, water counts, vehicle mentions) rather than by missed genuine matches. False negatives tend to occur when the specific discriminative detail (a count, a spatial relation) is buried in a long descriptive caption, despite the broader context being consistent.


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
  [A2], [*BiLSTM (attn, h=512, v=30k)*], [*Learned embeddings*], [*0.8509*],
  [A2], [*RoBERTa-base*], [*Contextual Transformer*], [*0.9502*],
)

= Conclusions

This assignment extended the Caption-Question Relevance Classification study to neural sequence models and yielded several important findings:

1. *The improved BiLSTM surpasses the A1 Random Forest baseline.* With targeted architectural changes — attention pooling over all hidden states, a 512-unit hidden size, a 30,000-token vocabulary, and 256-token input sequences — the BiLSTM achieves Test F1-Macro of *0.8509*, a +5.4% improvement over the best A1 result (0.8071). This confirms that the initial underperformance was due to underpowered design choices rather than an inherent limitation of recurrent architectures.

2. *RoBERTa sets the new state of the art.* Fine-tuned with a lower learning rate (1e-5), stronger weight decay (0.05), longer context (256 tokens) and larger effective batch via 4-step gradient accumulation, the model achieved a best validation F1-Macro of *0.9509* and a test F1-Macro of *0.9502*, substantially above all previous results and confirming state-of-the-art performance on this dataset.

3. *Vocabulary coverage and context length were the critical bottlenecks.* In both models, extending tokenisation coverage and input sequence length had the largest measurable impact. For the BiLSTM, expanding the vocabulary from 20k (min\_freq=2) to 30k (min\_freq=1) and doubling max\_len from 128 to 256 restored access to the rare domain-specific tokens that carry the matching signal. Attention pooling then allowed the model to locate and up-weight those tokens regardless of their position.

4. *The attention mechanism is critical for long-sequence BiLSTM performance.* Replacing the final-state readout with learned attention pooling over all hidden states yielded a dramatic gain: the model is no longer forced to compress all relevant evidence into the last hidden state of a long padded sequence, which is particularly important for this task where discriminative tokens can appear anywhere in the caption or QA text.

5. *Computational cost remains a relevant trade-off.* The improved BiLSTM trains in ~67 minutes and infers in ~13 seconds, offering the best accuracy-per-compute ratio across all models tested. RoBERTa requires ~430 minutes of GPU training and 231 seconds of test inference but delivers the highest accuracy (test F1 = 0.9502). Classical Random Forest training (~300 min) is no longer advantageous in either speed or performance relative to the improved neural alternatives.

6. *Pre-training provides a decisive data-efficiency advantage.* The learning curve analysis (Experiment 3) shows that RoBERTa surpasses the A1 RF baseline with only 25% of training data (F1 = 0.8823), while the BiLSTM requires between 50% and 75% (~142,000 examples) to cross the same threshold. At every training size the Transformer dominates, and its curve plateaus past 75%  (Δ = +0.0118), whereas the BiLSTM learning curve is steeper in absolute terms (+0.1322) and shows no saturation, suggesting that with even more data the gap between the two models could narrow slightly — but the Transformer's pre-training advantage persists throughout.
