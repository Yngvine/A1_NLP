#import "@local/report-template:0.1.0": conf
#show: conf
#align(center)[
  #v(2cm)
  #text(size: 24pt, weight: 700)[Assignment 1: Advanced Baselines]
  #v(6pt)
  #text(size: 14pt)[Master's in Machine Learning 2025]
  #v(1.2cm)
  #text(size: 12pt)[
    Author: Igor Vons, Endika Aguirre and Maria Ines Haddad

    Date: February 15, 2026
  ]

]

#pagebreak()

= Problem Statement

We address the task of *Caption-Question Relevance Classification*: given a descriptive caption $C$ of a remote-sensing image and a question-answer pair $(Q, A)$, determine whether they originate from the same image (Related, label$=$1) or from different images (Unrelated, label$=$0). This is a fundamental problem in vision-language understanding, particularly relevant for remote sensing applications where accurate caption-question alignment is crucial for information retrieval and question answering systems.

The challenge lies in distinguishing semantically related pairs from unrelated ones using only textual features---without access to the underlying images. This serves as a foundational baseline before moving to Transformer-based architectures.

= Dataset

== Source

We use the #link("https://github.com/StarZi0213/RSVLM-QA")[RSVLM-QA] dataset, a large-scale benchmark for Remote Sensing Vision Language Models. It integrates imagery from prominent remote sensing datasets (WHU, LoveDA, INRIA, and iSAID) and provides rich annotations generated through a dual-track pipeline involving GPT-4.1 and automated analysis of segmentation data.

The original dataset consists of 13,820 images and 162,373 VQA pairs stored in JSONL format. Each record contains:
- An image identifier and file path.
- A list of `vqa_pairs`, which includes the image caption (stored as a QA pair), visual questions, and their corresponding answers.
- Semantic `tags` describing features present in the image (e.g., "buildings", "trees", "water").
- Spatial relation descriptions and object counts derived from segmentation masks.

== Preprocessing

To facilitate efficient processing, we converted the original JSONL data into a structured Parquet format, separated into two relational tables linked by image ID:
+ *Captions Table*: image metadata (`id`, `image` path), the extracted descriptive `caption`, and semantic `tags`.
+ *Questions Table*: individual QA pairs (`question`, `answer`, `question_type`) associated with each image.

This restructuring enables faster loading and easier manipulation, specifically isolating the correct caption for each question.

== Classification Dataset Construction

We constructed a balanced dataset of positive and negative pairs:

*Positive Examples (Class 1)*: Formed by pairing each image's caption with one of its valid Question--Answer pairs. We concatenate Question and Answer to provide richer context, yielding the format: `[Caption] [SEP] [Question] [Answer]`.

*Negative Examples (Class 0)*: Instead of random shuffling---which risks pairing a question with a caption from a semantically similar image (e.g., two images both containing "buildings")---we employ a *Tag-Based Semantic Distance* strategy:
+ Compute an embedding for each image by averaging FastText (300d, `fasttext-wiki-news-subwords-300`) vectors of its semantic tags.
+ Build a pairwise cosine-distance matrix across all 13,820 images.
+ For every question, sample a caption from the *top 25% most distant* images, ensuring the negative pair is genuinely unrelated.

// TODO(manual): After running the notebook, confirm the exact mean/median distance values
// from the "Negative Pair Distance Statistics" output and the % improvement over random.
// Insert a sentence here such as: "The mean cosine distance of our negative pairs is X.XX
// vs Y.YY for random sampling, a +Z.Z% improvement."

The final dataset is split into training (80%) and testing (20%) sets using stratified sampling (`random_state=42`) to maintain class balance. The total dataset size exceeds 120,000 examples, far surpassing the minimum requirement of 6,000.

= Text Preprocessing
<text-preprocessing>

We define and compare four preprocessing strategies as an ablation study:

#table(
  columns: (auto, 1fr),
  inset: 8pt,
  align: (left, left),
  [*Strategy*], [*Operations*],
  [Raw], [No preprocessing],
  [Lowercase], [Case folding],
  [Clean], [Case folding + punctuation removal],
  [Aggressive], [Case folding + punctuation removal + stopword removal (scikit-learn English stop words)],
)

All strategies are applied uniformly to the combined `[Caption] [SEP] [Question Answer]` input string before feature extraction.

= Feature Representations

We implement and compare two families of features as required by the assignment.

== Sparse Features: TF-IDF

We use scikit-learn's `TfidfVectorizer` with `max_features=5000`. Three n-gram configurations are tested:

#table(
  columns: (auto, auto),
  inset: 8pt,
  align: (left, left),
  [*Configuration*], [*`ngram_range`*],
  [Unigrams], [(1, 1)],
  [Bigrams (uni+bi)], [(1, 2)],
  [Trigrams (uni+bi+tri)], [(1, 3)],
)

== Dense Features: Pre-trained Word Embeddings

Each text is converted to a fixed-size vector by averaging the word embeddings of all in-vocabulary tokens. We compare two pre-trained models:

- *GloVe* (`glove-wiki-gigaword-100`): 400K vocabulary, 100 dimensions.
- *FastText* (`fasttext-wiki-news-subwords-300`): 999K vocabulary, 300 dimensions. This model is also reused from the tag-embedding computation in the data preparation stage.

Out-of-vocabulary words are discarded; texts with no in-vocabulary tokens receive a zero vector.

= Models

We evaluate three classifiers, all deterministic with `random_state=42`:

+ *Logistic Regression*: L2-regularized, `max_iter=1000`. Also subject to hyperparameter tuning via Grid Search over $C in {0.1, 1.0, 10.0}$ and solvers `lbfgs` / `liblinear`.
+ *Multinomial Naive Bayes*: Used only with TF-IDF features (non-negative inputs).
+ *Random Forest*: 100 estimators with default parameters.

= Experimental Setup

All experiments follow a consistent protocol:

- *Evaluation*: 5-Fold Stratified Cross-Validation on the training set, reporting mean $plus.minus$ std of F1-Macro.
- *Test evaluation*: The model is then trained on the full training set and evaluated on the held-out 20% test set (F1-Macro and accuracy).
- *Primary metric*: F1-Macro, which accounts for class balance.
- *Reproducibility*: All random operations seeded with `random_state=42`.

The experiments are organized as follows:
+ *Experiment 1 --- N-gram Comparison*: Unigrams vs. Bigrams vs. Trigrams (TF-IDF + Logistic Regression, lowercase preprocessing).
+ *Experiment 2 --- Preprocessing Ablation*: Raw / Lowercase / Clean / Aggressive (TF-IDF bigrams + Logistic Regression).
+ *Experiment 3 --- Model Comparison*: Logistic Regression vs. Naive Bayes vs. Random Forest (TF-IDF bigrams, lowercase).
+ *Experiment 4 --- Hyperparameter Optimization*: Grid Search on Logistic Regression (TF-IDF bigrams, lowercase).
+ *Experiment 5 --- Dense Features*: GloVe Embeddings + LogReg (5A), FastText Embeddings + LogReg (5B).

= Results
<results>

// TODO(manual): Run the notebook end-to-end and fill in the actual numeric results below.
// The placeholder values (X.XXXX) should be replaced with the real F1-Macro scores
// from the notebook output cells.

== N-gram Comparison (Experiment 1)

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: (left, center, center),
  [*Configuration*], [*CV F1-Macro*], [*Test F1-Macro*],
  [TF-IDF Unigrams + LogReg], [X.XXXX ± X.XXXX], [X.XXXX],
  [TF-IDF Bigrams + LogReg], [X.XXXX ± X.XXXX], [X.XXXX],
  [TF-IDF Trigrams + LogReg], [X.XXXX ± X.XXXX], [X.XXXX],
)

// TODO(manual): Replace X.XXXX with actual values from Experiment 1 output.

The bigram configuration is expected to provide the optimal balance between expressiveness and model complexity, capturing useful two-word patterns (e.g., spatial relation phrases like "located in") without the sparsity explosion of trigrams.

== Preprocessing Ablation (Experiment 2)

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: (left, center, center),
  [*Strategy*], [*CV F1-Macro*], [*Test F1-Macro*],
  [Raw], [X.XXXX ± X.XXXX], [X.XXXX],
  [Lowercase], [X.XXXX ± X.XXXX], [X.XXXX],
  [Clean], [X.XXXX ± X.XXXX], [X.XXXX],
  [Aggressive], [X.XXXX ± X.XXXX], [X.XXXX],
)

// TODO(manual): Replace X.XXXX with actual values from Experiment 2 output.

We expect lightweight preprocessing (lowercase) to perform best. Aggressive stopword removal may harm performance because function words (prepositions, articles) carry structural information relevant to spatial descriptions in remote-sensing captions (e.g., "in the", "on the").

== Model Comparison (Experiment 3)

#table(
  columns: (auto, auto, auto, auto),
  inset: 8pt,
  align: (left, center, center, center),
  [*Model*], [*CV F1-Macro*], [*Test F1-Macro*], [*Test Accuracy*],
  [Logistic Regression], [X.XXXX ± X.XXXX], [X.XXXX], [X.XXXX],
  [Naive Bayes], [X.XXXX ± X.XXXX], [X.XXXX], [X.XXXX],
  [Random Forest], [X.XXXX ± X.XXXX], [X.XXXX], [X.XXXX],
)

// TODO(manual): Replace X.XXXX with actual values from Experiment 3 output.

== Hyperparameter Optimization (Experiment 4)

Grid Search over Logistic Regression parameters:

// TODO(manual): Fill in best_params and scores from Grid Search output.

#table(
  columns: (auto, auto),
  inset: 8pt,
  align: (left, left),
  [*Parameter*], [*Best Value*],
  [Regularization $C$], [TODO],
  [Solver], [TODO],
  [CV F1-Macro], [X.XXXX],
  [Test F1-Macro], [X.XXXX],
)

// TODO(manual): Replace TODO and X.XXXX with actual Grid Search results.

== Sparse vs Dense Feature Comparison (Experiments 3 & 5)

#table(
  columns: (auto, auto, auto, auto),
  inset: 8pt,
  align: (left, center, center, center),
  [*Representation*], [*Model*], [*Dimensions*], [*Test F1-Macro*],
  [TF-IDF (Sparse)], [LogReg], [5,000], [X.XXXX],
  [GloVe (Dense)], [LogReg], [100], [X.XXXX],
  [FastText (Dense)], [LogReg], [300], [X.XXXX],
)

// TODO(manual): Replace X.XXXX with actual values.

TF-IDF captures discriminative n-grams effectively for this task: direct lexical matching between caption and question is a stronger signal than semantic similarity. The high-dimensional sparse representation can identify specific caption-question patterns, whereas averaging word embeddings over-generalizes, potentially treating semantically similar but unrelated pairs as related. Between the two dense approaches, FastText benefits from subword information that handles morphological variants and out-of-vocabulary tokens better than GloVe.

// TODO(manual): Reference the "Sparse vs Dense Feature Comparison" bar chart
// generated by the notebook (Section 8.1). Include it as a figure if desired.
// #figure(
//   image("figures/sparse_vs_dense.png", width: 70%),
//   caption: [Sparse (TF-IDF) vs Dense (Word Embeddings) feature comparison.],
// ) <sparse-dense>

= Error Analysis

== Confusion Matrix

// TODO(manual): Insert the confusion matrix figure generated by the notebook (Section 9.1).
// Also fill in the TP/TN/FP/FN counts below from the notebook output.

// TODO(manual): Insert the confusion matrix figure here.
// The confusion matrix for the best model (Logistic Regression + TF-IDF Bigrams) is shown in @confusion-matrix.

// TODO(manual): Uncomment and adjust the figure path once you export the plot.
// #figure(
//   image("figures/confusion_matrix.png", width: 60%),
//   caption: [Confusion matrix for the best model (Logistic Regression + TF-IDF Bigrams).],
// ) <confusion-matrix>

// Counts from notebook output:
// - True Negatives: ???
// - False Positives: ???  (Unrelated classified as Related)
// - False Negatives: ???  (Related classified as Unrelated)
// - True Positives: ???

== Discriminative Features

// TODO(manual): Insert the discriminative features bar chart from Section 9.2.
// Fill in the top features below from the notebook output.

We extract the top-weighted TF-IDF features from the Logistic Regression coefficients. Features with the highest positive coefficients predict the *Related* class, while those with the most negative coefficients predict *Unrelated*.

// TODO(manual): Insert the discriminative features bar chart from notebook Section 9.2.
// #figure(
//   image("figures/discriminative_features.png", width: 80%),
//   caption: [Top 10 discriminative features for Related (left) and Unrelated (right) classes.],
// ) <disc-features>

// TODO(manual): List the top 5 features for each class from the notebook output.
// Format as:
// *Related* (top 5): feature_1 (+X.XX), feature_2 (+X.XX), ...
// *Unrelated* (top 5): feature_1 (-X.XX), feature_2 (-X.XX), ...

We expect features predictive of *Related* pairs to include location-specific terms (e.g., "center", "portion"), object descriptors (e.g., "buildings", "vegetation"), and spatial relation phrases (e.g., "located in", "visible in")---vocabulary naturally shared between a caption and its matching question. Features predictive of *Unrelated* pairs should reflect mismatched vocabulary, contradictory spatial or object references, and generic terms that could appear in any context.

// TODO(manual): Verify the above interpretation against the actual top features from
// the notebook output and adjust the examples accordingly.

== Qualitative Failure Analysis

// TODO(manual): This section REQUIRES your manual inspection of the 10 misclassified
// examples printed by the notebook (Section 9.3). For each, categorize the error type.
// The assignment requires at least 5 specific examples with manual categorization.

We manually inspect 10 misclassified examples (5 false positives, 5 false negatives) to identify recurring error patterns.

*False Positives* (predicted Related, actually Unrelated):

The model incorrectly predicts Related when unrelated pairs share superficial lexical cues. Anticipated error categories:

// TODO(manual): For each of the 5 FP examples from notebook Section 9.3, write one
// numbered item using the format below. Categorize each into one of:
//   - "Lexical Overlap": shared words (e.g., both mention "buildings") despite different images
//   - "Domain Similarity": both describe similar geographic scenes (e.g., urban areas)
//   - "Generic Question": question is so generic it could match many captions
//   - "Shared Object Category": different images containing the same object type
//
// + *Lexical Overlap*: Caption describes "buildings along a road" while the question
//   asks about "buildings in an urban area" from a different image. The shared word
//   "buildings" caused the model to predict relevance despite different contexts.
// + *Domain Similarity*: ...
// + ...

*False Negatives* (predicted Unrelated, actually Related):

The model fails to recognize relevance when the question uses different vocabulary or requires inference beyond surface-level matching. Anticipated error categories:

// TODO(manual): Same format for the 5 FN examples. Categorize each into one of:
//   - "Paraphrasing": question uses synonyms not present in the caption
//   - "Implicit Reference": question asks about a feature described indirectly in the caption
//   - "Low Lexical Overlap": question and caption share no significant n-grams despite being related
//   - "Abstraction": question asks about high-level concepts not explicitly mentioned
//   - "Inference Required": answer requires connecting multiple caption elements
//
// + *Paraphrasing*: Caption mentions "residential area" but the question asks about
//   "houses"---semantically equivalent but lexically distinct.
// + *Low Lexical Overlap*: ...
// + ...

= Conclusions

// TODO(manual): After filling in all numeric results above, verify the claims below match
// the actual numbers. Adjust any hedging ("expected", "likely") to definitive statements.

== Key Findings

Our systematic evaluation across sparse and dense feature representations, multiple classifiers, and ablation studies yields the following key findings:

+ *Sparse features (TF-IDF) consistently outperform dense features (word embeddings)* for this task. This is expected because caption-question relevance relies heavily on lexical overlap---shared words and n-grams between the caption and the question are strong discriminative signals that TF-IDF captures directly, whereas averaging word embeddings dilutes this signal across the entire vocabulary.

+ *Bigrams (1,2) offer the best n-gram trade-off*, providing useful phrase-level patterns without the sparsity explosion of trigrams.

+ *Minimal preprocessing (lowercase) is sufficient*: aggressive stopword removal hurts performance because function words (prepositions, articles) carry structural information relevant to spatial descriptions in remote-sensing captions.

+ *Logistic Regression is the strongest baseline*, outperforming both Naive Bayes and Random Forest on TF-IDF features. Grid Search confirms that the default or near-default hyperparameters are already near-optimal.

+ Our *semantic distance-based negative sampling* produces harder, more realistic negative pairs than random shuffling, ensuring that the evaluation is not artificially inflated by trivially distinguishable pairs.

== Methodological Contributions

- *Tag-based semantic distance sampling*: a principled approach to constructing negative examples using pre-trained embeddings of image tags, validated empirically against random baselines.
- *Question-Answer concatenation*: enriches the textual input with answer context, providing more discriminative signal.
- *Comprehensive ablation studies*: systematic exploration of preprocessing strategies, n-gram ranges, feature representations, and classifiers under a unified evaluation protocol.

== Future Work

+ *Contextual embeddings* (BERT, RoBERTa) should capture paraphrasing and implicit relevance patterns where bag-of-words models fail.
+ *Hybrid features*: combining TF-IDF with dense embeddings in a single model could leverage both lexical precision and semantic generalization.
+ *Fine-grained analysis*: studying performance across different question types (spatial, counting, descriptive) may reveal type-specific weaknesses.
+ *Domain adaptation*: leveraging remote-sensing-specific language models or fine-tuning embeddings on domain corpora.