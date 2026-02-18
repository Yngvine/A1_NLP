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

The mean cosine distance of our negative pairs is 0.1583 vs 0.1193 for random sampling, a +32.7% improvement.

#figure(
  image("distribution.png", width:90%),
  caption: [Distribution of cosine distances between question and caption embeddings for random sampling (red) vs. tag-based semantic distance sampling (green).],
)

The final dataset is split into training (80%) and testing (20%) sets using stratified sampling (`random_state=42`) to maintain class balance. The total dataset size is 297,116 examples, far surpassing the minimum requirements, with a 50/50 representation of each class.

= Text Preprocessing
<text-preprocessing>

We define and compare four preprocessing strategies as an ablation study:

- *Raw*: No preprocessing.
- *Lowercase*: Case folding.
- *Clean*: Case folding + punctuation removal.
- *Aggressive*: Case folding + punctuation removal + stopword removal (scikit-learn English stop words).

All strategies are applied uniformly to the combined `[Caption] [SEP] [Question Answer]` input string before feature extraction.

= Feature Representations

We implement and compare two families of features.

== Sparse Features: TF-IDF

We use scikit-learn's `TfidfVectorizer` with `max_features=5000`. Three n-gram configurations are tested:

- *Unigrams*: `ngram_range=(1, 1)`
- *Bigrams (uni+bi)*: `ngram_range=(1, 2)`
- *Trigrams (uni+bi+tri)*: `ngram_range=(1, 3)`

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

== Overall Performance Comparison

#figure(
  image("f1.png", width: 90%),
  caption: [Comparison of Test F1-Macro scores across all experimental configurations. The chart summarizes the impact of n-gram selection, preprocessing levels, and model architectures.],
)

The following subsections detail the individual ablation studies and comparisons summarized above.


== N-gram Comparison (Experiment 1)

#table(
  columns: (auto, auto, auto, auto),
  inset: 8pt,
  align: (left, center, center, center),
  [*Configuration*], [*CV F1-Macro*], [*Test F1-Macro*], [*Test Accuracy*],
  [TF-IDF Unigrams + LogReg], [0.6877 ± 0.0012], [0.6879], [0.6881],
  [TF-IDF Bigrams + LogReg], [0.6969 ± 0.0020], [0.6965], [0.6965],
  [TF-IDF Trigrams + LogReg], [0.6944 ± 0.0023], [0.6944], [0.6944],
)

The results indicate that including bigrams improves performance over using only unigrams (Test F1-Macro $+0.86$ percentage points). This suggests that capturing local context through two-word phrases (e.g., "residential area", "water body") provides valuable discriminative signals. However, extending to trigrams does not yield further improvement, likely due to increased data sparsity which offsets the potential gain from longer context.

== Preprocessing Ablation (Experiment 2)

#table(
  columns: (auto, auto, auto, auto),
  inset: 8pt,
  align: (left, center, center, center),
  [*Strategy*], [*CV F1-Macro*], [*Test F1-Macro*], [*Test Accuracy*],
  [Raw], [0.6969 ± 0.0020], [0.6965], [0.6965],
  [Lowercase], [0.6969 ± 0.0020], [0.6965], [0.6965],
  [Clean], [0.6965 ± 0.0019], [0.6983], [0.6983],
  [Aggressive], [0.6948 ± 0.0023], [0.6980], [0.6980],
)

The results suggest that minimal preprocessing is sufficient for this task. The *Raw* and *Lowercase* strategies yield identical performance, indicating that capitalization does not carry significant signal. Interestingly, while *Clean* and *Aggressive* strategies show a slight drop in cross-validation performance, they achieve marginally higher scores on the test set. However, given the overlap in standard deviations and the small magnitude of differences ($<0.2\%$), we conclude that sophisticated preprocessing does not provide a robust advantage over simple case folding for TF-IDF based models in this domain.

== Model Comparison (Experiment 3)

#table(
  columns: (auto, auto, auto, auto),
  inset: 8pt,
  align: (left, center, center, center),
  [*Model*], [*CV F1-Macro*], [*Test F1-Macro*], [*Test Accuracy*],
  [Logistic Regression], [0.6989 ± 0.0020], [0.6896], [0.6896],
  [Naive Bayes], [0.6671 ± 0.0022], [0.6705], [0.6708],
  [Random Forest], [0.8030 ± 0.0014], [0.8071], [0.8072],
)

The Random Forest classifier significantly outperforms both Logistic Regression and Naive Bayes, achieving a Test F1-Macro of 0.8071 ($+17.0\%$ relative to LogReg). This suggests that non-linear interactions between features are crucial for this task. However, this performance comes at a substantial computational cost: while the linear models train in under a minute, Random Forest requires approximately 300 minutes to complete the training process on this dataset.

== Hyperparameter Optimization (Experiment 4)

We ran the grid search over Logistic Regression parameters because it is the model that offers the best balance of performance and training time for TF-IDF features.

#table(
  columns: (auto, auto),
  inset: 8pt,
  align: (left, left),
  [*Parameter*], [*Best Value*],
  [Regularization $C$], [0.1],
  [Solver], [liblinear],
  [CV F1-Macro], [0.6989],
  [Test F1-Macro], [0.6896],
)

The grid search identifies $C=0.1$ with the `liblinear` solver as the optimal configuration. This preference for stronger regularization (compared to the default $C=1.0$) suggests that controlling model complexity is beneficial when dealing with high-dimensional sparse features (TF-IDF bigrams), likely helping to reduce overfitting. Despite the optimization, the performance gain over the baseline Logistic Regression model is modest, confirming that the default hyperparameters are already close to optimal for this specific task and feature set.

== Sparse vs Dense Feature Comparison (Experiments 3 & 5)

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 8pt,
  align: (left, center, center, center, center),
  [*Representation*], [*Model*], [*Dimensions*], [*Test F1-Macro*], [*Test Accuracy*],
  [TF-IDF (Sparse)], [LogReg], [5,000], [0.6896], [0.6896],
  [GloVe (Dense)], [LogReg], [100], [0.6513], [0.6517],
  [FastText (Dense)], [LogReg], [300], [0.6528], [0.6534],
)

The results demonstrate that the sparse TF-IDF representation significantly outperforms both dense embedding methods (GloVe and FastText). TF-IDF achieves a Test F1-Macro of 0.6896, compared to 0.6513 for GloVe and 0.6528 for FastText. This suggests that for determining caption-question relevance, precise lexical matching (captured by TF-IDF) is more critical than the semantic generalization provided by averaged word embeddings. The aggregation of word embeddings into a single document vector likely dilutes key discriminative signals present in specific keywords, which are preserved in the high-dimensional sparse representation of TF-IDF. Additionally, despite FastText having a larger vocabulary and higher dimensionality than GloVe, it offers only a marginal improvement ($+0.15\%$), reinforcing the limitation of mean-pooled embeddings for this pair-matching task.

#figure(
  image("sparseVSdense.png", width: 70%),
  caption: [Comparison of Test F1-Macro scores for sparse (TF-IDF) vs. dense (GloVe, FastText) feature representations using Logistic Regression.],
)

= Error Analysis

== Confusion Matrix

#figure(
  image("confusionMatrix.png", width: 60%),
  caption: [Confusion matrix for the best-performing model (Random Forest with TF-IDF bigrams) on the test set.],
)

The confusion matrix reveals a relatively balanced error distribution, with a slight tendency towards False Negatives (9,210) over False Positives (8,825). This indicates that the model is marginally more likely to miss a related pair (predicting Unrelated) than to hallucinate a relationship where none exists. The high number of errors in both off-diagonal quadrants suggests that while the model captures general patterns, it struggles with the nuances that distinguish truly related captions from those that are merely topically similar, a challenge exacerbated by our tag-based hard negative mining strategy.

== Discriminative Features

We extract the top-weighted TF-IDF features from the Logistic Regression coefficients. Features with the highest positive coefficients predict the *Related* class, while those with the most negative coefficients predict *Unrelated*.

#figure(
  image("top10features.png", width: 90%),
  caption: [Top discriminative features for Related (positive coefficients) and Unrelated (negative coefficients) classes from the Logistic Regression model.],
)

The analysis of the top discriminative features reveals interesting patterns in how the model determines relevance:

- *Content Overlap*: The positive features are dominated by specific scene elements appearing in bigrams ("several buildings", "residential or", "trees and"). This confirms that the model relies heavily on finding matching content descriptors between the caption and the QA pair. When both parts of the input mention specific objects like "buildings" or "roads", the likelihood of them being related increases significantly.

- *Structural Artifacts significantly predict "Unrelated"*: The most negative feature is `sep`, which is our separator token. Since every example contains exactly one `[SEP]`, its high negative weight is counter-intuitive. However, combined with features like `sep what` and `sep how`, this suggests the model might be latching onto the syntax of questions in negative pairs. The presence of `what` and `how` immediately after the separator (indicating the start of the question) is strongly associated with the Unrelated class in the learned weights. This could imply that certain question types (starting with "What" or "How") are harder to match or more frequent in the negative samples generated by our distance-based strategy.

- *Vague Terminology*: Negative features also include generic terms like "blank", "features", and "environment". This suggests that when the text relies on vague or high-level descriptors rather than specific object names, the model is less confident in predicting a relationship, or that these terms appear frequently in mismatched pairs where precise lexical overlap is missing.

== Qualitative Failure Analysis

We manually inspect 10 misclassified examples (5 false positives, 5 false negatives) to identify recurring error patterns.

*False Positives* (predicted Related, actually Unrelated):

+ Caption: The image presents a diverse coastal landscape where the left side is dominated by human activity, including a dense urban or residential area in the upper left and extensive industrial or port facili[...] \
   Question: Is the paved surface area larger or smaller compared to the green open land in the image? The paved surface, which is the runway, dominates the central and lower portions, but the green open land on e[...] 
+ Caption: [Empty caption] \
   Question: Where is the water body located in relation to the agricultural fields? The water body is located adjacent to or near the agricultural fields. 
+ Caption: [Empty caption] \
   Question: N/A 
+ Caption: The image primarily features open land organized into neat rows, which could indicate a cemetery or agricultural use, occupying the majority of the landscape. There are several paved roads intersectin... \
   Question: What type of area is depicted in the image? An urban area with significant building coverage. 
+ Caption: The image depicts a cityscape dominated by numerous residential buildings arranged in a grid-like pattern, with a prominent road cutting across the upper portion of the frame. The spaces between the b... \
   Question: What are the main features characterizing the landscape in the image? The landscape is characterized by industrial infrastructure, transportation facilities, and limited natural elements. 

*False Negatives* (predicted Unrelated, actually Related):

+ Caption: The image predominantly displays a suburban residential area characterized by tightly packed houses occupying most of the land, with only small portions of green space such as lawns and trees visible ... \
   Question: What pattern do the streets in the neighborhood form? The streets form a grid pattern.
+ Caption: The image shows a landscape dominated by dense green forest covering most of the area, with a wide, multi-lane road running vertically through the center. To the left of the road, there is a small, da... \
   Question: Are there more buildings or roads in the image? There are more roads (2) than buildings (1). 
+ Caption: The landscape is dominated by a residential area with tightly packed houses, each with individual yards and driveways, primarily occupying the left and upper portions of the image. A large, white-roof... \
   Question: Where is the large white-roofed building complex located in the image? It is located in the lower right portion of the image. 
+ Caption: The image is largely characterized by natural land cover, with dense clusters of trees and shrubs dominating most of the scene, especially in the central and right portions. A road runs vertically alo... \
   Question: What type of area does the image primarily depict? The image primarily depicts a natural area dominated by vegetation with minimal human-made features. 
+ Caption: [Empty caption] \
   Question: Is dense vegetation more extensive than water bodies in the image? Yes, dense vegetation appears more extensive than water bodies. 

These error examples highlight several critical limitations of the TF-IDF approach:

- *Empty Captions*: Several failures (both FP and FN) involve empty or N/A captions. The model possibly learns a bias for the specific token distribution of empty strings or short placeholders, rather than actual content matching. This highlights an oversight in data generation and the need for better handling of missing information.
- *Handling of "N/A"*: Similarly, the presence of "N/A" in questions suggests data quality issues that the model propagates.
- *Lexical Overlap vs. Semantic Truth*: In False Positives, the model is often fooled by shared vocabulary (e.g., "area", "land", "landscape") even when the specific details are contradictory (e.g., "residential" in caption vs. "industrial" in Q&A).
- *Specificity in False Negatives*: In False Negatives, the model fails despite strong apparent overlaps (e.g., "white-roofed building"). This suggests that when the phrasing differs slightly ("houses" vs "building complex") or when the relationship relies on spatial reasoning ("lower right") which isn't captured by bag-of-words, the simple lexical matching falls short.

= Conclusions

This study established a robust baseline for Caption-Question Relevance Classification in the remote sensing domain, demonstrating that effective relevance matching is possible using only textual features. Our investigation yields three primary insights:

1.  *Lexical Precision Dominates*: Sparse TF-IDF representations significantly outperformed dense embeddings (GloVe, FastText) by approximately 4% in F1-score. This indicates that for this specific task, exact keyword matching (e.g., specific terrain features like "residential area") provides a stronger signal than the averaged semantic representations of static word embeddings. Bigrams provided a crucial boost over unigrams by capturing local context.

2.  *Non-Linearity is Key*: The Random Forest classifier achieved the highest performance (Test F1-Macro 0.8071), surpassing linear models such as Logistic Regression by a wide margin ($+17.0\%$). This suggests that the relationship between caption and question content is complex and relies heavily on feature interactions rather than independent additive signals.

3.  *Impact of Hard Negatives*: Our tag-based sampling strategy successfully created a challenging dataset where random guessing or simple topic matching is insufficient. However, the error analysis reveals that models still struggle with subtle semantic contradictions where vocabulary overlaps but the underlying facts differ (e.g., "residential" vs "industrial"), highlighting the limitations of bag-of-words approaches.

The investigation demonstrates that while the established baseline (Random Forest + TF-IDF) achieves a respectable Test F1-Macro 0.8071, there is a clear upper bound to what purely lexical models can achieve, especially given the +17.0% gap between linear and non-linear models. 