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

= Dataset
We use the [RSVLM-QA](https://github.com/StarZi0213/RSVLM-QA) dataset, a large-scale benchmark for Remote Sensing Vision Language Models. It integrates imagery from prominent remote sensing datasets (WHU, LoveDA, INRIA, and iSAID) and provides rich annotations generated through a dual-track pipeline involving GPT-4.1 and automated analysis of segmentation data.

== Original
The original dataset consists of 13,820 images and 162,373 VQA pairs stored in JSONL format. Each record contains:
- An image identifier and file path.
- A list of `vqa_pairs`, which includes the image caption (stored as a QA pair), visual questions, and their corresponding answers.
- Semantic `tags` describing features present in the image (e.g., "buildings", "trees", "water").
- Spatial relation descriptions and object counts derived from segmentation masks.

== Restructured
To facilitate efficient processing for our task, we converted the original JSONL data into a structured Parquet format. We separated the data into two relational tables linked by the image ID:
1. *Captions Table*: Contains the image metadata (`id`, `image` path), the extracted descriptive `caption`, and the semantic `tags`.
2. *Questions Table*: Contains the individual QA pairs (`question`, `answer`, `question_type`) associated with each image.

This restructuring allows for faster loading and easier manipulation of the data, specifically identifying the correct caption for each question.

== Classification Dataset
For the binary classification task (Caption-Question Relevance), we constructed a balanced dataset of positive and negative pairs:

- *Positive Examples (Class 1)*: Formed by pairing an image's caption with one of its valid Questions and Answers. We concatenate the Question and Answer to provide richer context using the format: `[Caption] [SEP] [Question] [Answer]`.

- *Negative Examples (Class 0)*: Constructed using a *Tag-Based Semantic Distance* strategy rather than random sampling.
  - We compute embeddings for each image based on its semantic `tags` using FastText (or GloVe if the embeddings are not available).
  - For each question, we identify images that are semantically distant (based on cosine distance of tag embeddings) from the source image.
  - We pair the question with a caption from one of these distant images.

This strategy ensures that negative pairs are truly unrelated, preventing "false negatives" where a random sampling might accidentally pair a question with a semantically similar image (e.g., pairing a question about "buildings" with a different image that also contains buildings).

The final dataset is split into training (80%) and testing (20%) sets using stratified sampling to maintain class balance.