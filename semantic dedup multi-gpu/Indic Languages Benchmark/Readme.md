# Multilingual Semantic Deduplication: Evaluation on Wikipedia Paraphrases

## Overview

This project evaluates **semantic deduplication** performance across **22 Indic languages** using **five multilingual embedding models**. The deduplication pipeline tests each model's ability to correctly detect paraphrased sentence pairs across translated Wikipedia content.

## Dataset Preparation

1. **Sample Collection**:
   - We collected **100 English Wikipedia documents**.

2. **Paraphrasing**:
   - The 100 documents were paraphrased using [chatgpt_paraphraser_on_T5_base]("https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base").

3. **Translation to 22 Indic Languages**:
   - Each of the 100 original and paraphrased English texts was translated to 22 languages using [sarvamai/sarvam-translate](""https://huggingface.co/sarvamai/sarvam-translate).

4. **Final Structure**:
   - For each language, the original and paraphrased texts were stacked vertically → **200 rows per language**.
   - This resulted in **4600 total rows** across 23 languages (English + 22 Indic).

## Embedding Models Used

The following models were used for semantic embedding and deduplication:

- [qwen2-7b]("https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct")  
- [sarvam]("https://huggingface.co/sarvamai/sarvam-m")  
- [intfloat-multilingual-e5-large-instruct]("https://huggingface.co/intfloat/multilingual-e5-large-instruct")  
- [Lajavaness/bilingual-embedding-large]("https://huggingface.co/Lajavaness/bilingual-embedding-large")  
- [solon-embeddings]("https://huggingface.co/OrdalieTech/Solon-embeddings-large-0.1")

## Deduplication & Evaluation

We applied a **semantic deduplication pipeline** using cosine similarity with varying **epsilon (ε) thresholds**: `0.025`, `0.05`, `0.1`.

### Definitions:
- **True Positive (TP)**: One sentence from a paraphrased pair is retained, the other is removed → correct deduplication.
- **False Negative (FN)**: Both paraphrased sentences are retained → deduplication failed.
- **False Positive**: Both paraphrased sentences are removed → over-deletion.

| Metric | Formula |
|--------|---------|
| **Recall** | TP / (TP + FN) |

## Results File

A single consolidated Excel file has been uploaded:

**`Indic Languages Semdedup Stats.xlsx`**

It contains the following **4 sheets**:

1. **Model-Wise (eps:0.1)**  
2. **Model-Wise (eps:0.05)**  
3. **Model-Wise (eps:0.025)**  
4. **Language-Wise** — grouped by language, showing results across all models

Each sheet includes per-language statistics: `TP`, `FN`, `FP`, `Recall`, and `Average of Recall values`.

---

## Threshold Sensitivity

> "The results are poor in some cases owing to the similarity threshold we use. As we used a threshold above 0.9, some duplicates were not found. This is likely because the embeddings could not achieve the required similarity score.  
The number of documents removed decreases as ε increases, indicating that lower thresholds capture more duplicates—but also risk more false positives."

---

## Conclusion

This analysis offers insights into the effectiveness of top multilingual embedding models in identifying semantic duplicates across diverse Indic languages. The `Recall` metric across languages and models provides a clear measure of performance and threshold sensitivity.

---

Feel free to raise an issue or contribute improvements.
