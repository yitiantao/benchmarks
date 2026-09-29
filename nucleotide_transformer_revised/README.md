---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{}
---

# Dataset Card for Dataset Name
The `nucleotide_transformer_downstream_tasks` dataset features the 18 downstream tasks presented in the [Nucleotide Transformer paper](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v3). They consist of both binary and multi-class classification tasks that aim at providing a consistent genomics benchmark.

**We note that this is an updated version of this benchmark after the paper has been through peer-review. We highly encourage to move to this version in detriment of the [older version](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks).**  
Keypoints about the updated datasets: 
  1. It maintains similar tasks, number of classes and train and test set sizes as the previous benchmark.
  2. All the data is now from human samples and from recent and high-quality datasets (see descriptions of each dataset source below). 
  3. Randomly generated negative samples have been replaced by existing chunks of genomes not containing the respective elements.
  4. All datasets have now proper chromosome held-out test sets.

## Dataset Description

- **Repository:** [Nucleotide Transformer](https://github.com/instadeepai/nucleotide-transformer)
- **Paper:** [The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v3) 

### Dataset Summary

The different datasets are collected from the following resources: 
- [ENCODE](https://www.encodeproject.org/): Histone ChIP-seq data for 10 histone marks in the K562 human cell line were obtained from ENCODE. We downloaded bed narrowPeak files with the following identifiers: `H3K4me3` (ENCFF706WUF), `H3K27ac` (ENCFF544LXB), `H3K27me3` (ENCFF323WOT), `H3K4me1` (ENCFF135ZLM), `H3K36me3` (ENCFF561OUZ), `H3K9me3` (ENCFF963GZJ), `H3K9ac` (ENCFF891CHI), `H3K4me2` (ENCFF749KLQ), `H4K20me1` (ENCFF909RKY), `H2AFZ` (ENCFF213OTI). For each dataset, we selected 1kb genomic sequences containing peaks as positive examples and all 1kb sequences not overlapping peaks as negative examples.
- [Screen](https://screen.wenglab.org/): Human enhancer elements were retrieved from ENCODE's SCREEN database. Distal and proximal enhancers were combined. Enhancers were split in tissue-specific and tissue-invariant based on the vocabulary from [Meuleman et al.](https://www.meuleman.org/research/dhsindex/). Enhancers overlapping regions classified as tissue-invariant were defined as that, while all other enhancers were defined as tissue-specific. We selected 400bp genomic sequences containing enhancers as positive examples and all 400bp sequences not overlapping enhancers as negative examples. We created a binary classification task for the presence of enhancer elements in the sequence (`Enhancer`) and a multi-label prediction task with labels being tissue-specific enhancer, tissue-invariant enhancer or none (`Enhancer types`).
- [Eukaryotic Promoter Database](https://epd.expasy.org/epd/): We downloaded all human promoters from the Eukaryotic Promoter Database, spanning 49bp upstream and 10bp downstream of transcription start sites ([file](https://epd.expasy.org/ftp/epdnew/H_sapiens/006/Hs_EPDnew_006_hg38.bed)). This resulted in 29,598 promoter regions, 3,065 of which were TATA-box promoters (using the motif annotation at [https://epd.expasy.org/ftp/epdnew/H_sapiens/006/db/promoter_motifs.txt](https://epd.expasy.org/ftp/epdnew/H_sapiens/006/db/promoter_motifs.txt)). We selected 300bp genomic sequences containing promoters as positive examples and all 300bp sequences not overlapping promoters as negative examples. These positive and negative examples were used to create three different binary classification tasks: presence of any promoter element (`Promoter all`), a promoter with a TATA-box motif (`Promoter TATA`) or a promoter without a TATA-box motif (`Promoter no-TATA`).
- [GENCODE](https://www.gencodegenes.org/human/release_44.html): We obtained all human annotated splice sites from GENCODE V44 gene annotation. Annotations were filtered to exclude level 3 transcripts (automated annotation), so all training data was annotated by a human.

## Dataset Structure
```
| Task                  | Number of train sequences | Number of test sequences | Number of labels | Sequence length |
| --------------------- | ------------------------- | ------------------------ | ---------------- | --------------- |
| promoter_all          | 30,000                    | 1,584                    | 2                | 300             |
| promoter_tata         | 5,062                     | 212                      | 2                | 300             |
| promoter_no_tata      | 30,000                    | 1,372                    | 2                | 300             |
| enhancers             | 30,000                    | 3,000                    | 2                | 400             |
| enhancers_types       | 30,000                    | 3,000                    | 3                | 400             |
| splice_sites_all      | 30,000                    | 3,000                    | 3                | 600             |
| splice_sites_acceptor | 30,000                    | 3,000                    | 2                | 600             |
| splice_sites_donor    | 30,000                    | 3,000                    | 2                | 600             |
| H2AFZ                 | 30,000                    | 3,000                    | 2                | 1,000           |
| H3K27ac               | 30,000                    | 1,616                    | 2                | 1,000           |
| H3K27me3              | 30,000                    | 3,000                    | 2                | 1,000           |
| H3K36me3              | 30,000                    | 3,000                    | 2                | 1,000           |
| H3K4me1               | 30,000                    | 3,000                    | 2                | 1,000           |
| H3K4me2               | 30,000                    | 2,138                    | 2                | 1,000           |
| H3K4me3               | 30,000                    | 776                      | 2                | 1,000           |
| H3K9ac                | 23,274                    | 1,004                    | 2                | 1,000           |
| H3K9me3               | 27,438                    | 850                      | 2                | 1,000           |
| H4K20me1              | 30,000                    | 2,270                    | 2                | 1,000           |
```

## How to use it
```python3
from datasets import load_dataset

# Import the dataset
dataset = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")
# It loads all the tasks
print(dataset)

# You can filter for a given task, if needed
enhancers_dataset = dataset.filter(lambda example: example["task"] == "enhancers")
print(enhancers_dataset)
```


