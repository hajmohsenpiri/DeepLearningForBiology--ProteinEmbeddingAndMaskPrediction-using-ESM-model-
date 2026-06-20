# DeepLearningForBiology--ProteinEmbeddingAndMaskPrediction-using-ESM-model-
A hands-on exploration of Meta AI's ESM-2 protein language model, covering tokenization, masked amino acid prediction, and embedding-based analysis of protein function.

Developed while working through the "Proteins" chapter of *Deep Learning for Biology*
(Hofmann, Bates & Berman), as part of self-directed study connecting NLP-style
representation learning to structural and functional biology.

## What's inside

- **3D structure visualization** — fetching PDB entries (insulin, collagen, proteasome)
  from RCSB and rendering them with py3Dmol
- **From-scratch one-hot encoding** — building an amino acid vocabulary and encoding
  sequences manually with JAX, as a baseline before using pretrained representations
- **ESM-2 tokenization and embeddings** — loading `facebook/esm2_t33_650M_UR50D`,
  inspecting the tokenizer's vocabulary, and extracting the model's input embedding matrix
- **Masked language modeling on protein sequences** — masking a residue in the human
  insulin sequence and using `EsmForMaskedLM` to predict the masked amino acid,
  visualizing the model's softmax distribution over the vocabulary
- **`MaskPredictor`** — a small reusable class that wraps masking + inference so any
  position in any sequence can be probed
- **Functional embedding analysis** — using a GO-annotated UniProt subset to compare
  mean-pooled ESM-2 embeddings for extracellular vs. membrane proteins, visualized
  with t-SNE to see how subcellular localization is reflected in embedding space

## Stack

PyTorch · JAX · HuggingFace Transformers (`EsmModel`, `EsmForMaskedLM`) · py3Dmol ·
scikit-learn (t-SNE) · pandas

## Background

This notebook builds toward broader interests in structure-based modeling of
protein–protein and TCR–pMHC interactions, where understanding what pretrained
sequence representations do and don't capture about structure and function is a
useful foundation.

## Credit

Based on material from the [Deep Learning for Biology](https://github.com/deep-learning-for-biology/dlfb)
book repository. This notebook is a personal annotated walkthrough/exploration of
that chapter's content, not original coursework.
