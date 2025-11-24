---
tags:
- medical
- dermatology
- vision-language
- clip
- multimodal
- concept-based explanation
- skin-disease
size_categories:
- 100K<n<1M
license: cc-by-nc-4.0
extra_gated_prompt: >-
  I understand that the Derm1M dataset is released under the Creative Commons
  Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. I
  acknowledge that this dataset is intended for non-commercial research purposes
  only. I agree to comply with the licensing terms and understand that
  commercial use requires separate permission from the dataset creators. I
  further agree to use this dataset responsibly and ethically for advancing
  dermatological research and medical AI development.
extra_gated_fields:
  I confirm that I have read and agree to the data usage agreement outlined above by checking this box: checkbox
  I want to use this dataset for: text
  Affiliation: text
  Research purpose: text
language:
- en

configs:
- config_name: default
  data_files:
  - split: train
    path: "Derm1M_v2_pretrain.csv"
  - split: valid
    path: "Derm1M_v2_validation.csv"
---

# Dataset Card for Derm1M

<div align="center">
    <img src="https://raw.githubusercontent.com/SiyuanYan1/Derm1M/main/assets/ICCV_Derm1M_poster.png" 
     alt="Derm1M Overview" 
     width="800" />
</div>

<p align="center">
  <strong>Paper:</strong> <a href="https://arxiv.org/abs/2503.14911" target="_blank">ArXiv</a> 
  &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Code:</strong> <a href="https://github.com/SiyuanYan1/Derm1M" target="_blank">GitHub</a>
  &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Models:</strong> <a href="https://huggingface.co/redlessone/DermLIP_ViT-B-16" target="_blank">DermLIP-ViT-B-16</a> | <a href="https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256" target="_blank">DermLIP-PanDerm</a>
</p>

## Dataset Summary

**Derm1M** is a large-scale, million-scale vision-language dataset for dermatology containing **1,029,761 dermatological image-text pairs** from **403,563 unique images**. The dataset covers **390 skin conditions** organized in a four-level expert ontology and includes **130 clinical concepts**. With rich contextual captions averaging 41 tokens, Derm1M enables explainable multimodal learning, zero-shot and few-shot diagnosis, cross-modal retrieval, and visual question answering in clinical dermatology settings.

This dataset is **257× larger** than any previous dermatology vision-language corpus and is specifically designed for training and evaluating vision-language models in the dermatology domain.

## Dataset Details
Derm1M provides comprehensive annotations including:
- **1,029,761 image-text pairs** with detailed clinical captions
- **390 skin conditions** structured in a hierarchical ontology
- **130 clinical concepts** extracted per image
- **Rich metadata** including image sources, clinical contexts, and ontological relationships
- **Structured ontology** in JSON format for hierarchical disease understanding

### Dataset Description
- **Curated by:** Siyuan Yan, Ming Hu, Yiwen Jiang, Xieji Li
- **Language(s):** English
- **License:** CC BY-NC 4.0 (Non-Commercial Use Only)
- **Supported Tasks:**
  - Vision-language pre-training
  - Zero-shot classification
  - Few-shot learning
  - Cross-modal retrieval
  - Concept annotation/explanation
  - Visual question answering

### Dataset Sources

- **Repository:** https://github.com/SiyuanYan1/Derm1M
- **Paper:** https://arxiv.org/abs/2503.14911
- **Models:** 
  - [DermLIP-ViT-B-16](https://huggingface.co/redlessone/DermLIP_ViT-B-16)
  - [DermLIP-PanDerm-base-w-PubMed-256](https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256)

## Dataset Structure
```
dataset_root/
├── xxx/                   # unzip all zip files
├── Derm1M_v2_pretrain.csv    # text + meta per image for model pretraining
├── Derm1M_v2_validation.csv  # text + meta per image for model validation
├── concept.csv               # extracted concept annotations per image
├── ontology.json             # skin disease hierarchy
```
### Data Instances
```python
{
  'filename': 'image_001.jpg',
  'truncated_caption': 'Clinical photograph showing erythematous papules and pustules on facial skin, consistent with inflammatory acne...',
  'disease_label': 'Acne Vulgaris',
  'hierarchical_disease_label': 'Inflammatory Skin Diseases, Acne and Related Disorders, Acne Vulgaris'
  'skin_concept': 'erythema, papule, pustule, facial_distribution',
  'source': 'pubmed',
  'source_type': 'knowledge',
  .......
}
```
## Citation

```
@misc{yan2025derm1m,
  title        = {Derm1M: A Million‑Scale Vision‑Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology},
  author       = {Siyuan Yan and Ming Hu and Yiwen Jiang and Xieji Li and Hao Fei and Philipp Tschandl and Harald Kittler and Zongyuan Ge},
  year         = {2025},
  eprint       = {2503.14911},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  url          = {https://arxiv.org/abs/2503.14911}
}

@article{yan2025multimodal,
  title={A multimodal vision foundation model for clinical dermatology},
  author={Yan, Siyuan and Yu, Zhen and Primiero, Clare and Vico-Alonso, Cristina and Wang, Zhonghua and Yang, Litao and Tschandl, Philipp and Hu, Ming and Ju, Lie and Tan, Gin and others},
  journal={Nature Medicine},
  pages={1--12},
  year={2025},
  publisher={Nature Publishing Group}
}
```