# SanskritRasaBank
> **The First Large-Scale, Expert-Validated Corpus for Computational Rasa Analysis in Sanskrit.**

[![License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## 📖 Overview
**SanskritRasaBank** is a state-of-the-art dataset and benchmarking suite designed to bridge the gap between classical Indian aesthetics (**Nava-Rasa**) and modern Natural Language Processing for low-resource languages. 

For over two millennia, the *Nāṭyaśāstra* framework has defined nine fundamental "essences" of human emotion (*rasas*). This project provides the first computational grounding for these rasas at scale, featuring **17,462 expert-verified verse annotations** drawn from eight classical sources including the *Vālmīki Rāmāyaṇa*, *Pañcatantra*, and *Rigveda*.

### The Nine Rasas (Nava-Rasa)
| Rasa | Meaning | Dominant Emotion |
|:---:|---|---|
| **Śṛṅgāra** | Love / Beauty | Rati (Love) |
| **Hāsya** | Laughter / Humor | Hāsa (Mirth) |
| **Karuṇā** | Grief / Compassion | Śoka (Sorrow) |
| **Raudra** | Fury / Anger | Krodha (Anger) |
| **Vīra** | Heroism / Valor | Utsāha (Enthusiasm) |
| **Bhayānaka** | Terror / Fear | Bhaya (Fear) |
| **Bībhatsa** | Disgust / Revulsion | Jugupsā (Disgust) |
| **Adbhuta** | Wonder / Amazement | Vismaya (Wonder) |
| **Śānta** | Serenity / Peace | Śama (Calmness) |

---

## 📊 Dataset Statistics
Our corpus was constructed using a **validated LLM-ensemble framework** (GPT-4o, DeepSeek-Chat, LLaMA-3.1) and audited by a team of Sanskrit Philologists from Tribhuvan University.

*   **Total Verified Samples:** 17,462
*   **Methodology:** 3-LLM Ensemble + Strict Consensus Rules + Expert Human-in-the-Loop Verification.
*   **Sources:** *Vālmīki Rāmāyaṇa* (All Kandas), *Kathāsaritsāgara*, *Pañcatantra*, *Amaruśataka*, *Vetālapañcaviṃśati*, *Abhijñānaśākuntalam*, *Rigveda* (Mandala 1).

### 🔬 Research Insights
- **Emotional Cartography of Rāmāyaṇa**: Ayodhyā Kāṇḍa is decisively **Karuṇā-dominant (33.3%)**, while Yuddha Kāṇḍa reaches a martial peak of **39% Vīra**.
- **Vedic Aesthetics**: Maṇḍala 1 of the *Rigveda* is uniquely characterized by **Adbhuta (Wonder) at 23.78%**, capturing the hymnal aesthetic of divine address. Our model supports **multi-label inference**, identifying the three most prominent rasas for each verse to capture emotional complexity.

---

## 🚀 Model Benchmarks (SOTA)
We benchmarked classical ML (SVM) against specialized transformer architectures. Following targeted hyperparameter optimization (**Phase 2**), **IndicBERT V2** and **MuRIL** achieved state-of-the-art performance.

| Model | Accuracy (%) | Weighted F1 | Macro F1 |
|:---|:---:|:---:|:---:|
| **IndicBERT V2 (Phase 2)** ⭐ | **81.45** | **81.49** | **76.73** |
| **MuRIL-large (Phase 2)** | 80.65 | 80.67 | 76.51 |
| XLM-RoBERTa-large (Phase 2) | 78.87 | 78.83 | 73.56 |
| MuRIL (Phase 1 Baseline) | 80.48 | 80.60 | 75.69 |
| SVM (Character N-gram) | 53.60 | 51.50 | 47.90 |

> **Note on Multi-Output Inference**: While the primary label (most prominent rasa) is the most accurate, the model also predicts second and third most prominent rasas, providing a granular look at the affective transitions in classical Sanskrit texts.

---

## 📂 Project Structure
```bash
SanskritRasaBank/
├── data/
│   ├── raw/            # Initial source files
│   ├── verified/       # Main gold-standard dataset (MERGED_FINAL.xlsx)
│   └── inference/      # Large-scale inference results (Rāmāyaṇa, Rigveda)
├── scripts/
│   ├── training/       # Fine-tuning scripts for Transformers
│   ├── evaluation/     # Metrics and literary analysis scripts
│   ├── annotation/     # LLM-Ensemble pipeline and consensus logic
│   └── data_processing/# Cleaning and dataset construction scripts
├── experiments/        # Folders containing logs and reports for each model run
├── results/            # Visual insights and summary reports
└── models/             # Saved model artifacts (architecture specific)
```

---

## 🛠️ Quick Start

### Installation
```bash
git clone https://github.com/suhritghimire/SanskritRasaBank.git
cd SanskritRasaBank
pip install -r requirements.txt
```

### Citation
If you use this project or dataset, please cite our work:

```bibtex
@article{ghimire2025sanskritrasabank,
  title={Tasting the Poem: Benchmarking Multi-Label Rasa Classification with SanskritRasaBank},
  author={Ghimire, Suhrit and Timilsina, Rohini Raj and Jain, Minni},
  journal={Language Resources and Evaluation (Springer)},
  year={2025},
  note={Submitted / Under Review}
}
```

---

## 🤝 Acknowledgments
- **Project Lead:** Suhrit Ghimire (Delhi Technological University)
- **Expert Validation:** Rohini Raj Timilsina (Lecturer, Sanskrit Dept, Tribhuvan University)
- **Mentorship:** Dr. Minni Jain (Assistant Professor, DTU)

---
© 2025 SanskritRasaBank Team. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) & [MIT](LICENSE).
