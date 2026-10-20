# 🩺 Clinical Text Simplifier: LLM Fine-Tuning for Patient Health Literacy

**Course:** CSI-4130/5130: Artificial Intelligence — Course Project  
**Approach:** Model Fine-Tuning (Path 2)  
**Base Model:** Gemma-2-2B-Instruct / Mistral-7B-Instruct  
**Technique:** Parameter-Efficient Fine-Tuning (PEFT) via QLoRA  

---

## 📌 Overview
Dense medical records, clinical trial summaries, and hospital discharge instructions are frequently written at a graduate reading level, creating significant barriers to patient health literacy and informed care. 

This project fine-tunes an open-source Large Language Model (LLM) to translate complex clinical text into plain, compassionate, and accessible language for patients, while strictly preserving vital medical facts and clinical accuracy.

---

## ⚙️ Technical Approach & Methodology

* **Base Model:** Pre-trained open-source instruction models (e.g., Gemma-2-2B or Mistral-7B-Instruct).
* **Fine-Tuning Paradigm:** Parameter-Efficient Fine-Tuning (PEFT) using **QLoRA (Quantized Low-Rank Adaptation)** with 4-bit quantization (`bitsandbytes`) to enable efficient training on standard cloud resources (e.g., Google Colab T4/L4 GPUs).
* **Frameworks & Libraries:** 
  * Hugging Face `transformers`, `peft`, `trl` (Transformer Reinforcement Learning / SFTTrainer)
  * PyTorch
  * `datasets` and `accelerate`

---

## 📊 Data Sources & Preprocessing
* **Dataset:** Curated instruction-tuning datasets from Hugging Face containing parallel corpora of technical medical text and patient-friendly lay translations (e.g., scientific lay-summarization benchmarks or medical simplification pairs).
* **Format:** Data is structured into standard instruction-response templates:
  ```json
  {
    "instruction": "Simplify the following clinical text for a non-technical patient.",
    "input": "[Complex medical text / discharge summary]",
    "output": "[Plain language patient-friendly summary]"
  }
