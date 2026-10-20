#  Clinical Text Simplifier: LLM Fine-Tuning for Patient Health Literacy

**Course:** CSI-4130/5130: Artificial Intelligence — Course Project  
**Approach:** Model Fine-Tuning (Path 2)  
**Base Model:** `google/gemma-2-2b-it` or `mistralai/Mistral-7B-Instruct-v0.3`  
**Technique:** Parameter-Efficient Fine-Tuning (PEFT) via QLoRA (4-bit quantization)  
**Environment:** Google Colab (T4 / L4 GPU) & Google Drive Integration  

---

##  Project Overview
Dense medical records, clinical trial summaries, and hospital discharge instructions are frequently written at a graduate reading level, creating significant barriers to patient health literacy and informed care. 

This project fine-tunes an open-source Large Language Model (LLM) to translate complex clinical text into plain, compassionate, and accessible language for patients, while strictly preserving vital medical facts and clinical accuracy.

---

##  Google Colab Tech Stack & Frameworks

* **Compute Unit:** Google Colab (Free T4 or Colab Pro L4/A100 GPU)
* **Storage:** Google Drive mounted at `/content/drive` for persistent checkpoint saving
* **Fine-Tuning Paradigm:** Parameter-Efficient Fine-Tuning (PEFT) using **QLoRA (Quantized Low-Rank Adaptation)** to fit training within Colab GPU memory constraints.
* **Core Libraries:** 
  * `transformers` & `datasets` (Hugging Face ecosystem)
  * `bitsandbytes` (4-bit quantization support)
  * `peft` (LoRA configuration and adapter integration)
  * `trl` (`SFTTrainer` for instruction tuning)
  * `accelerate` & `torch`

---

##  Data Sources & Instruction Format

* **Dataset:** Curated instruction-tuning datasets from Hugging Face containing parallel corpora of technical medical text and patient-friendly lay translations (e.g., scientific lay-summarization benchmarks or medical simplification pairs).
* **Instruction Template:**
  ```json
  {
    "instruction": "Simplify the following clinical text for a non-technical patient.",
    "input": "[Complex medical text / discharge summary]",
    "output": "[Plain language patient-friendly summary]"
  }
