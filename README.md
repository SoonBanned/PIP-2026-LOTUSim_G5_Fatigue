# 🚢 LLaMA 3.1 LotuSim – COLREGs Quiz Model

This repository contains a fine-tuned **LLaMA 3.1** model designed for **COLREGs (International Regulations for Preventing Collisions at Sea)** quiz-style interactions.

---

## 🔗 Model

The model is hosted on Hugging Face:

👉 [https://huggingface.co/SoonBanned/LLaMa_3.1_LotuSim_quiz_Colregs](https://huggingface.co/SoonBanned/LLaMa_3.1_LotuSim_quiz_Colregs)

---

## 🧠 Usage with Ollama

To use this model within the project, you must import it into **Ollama** under the name:

```
llama3.1_fine
```

---

## 📦 Setup Instructions

A ready-to-use **ModelFile** is included in this repository.

1. Ensure **Ollama** is installed on your system.
2. Copy the provided `.gguf` model file into the `Finetune/` directory.
3. From the **root of the project**, run:

```bash
ollama create llama3.1_fine -f ModelFile
```

Once completed, the model will be available in Ollama as `llama3.1_fine`.

---

## ✅ Notes

* Ollama installation is required.
* Make sure the model file path in `ModelFile` matches the location of your `.gguf` file.

Happy experimenting! ⚓🤖
