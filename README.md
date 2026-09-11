
## Medical QA Bot using FAISS + Sentence Transformers

This project builds a question-answering bot using a custom medical dataset (MedQuAD), FAISS for similarity search, and Sentence Transformers for semantic embeddings. The goal is to allow users to ask common medical questions and receive accurate, relevant answers from a curated set of FAQs.

---

## 🚀 Features

- Indexed medical Q&A dataset (MedQuAD)
- Semantic search with FAISS
- Sentence-transformer embeddings for better similarity
- Easily extensible with new Q&A data
- Python-based script to update the index
- Streamlit interface (optional)

---

## 🛠️ Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/medical-qa-bot.git
   cd medical-qa-bot
   ```

2. **Install dependencies**  
   Ensure you have Python 3.8 or higher.
   ```bash
   pip install -r requirements.txt
   ```

3. **📥 Download MedQuAD dataset**  
   Choose one of the options below to obtain the dataset:

   **Option A – GitHub (Recommended)**  
   Clone or download from the official repo:  
   🔗 [https://github.com/abachaa/MedQuAD](https://github.com/abachaa/MedQuAD)  
   ```bash
   git clone https://github.com/abachaa/MedQuAD.git data/
   ```

   **Option B – Hugging Face**  
   Load directly via the Hugging Face datasets API:  
   ```python
   from datasets import load_dataset
   dataset = load_dataset("lavita/MedQuAD")
   ```

   Make sure the downloaded data is placed in a folder named `data/` at the project root:
   ```
   medical-qa-bot/
   └── data/
       └── [XML files or parsed data here]
   ```

---

## 📦 Usage

### 1. **Build or Rebuild the FAISS Index**
Run the indexing script to parse MedQuAD and build the FAISS index.
```bash
python build_faiss_index.py
```

### 2. **Ask a Question**
Run the main interface script (e.g., console or Streamlit).
```bash
python qa_interface.py
```

---

## 🧩 Adding More Data to the Index

To extend your knowledge base:

1. Add new XML files or structured Q&A to `data/`.
2. Make sure the format matches the MedQuAD XML structure.
3. Re-run:
   ```bash
   python build_faiss_index.py
   ```

---

## 📁 Project Structure

```
medical-qa-bot/
├── data/                    # Raw MedQuAD XML files
├── faiss_index/            # Stored FAISS index + embeddings
├── build_faiss_index.py    # Builds the FAISS index
├── qa_interface.py         # User interface for QA
├── utils.py                # Utility functions (parsing, embedding, etc.)
├── requirements.txt        # Python dependencies
└── README.md               # Project instructions
```

---

## 📌 Dependencies

- `faiss-cpu`
- `sentence-transformers`
- `streamlit`
- `groq`
- `dotenv`
  
Install manually if needed:
```bash
pip install faiss-cpu sentence-transformers streamlit groq dotenv
```

---

## 📖 Credits
