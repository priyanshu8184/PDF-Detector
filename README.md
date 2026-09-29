# 🔍 PDF Detective

A full-stack, production-grade, and clean modular RAG (Retrieval-Augmented Generation) application with an interactive **Streamlit** web app.

---

## 🚀 How to Run the App

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Configure API Key**:
   Create a `.env` file from `.env.example`:
   ```bash
   OPENROUTER_API_KEY=your_key_here
   # or
   OPENAI_API_KEY=your_key_here
   ```
   *(You can also paste the API key directly into the Streamlit sidebar).*

3. **Launch the Streamlit Web App**:
   ```bash
   streamlit run app.py
   ```

---

## 📁 Clean Modular Architecture

```
├── static/
│   └── style.css            # All CSS styles separated cleanly
├── modules/
│   ├── __init__.py          # Package initialization
│   ├── document_loader.py   # Document loading & extraction (.pdf, .txt)
│   ├── ui_utils.py          # UI helpers & CSS injection
│   ├── chunking.py          # Step 1: Text splitting (Fixed, Semantic, Recursive)
│   ├── embeddings.py        # Step 2: Vector embeddings & Cosine Similarity
│   ├── vector_store.py      # Step 3: FAISS indexing & nearest-neighbor search
│   ├── generator.py         # Step 4: Grounded prompt engineering & LLM generation
│   └── pipeline.py          # Step 5: Master orchestrator connecting all 4 steps
├── rag_pipeline.py          # Re-export bridge for backward-compatibility
├── app.py                   # Clean Streamlit presentation layer
├── requirements.txt         # Project dependencies
└── README.md                # Documentation & architecture reference
```

---

## 🎓 Running Individual Modules in Terminal

Har module ko independent demo ke roop me test kiya ja sakta hai:

```bash
# 1. Test Chunking Strategies
python modules/chunking.py

# 2. Test Vector Math & Cosine Similarity
python modules/embeddings.py

# 3. Test FAISS Vector Search
python modules/vector_store.py

# 4. Test Grounded Prompting
python modules/generator.py
```
