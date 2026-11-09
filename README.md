# MedQuAD Bot

A Streamlit chatbot that answers medical questions using retrieval from the [MedQuAD](https://github.com/abachaa/MedQuAD) dataset, a collection of question-answer pairs from NIH websites. It finds the closest questions in a local FAISS index, then asks a model hosted on Groq to write an answer from them.

> [!WARNING]
> This is a learning project and is not medical advice. Answers can be wrong or incomplete, so talk to a medical professional about any health decision. In an emergency, contact your local emergency services.

## How it works

```mermaid
flowchart LR
    Q[Your question] --> E[Embed locally<br>all-MiniLM-L6-v2]
    E --> F[Search FAISS index<br>15,426 Q&A pairs]
    F --> P[Closest Q&A pairs<br>plus your question]
    P --> G[Model on Groq]
    G --> A[Answer]
```

The search compares your question with the dataset's questions, not its answers. The app always passes the closest matches to the model, however weak they are. If they don't cover your question, the model is told to say so and then give a general explanation that doesn't come from the dataset.

## Getting started

You need Python 3.13 and a [Groq API key](https://console.groq.com/keys). Python 3.14 won't work because the pinned `faiss-cpu` has no build for it.

```bash
git clone https://github.com/FROZENPOP/MEDQUAD_BOT.git
cd MEDQUAD_BOT
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS and Linux. Then install the dependencies and create your config file:

```bash
pip install -r requirements.txt
cp .env.example .env
```

On Windows Command Prompt, use `copy` instead of `cp`. Add your key to `.env`, then start the app from the repo root:

```bash
streamlit run app.py
```

The first run downloads the embedding model from Hugging Face.

## Configuration

These go in `.env`:

| Variable | Required | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | Yes | Your Groq API key. Never commit it. |
| `GROQ_MODEL` | No | The Groq model to use. Defaults to `openai/gpt-oss-120b`. Change it if Groq [retires that model](https://console.groq.com/docs/deprecations). |

The sidebar controls temperature, maximum response length, detailed or general response style, and how many Q&A pairs to retrieve (Top K). It can also show the retrieved source text under each answer. You can download the conversation as a text file below the chat.

## Rebuilding the index

The repo includes a prebuilt index, so you only need this to change the data. Run it from the repo root:

```bash
git clone https://github.com/abachaa/MedQuAD.git ../MedQuAD
python index_medquad.py ../MedQuAD
```

This overwrites `medquad_index.faiss` and `medquad_answers.pkl`. To add your own data, use XML files with the same `QAPair`, `Question` and `Answer` structure and point the script at their folder.

A rebuild produces 16,407 pairs rather than the prebuilt 15,426, because the prebuilt index leaves out the 981 pairs in `4_MPlus_Health_Topics_QA`. Three other MedQuAD sources contribute nothing, because the dataset's authors removed their answers for copyright reasons.

## Testing

```bash
python test_index_medquad.py
```

This checks the XML parser. There are no automated tests for the Streamlit app or for answer quality.

## Limitations

- **Privacy:** your questions and the retrieved text are sent to Groq, so don't enter personal health information.
- **No conversation memory:** each question is answered on its own, so follow-ups like "what about in children?" lose their context.
- **No relevance cutoff:** weak matches still reach the model, which decides for itself whether they're relevant.
- **No accuracy evaluation:** answer quality hasn't been measured.
- **Dated content:** the data is a snapshot from when MedQuAD was published and may not reflect current guidance.

## Project structure

```
├── app.py                  # Streamlit app: retrieval, prompt and chat UI
├── index_medquad.py        # Builds the index from MedQuAD XML files
├── test_index_medquad.py   # Test for the XML parser
├── medquad_index.faiss     # Prebuilt vector index
├── medquad_answers.pkl     # Questions and answers matching the index
├── requirements.txt
└── .env.example            # Template for your .env file
```

## Credits and license

MedQuAD is by Asma Ben Abacha and Dina Demner-Fushman and is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The included index and answers are derived from it. The paper is [A Question-Entailment Approach to Question Answering](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4), BMC Bioinformatics, 2019.

The code in this repo doesn't have a license yet.
