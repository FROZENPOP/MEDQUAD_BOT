"""Build the FAISS index and answer metadata from the MedQuAD XML files.

Usage: python index_medquad.py path/to/MedQuAD
"""
import argparse
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_qa_pairs(xml_folder):
    """Return (questions, answers) for every non-empty QAPair under xml_folder."""
    questions, answers = [], []
    # Sorted so a rebuild of the same data gives the same index order.
    for path in sorted(Path(xml_folder).rglob("*.xml")):
        for qa in ET.parse(path).getroot().findall(".//QAPair"):
            question = (qa.findtext("Question") or "").strip()
            answer = (qa.findtext("Answer") or "").strip()
            if question and answer:
                questions.append(question)
                answers.append(answer)
    return questions, answers


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("xml_folder", help="folder containing the MedQuAD XML files, searched recursively")
    args = parser.parse_args()

    questions, answers = parse_qa_pairs(args.xml_folder)
    # Stop here so a wrong path can't overwrite a working index.
    if not questions:
        raise SystemExit(f"No Q&A pairs found under {args.xml_folder}")
    print(f"Parsed {len(questions)} valid Q&A pairs.")

    # Imported here because these are slow to load and only needed for a real build.
    import faiss
    from sentence_transformers import SentenceTransformer

    embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(questions, show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "medquad_index.faiss")
    with open("medquad_answers.pkl", "wb") as f:
        pickle.dump({"questions": questions, "answers": answers}, f)

    print("FAISS index and answer metadata saved.")


if __name__ == "__main__":
    main()
