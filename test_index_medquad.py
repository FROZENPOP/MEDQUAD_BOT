"""Run with: python test_index_medquad.py"""
import tempfile
from pathlib import Path

from index_medquad import parse_qa_pairs

XML = """<Document><QAPairs>
<QAPair pid="1"><Question qid="1"> What is X? </Question><Answer> X is Y. </Answer></QAPair>
<QAPair pid="2"><Question qid="2">Answer removed?</Question><Answer/></QAPair>
<QAPair pid="3"><Question qid="3"/><Answer>Orphan answer.</Answer></QAPair>
</QAPairs></Document>"""


def test_parse_qa_pairs():
    with tempfile.TemporaryDirectory() as folder:
        # MedQuAD keeps its XML in one subfolder per source, so this must be found.
        nested = Path(folder, "1_CancerGov_QA")
        nested.mkdir()
        (nested / "a.xml").write_text(XML, encoding="utf-8")
        (nested / "notes.txt").write_text("not xml", encoding="utf-8")

        assert parse_qa_pairs(folder) == (["What is X?"], ["X is Y."])


if __name__ == "__main__":
    test_parse_qa_pairs()
    print("ok")
