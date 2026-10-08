"""
Measure how well document search finds the passage that answers a question.

    python evaluate_retrieval.py run examples/eval/questions.json
    python evaluate_retrieval.py run test_docs/questions.json --chunk-size 150 --chunk-overlap 20
    python evaluate_retrieval.py draft test_docs --count 20

`run` indexes every supported file in the questions file's folder into a
temporary index (your chroma_db/ isn't touched), asks each question, and
reports how often the right passage comes back, and how high it ranks. The
other documents in the folder act as distractors.

`draft` has your model write questions about your own documents, so you
don't have to start from scratch. Review them (fix, delete, add your own)
before renaming the draft to questions.json.

A questions file is JSON, next to the documents it asks about:

    {"questions": [
      {"question": "When do the backups run?",
       "document": "home-server-handbook.pdf",
       "quote": "Nightly backups run at 02:00 with restic"}
    ]}

"quote" is text copied from the document that answers the question; case,
punctuation and line breaks don't matter. A search result counts as right when
it's from that document and holds most of the quote.

Questions the documents can't answer can be listed too, to see whether the
similarity threshold keeps unrelated excerpts away from them:

    {"questions": [...],
     "unanswerable": ["What's the capital of Peru?"]}
"""

import argparse
import contextlib
import io
import json
import os
import random
import re
import sys
import tempfile
import unicodedata
from typing import Dict, List, Optional, Tuple

from config import DEFAULT_MODEL, GENERATION_CONFIG, RAG_CONFIG, SUPPORTED_EXTENSIONS
from document_processor import DocumentProcessor
from rag_pipeline import source_label

DEPTH = 10  # How many results are ranked for each question
THRESHOLDS = [round(0.05 * i, 2) for i in range(11)]  # Similarity thresholds compared in the report

DRAFT_PROMPT = """Below is an excerpt from a document called {document}.

---
{excerpt}
---

Write one question that someone who hasn't read the excerpt might ask, which this excerpt answers. Use your own words rather than the excerpt's. Then copy, word for word, the sentence from the excerpt that answers it.

Reply with only this JSON: {{"question": "...", "quote": "..."}}"""

DRAFT_CONFIG = {**GENERATION_CONFIG, "max_new_tokens": 200, "temperature": 0.3}


def normalize(text: str) -> str:
    """
    Reduce text to lowercase words separated by single spaces, so a quote
    matches however the document breaks lines or punctuates (and "off-\\nsite"
    in a PDF still matches "off-site").
    """
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).lower()))


class Document:
    """A processed document: its chunks and where each sits in the normalized text."""

    def __init__(self, filename: str, text: str, chunks: List[Dict]):
        self.filename = filename
        self.text = normalize(text)
        self.chunks = chunks
        self.spans = []
        position = 0
        for chunk in chunks:
            words = normalize(chunk["text"])
            start = self.text.find(words, position)
            if start < 0:  # shouldn't happen: chunks are pieces of the text in order
                start = self.text.find(words)
            position = max(start, position)
            self.spans.append((start, start + len(words)) if start >= 0 else None)

    def find(self, quote: str) -> Optional[Tuple[int, int]]:
        """Where the quote is in the normalized text, or None."""
        words = normalize(quote)
        start = self.text.find(words) if words else -1
        return (start, start + len(words)) if start >= 0 else None

    def chunks_holding(self, span: Tuple[int, int]) -> List[int]:
        """
        The chunks that count as holding the text at span: each one with at
        least half of it (or that's at least half made of it), and whichever
        holds the most of it, so a quote split between chunks still has one.
        """
        overlaps = [0 if chunk is None else max(0, min(chunk[1], span[1]) - max(chunk[0], span[0]))
                    for chunk in self.spans]
        most = max(overlaps, default=0)
        return [i for i, (chunk, overlap) in enumerate(zip(self.spans, overlaps))
                if overlap > 0 and (overlap == most or overlap >= min(span[1] - span[0], chunk[1] - chunk[0]) / 2)]


def load_documents(folder: str, processor: DocumentProcessor) -> Dict[str, Document]:
    """Process every supported file directly in folder (hidden files aside)."""
    documents = {}
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if name.startswith(".") or not os.path.isfile(path) or not processor.is_supported(name):
            continue
        result = processor.process_file(path)
        documents[name] = Document(name, result["text"], result["chunks"])
    return documents


def evaluate(questions_path: str, processor: DocumentProcessor = None, embedding_model=None,
             embedding_model_name: str = None, top_k: int = None, min_similarity: float = None) -> Dict:
    """
    Index the documents next to a questions file, ask each question and rank the results.

    Args:
        questions_path: Path to the questions file
        processor: Chunking settings to test (default: the app's)
        embedding_model: Preloaded embedding model (tests use a stand-in)
        embedding_model_name: sentence-transformers model to load instead of the app's
        top_k: Results the app uses (default from config)
        min_similarity: Similarity the app requires (default from config)

    Returns:
        Report dictionary (see format_report)
    """
    from vector_store_manager import VectorStoreManager

    processor = processor or DocumentProcessor()
    top_k = top_k or RAG_CONFIG["top_k"]
    min_similarity = RAG_CONFIG["min_similarity"] if min_similarity is None else min_similarity
    with open(questions_path, encoding="utf-8") as f:
        questions_file = json.load(f)
    questions = questions_file.get("questions", [])
    documents = load_documents(os.path.dirname(os.path.abspath(questions_path)), processor)

    results, problems = [], []
    # ignore_cleanup_errors: on Windows, ChromaDB can still hold its files open
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as index_dir:
        print("Indexing documents...", file=sys.stderr)
        with contextlib.redirect_stdout(io.StringIO()):  # the store's progress messages
            store = VectorStoreManager(persist_directory=index_dir, embedding_model=embedding_model,
                                       embedding_model_name=embedding_model_name)
            for document in documents.values():
                store.add_documents(document.chunks)
        model_name = store.embedding_model_name

        for item in questions:
            question, filename = item.get("question", ""), item.get("document", "")
            document = documents.get(filename)
            if document is None:
                problems.append(f"“{question}”: no document called {filename} in the folder")
                continue
            span = document.find(item.get("quote", ""))
            if span is None:
                problems.append(f"“{question}”: the quote isn't in {filename}")
                continue

            with contextlib.redirect_stdout(io.StringIO()):
                hits = store.search(question, top_k=DEPTH, min_similarity=-1.0)
            right = document.chunks_holding(span)
            rank = similarity = None
            for i, hit in enumerate(hits, 1):
                if hit["metadata"]["filename"] == filename and hit["metadata"]["chunk_index"] in right:
                    rank, similarity = i, hit["similarity"]
                    break
            results.append({
                "question": question,
                "expected": source_label(document.chunks[right[0]]),
                "rank": rank,
                "similarity": similarity,
                "found": rank is not None and rank <= top_k and similarity >= min_similarity,
                "retrieved": [source_label(h["metadata"]) for h in hits[:3]],
            })

        unanswerable = []
        for question in questions_file.get("unanswerable", []):
            with contextlib.redirect_stdout(io.StringIO()):
                hits = store.search(question, top_k=top_k, min_similarity=-1.0)
            best = max((h["similarity"] for h in hits), default=None)
            unanswerable.append({
                "question": question,
                "best_similarity": best,
                "given_excerpts": best is not None and best >= min_similarity,
                "retrieved": [source_label(h["metadata"]) for h in hits[:3]],
            })

    return {
        "questions_file": questions_path,
        "documents": len(documents),
        "chunks": sum(len(d.chunks) for d in documents.values()),
        "chunk_config": processor.chunk_config,
        "embedding_model": model_name,
        "top_k": top_k,
        "min_similarity": min_similarity,
        "results": results,
        "unanswerable": unanswerable,
        "problems": problems,
        "summary": summarize(results, top_k, min_similarity),
        "thresholds": compare_thresholds(results, unanswerable, top_k),
    }


def summarize(results: List[Dict], top_k: int, min_similarity: float) -> Dict:
    """Headline numbers: how often the right passage is found, and how high it ranks."""
    ranks = [r["rank"] for r in results]

    def within(n):
        return sum(1 for rank in ranks if rank is not None and rank <= n)

    return {
        "questions": len(results),
        "found": sum(1 for r in results if r["found"]),
        "top_1": within(1),
        "top_3": within(3),
        "top_k": within(top_k),
        "top_10": within(DEPTH),
        "below_threshold": sum(1 for r in results
                               if r["rank"] is not None and r["rank"] <= top_k and r["similarity"] < min_similarity),
        "mrr": sum(1 / rank for rank in ranks if rank is not None) / len(results) if results else 0.0,
    }


def compare_thresholds(results: List[Dict], unanswerable: List[Dict], top_k: int) -> List[Dict]:
    """
    For each threshold in THRESHOLDS: how many questions the app would answer
    from the right passage, and how many unanswerable ones it would still give
    excerpts to. Raising the threshold lowers both.
    """
    return [{
        "threshold": threshold,
        "found": sum(1 for r in results
                     if r["rank"] is not None and r["rank"] <= top_k and r["similarity"] >= threshold),
        "unanswerable_given_excerpts": sum(1 for u in unanswerable
                                           if u["best_similarity"] is not None and u["best_similarity"] >= threshold),
    } for threshold in THRESHOLDS]


def format_report(report: Dict) -> str:
    """The report as Markdown (readable in a terminal too)."""
    s, n = report["summary"], report["summary"]["questions"]
    top_k, threshold = report["top_k"], report["min_similarity"]

    def share(count):
        return f"{count}/{n} ({count / n:.0%})" if n else "0/0"

    lines = [
        f"## Retrieval evaluation: {report['questions_file']}",
        "",
        f"{n} questions · {report['documents']} documents, {report['chunks']} chunks · "
        f"chunks {report['chunk_config']} · embedding model {report['embedding_model']}",
        "",
        f"**Found by the app: {share(s['found'])}**: the right passage was in the top {top_k} "
        f"with similarity of at least {threshold:.2f}",
        "",
        "| Measure | Result |",
        "|---|---|",
        f"| Ranked first | {share(s['top_1'])} |",
        f"| In the top 3 | {share(s['top_3'])} |",
        f"| In the top {top_k} (the app's Context chunks) | {share(s['top_k'])} |",
        f"| In the top {DEPTH} | {share(s['top_10'])} |",
        f"| Mean reciprocal rank (1.0 = always first) | {s['mrr']:.2f} |",
        f"| In the top {top_k}, but under the similarity threshold | {s['below_threshold']} |",
    ]
    missed = [r for r in report["results"] if not r["found"]]
    if missed:
        lines += ["", "### Not found by the app", "",
                  "| Question | Right passage | Its rank | Similarity | Top results instead |", "|---|---|---|---|---|"]
        for r in missed:
            rank = str(r["rank"]) if r["rank"] else f"not in top {DEPTH}"
            similarity = f"{r['similarity']:.2f}" if r["similarity"] is not None else "-"
            lines.append(f"| {r['question']} | {r['expected']} | {rank} | {similarity} | {'; '.join(r['retrieved'])} |")
    unanswerable = report["unanswerable"]
    if unanswerable:
        given = sum(1 for u in unanswerable if u["given_excerpts"])
        lines += ["", "### Questions the documents can't answer", "",
                  f"The app would pass excerpts to the model for {given} of {len(unanswerable)} "
                  f"(their best result has similarity of at least {threshold:.2f}).", "",
                  "| Question | Best similarity | Top results |", "|---|---|---|"]
        for u in unanswerable:
            best = f"{u['best_similarity']:.2f}" if u["best_similarity"] is not None else "-"
            lines.append(f"| {u['question']} | {best} | {'; '.join(u['retrieved'])} |")
    if n:
        u_count = len(unanswerable)
        lines += ["", "### Similarity thresholds compared", "",
                  "| Threshold | Answered from the right passage | Unanswerable questions given excerpts |",
                  "|---|---|---|"]
        for row in report["thresholds"]:
            current = " (current)" if abs(row["threshold"] - threshold) < 1e-9 else ""
            excerpts = f"{row['unanswerable_given_excerpts']}/{u_count}" if u_count else "-"
            lines.append(f"| {row['threshold']:.2f}{current} | {share(row['found'])} | {excerpts} |")
    if report["problems"]:
        lines += ["", "### Questions skipped", ""] + [f"- {p}" for p in report["problems"]]
    return "\n".join(lines)


def parse_draft(reply: str) -> Optional[Dict]:
    """The {"question", "quote"} object in a model's reply, or None."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        parsed = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    question, quote = str(parsed.get("question", "")).strip(), str(parsed.get("quote", "")).strip()
    return {"question": question, "quote": quote} if question and quote else None


def draft_questions(folder: str, llm, processor: DocumentProcessor = None, count: int = 20,
                    seed: int = 0) -> Tuple[List[Dict], int]:
    """
    Have a model write a question about each of a sample of chunks.

    Chunks are picked in turn from each document. A draft is kept only if its
    quote really is in the chunk.

    Args:
        folder: Folder with the documents
        llm: LLM backend to write the questions
        processor: Chunking settings (default: the app's)
        count: How many questions to draft
        seed: Random seed for picking chunks

    Returns:
        Tuple of (drafted questions, number of replies rejected)
    """
    documents = load_documents(folder, processor or DocumentProcessor())
    rng = random.Random(seed)
    queues = []
    for document in documents.values():
        chunks = [c for c in document.chunks if len(c["text"]) >= 200] or document.chunks
        rng.shuffle(chunks)
        queues.append((document.filename, chunks))

    drafts, rejected, attempts = [], 0, 0
    while len(drafts) < count and attempts < 2 * count and any(chunks for _, chunks in queues):
        for filename, chunks in queues:
            if not chunks or len(drafts) >= count:
                continue
            chunk = chunks.pop()
            attempts += 1
            prompt = DRAFT_PROMPT.format(document=filename, excerpt=chunk["text"])
            reply = "".join(llm.stream_chat([{"role": "user", "content": prompt}], DRAFT_CONFIG))
            draft = parse_draft(reply)
            if draft and normalize(draft["quote"]) in normalize(chunk["text"]):
                drafts.append({"question": draft["question"], "document": filename, "quote": draft["quote"]})
                print(f"  {len(drafts)}. {draft['question']}", file=sys.stderr)
            else:
                rejected += 1
    return drafts, rejected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="evaluate retrieval with a questions file")
    run.add_argument("questions", help="questions file; the documents are the files next to it")
    run.add_argument("--chunk-size", type=int, default=RAG_CONFIG["chunk_size"])
    run.add_argument("--chunk-overlap", type=int, default=RAG_CONFIG["chunk_overlap"])
    run.add_argument("--top-k", type=int, default=RAG_CONFIG["top_k"])
    run.add_argument("--min-similarity", type=float, default=RAG_CONFIG["min_similarity"])
    run.add_argument("--embedding-model", default=RAG_CONFIG["embedding_model"])
    run.add_argument("--json", metavar="PATH", help="also save the full results as JSON")

    draft = commands.add_parser("draft", help="have your model draft questions about your documents")
    draft.add_argument("folder", help="folder with your documents")
    draft.add_argument("--count", type=int, default=20)
    draft.add_argument("--model", help="model to use, e.g. llama3.1:8b (default: the app's)")
    draft.add_argument("--output", help="where to save the draft (default: FOLDER/questions-draft.json)")
    draft.add_argument("--seed", type=int, default=0, help="change to pick different chunks")

    args = parser.parse_args(argv)
    if args.command == "run":
        report = evaluate(args.questions, DocumentProcessor(args.chunk_size, args.chunk_overlap),
                          embedding_model_name=args.embedding_model, top_k=args.top_k,
                          min_similarity=args.min_similarity)
        print(format_report(report))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        return 0

    from llm_backends import default_model_key, list_ollama_models, load_backend

    output = args.output or os.path.join(args.folder, "questions-draft.json")
    if os.path.exists(output):
        parser.error(f"{output} already exists; move it or pass --output")
    model = default_model_key(args.model or DEFAULT_MODEL, list_ollama_models())
    print(f"Drafting {args.count} questions with {model}...", file=sys.stderr)
    drafts, rejected = draft_questions(args.folder, load_backend(model), count=args.count, seed=args.seed)
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"questions": drafts}, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(drafts)} question{'s' * (len(drafts) != 1)} to {output}; {rejected} "
          f"{'reply' if rejected == 1 else 'replies'} didn't quote the excerpt and {'was' if rejected == 1 else 'were'} "
          "dropped. Review them, then rename the file to questions.json.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
