# Databricks notebook source
# MAGIC %md
# MAGIC # 09_rag_index — document chunks, embeddings and a search index for the RAG assistant
# MAGIC
# MAGIC **RAG** (retrieval-augmented generation): before a language model answers a question, the most relevant passages are *retrieved* from the project documents and given to the model as evidence, so the answer can cite real text. This notebook builds the **retrieval** half; `10_rag_assistant` adds the language model, `11_rag_evaluation` measures it and `12_flagged_project_notes` uses it to explain high-risk projects. Shared settings and the `search()` helper live in `rag_common`.
# MAGIC
# MAGIC | Step | Output |
# MAGIC |---|---|
# MAGIC | 1. Measure the candidate documents (active + test-period projects) | printed counts and word distributions |
# MAGIC | 2–3. Clean and chunk the **full** documents (sentence-aware, ~260 words, 15% overlap, section headings) | in memory |
# MAGIC | 4. Choose a subset that fits the chunk budget (`MAX_CHUNKS`): highest-risk active projects first, then stratified by region × sector | printed selection tables |
# MAGIC | 5–7. Write and check the chunks | `silver.rag_chunks` (Delta, Change Data Feed on, primary key `chunk_id`) |
# MAGIC | 8. Embed new chunks with `databricks-gte-large-en` (self-managed embeddings, stored in the table) | `embedding` column |
# MAGIC | 9. Build the search indexes: FAISS file (always) and Vector Search Delta Sync index (default backend) | volume `gold.rag_index`, `gold.rag_chunks_index` |
# MAGIC | 10. Smoke-test retrieval on both backends | printed top-5 results |
# MAGIC
# MAGIC **Needs:** `01b_ingest_docs`, `04_gold_features`, `07_deploy`. **Attach to:** `wb-ml-single`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## How it was built and tested
# MAGIC
# MAGIC The key decisions behind the index, in the order they came up.
# MAGIC
# MAGIC **Step 1 — How much text is there?**
# MAGIC - **Question:** can every candidate document be indexed?
# MAGIC - **What we found:** 3,777 documents with about 106 million words, nearly half a million chunks. That is about ten times the 50,000-chunk budget.
# MAGIC - **Decision:** index a subset. All 323 high-risk active projects go in first, then the remaining budget is spread across other projects by region and sector. Side effect, stated openly: about 65% of the indexed documents are African projects.
# MAGIC
# MAGIC **Step 2 — Chunk size and boundaries**
# MAGIC - **Question:** how big should a chunk be, and where should it break?
# MAGIC - **What we found:** the embedding model's tokenizer gives 1.33 tokens per word on these documents. The documents also have real section structure (numbered headings such as "VI. KEY RISKS").
# MAGIC - **Decision:** chunks of about 260 words (about 350 tokens). A new chunk starts at each section heading, and about 40 words of whole sentences overlap between chunks so that a fact on a boundary appears whole in one of them. Each chunk is embedded with a short header (project, country, document type, section) so that it still makes sense on its own.
# MAGIC
# MAGIC **Step 3 — Printing sample chunks early**
# MAGIC - **Question:** do the chunks look right before any slow step runs?
# MAGIC - **What we found:** printing a few real chunks exposed problems that the counts alone did not show:
# MAGIC   - Windows line endings meant that no heading was detected.
# MAGIC   - The running page header (the project name in capitals), footnotes and risk-table rows were treated as section headings.
# MAGIC   - Garbled characters (`â€™`) came from double-encoded text.
# MAGIC   - A short chunk was copied whole into the next one as "overlap".
# MAGIC - **Decision:** normalise line endings first, remove short lines repeated on many pages before detecting headings, tighten the heading rules, repair the encoding and cap the overlap. Sample chunks and the chunks either side of the quality cut-off stay printed in the notebook, so the cleaning can be checked by eye.
# MAGIC
# MAGIC **Step 4 — Making both search backends agree**
# MAGIC - **Question:** will FAISS (cosine similarity) and Vector Search (distance) rank chunks the same way?
# MAGIC - **What we found:** the GTE vectors are not unit length (length about 24). For vectors like that, ranking by distance and ranking by cosine differ.
# MAGIC - **Decision:** normalise every vector to length 1 before storing it. Check: FAISS and Vector Search returned the same top 5 for every smoke-test question, and identical retrieval scores in `11`.
# MAGIC
# MAGIC **Step 5 — Re-runs should not redo the slow work**
# MAGIC - **Question:** what happens when the notebook runs again?
# MAGIC - **What we found:** the workspace accepts only about 15 texts per second for embedding, whatever the batch size, so embedding the 50,000 chunks took about an hour.
# MAGIC - **Decision:** store the vectors in the Delta table and write with MERGE on a deterministic `chunk_id`, re-embedding only chunks whose text changed. The final re-run changed 0 rows and embedded 0 chunks. The same stored vectors feed both FAISS and Vector Search.
# MAGIC
# MAGIC Whether hybrid (vector + keyword) search beats vector-only search was settled by measurement in `11_rag_evaluation`, not by the smoke test here.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install the client libraries
# MAGIC
# MAGIC **What:** installs `databricks-vectorsearch` (Python client for Vector Search) and `faiss-cpu` (FAISS, Facebook AI Similarity Search, an open-source vector index library), then restarts Python so the new packages load.
# MAGIC
# MAGIC **Why:** neither ships with Databricks Runtime 18 ML. Installing at notebook scope leaves the cluster unchanged for the other notebooks.

# COMMAND ----------

# MAGIC %pip install -q databricks-vectorsearch faiss-cpu

# COMMAND ----------

# MAGIC %md
# MAGIC **What:** restarts the notebook's Python process.
# MAGIC
# MAGIC **Why:** packages installed with `%pip` are only importable after a restart; everything defined before this point is cleared, so settings come after it.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Settings
# MAGIC
# MAGIC **What:** the vector backend, the chunk budget and the chunking parameters. Names of tables, index, endpoint and embedding model come from `rag_common` (next cell).
# MAGIC
# MAGIC **Why:**
# MAGIC - **`MAX_CHUNKS = 50,000`**: the 3,777 candidate documents hold ~106 million words, nearly half a million chunks — far more than a demo needs. 50,000 chunks keeps the index small while still covering a few hundred full documents. It also bounds the embedding time: this workspace's pay-per-token limit is about **15 texts per second**, so 50,000 chunks take ~1 hour (470,000 would take ~9 hours).
# MAGIC - **Chunk size in tokens, approximated from words:** the embedding model reads *tokens* (word pieces). On 20 sample documents the GTE tokenizer produced **1.33 tokens per word**, so the target of 300–400 tokens per chunk is **260 words target, 300 words maximum** (~350–400 tokens). Big enough to hold a whole argument (a risk and its mitigation), small enough that one chunk is about one topic.
# MAGIC - **Overlap ~15% (40 words of whole sentences):** a fact that straddles a chunk boundary still appears whole in one chunk.
# MAGIC - **`MIN_SECTION_FLUSH = 80`:** a new heading starts a new chunk only once the current chunk has 80+ new words, so tiny sections don't become tiny chunks.
# MAGIC - **`PROSE_MIN`** drops chunks that are mostly tables, acronym lists or tables of contents (calibrated in step 3a).

# COMMAND ----------

dbutils.widgets.dropdown("vector_backend", "vector_search", ["vector_search", "faiss"], "Vector backend")
VECTOR_BACKEND = dbutils.widgets.get("vector_backend")
RAG_BUILDER = True             # tells rag_common not to fall back to FAISS: this notebook creates the endpoint

MAX_CHUNKS = 50_000            # total chunk budget for the index
TOKENS_PER_WORD = 1.33         # measured with the GTE tokenizer on 20 documents
TARGET_WORDS = 260             # ~350 tokens
MAX_WORDS = 300                # ~400 tokens; only a single over-long "sentence" (a flattened table) is cut
OVERLAP_WORDS = 40             # ~15% of TARGET_WORDS, carried as whole sentences
MIN_SECTION_FLUSH = 80
MIN_CHUNK_WORDS = 25           # shorter chunks carry too little meaning to be useful evidence
PROSE_MIN = 0.35               # minimum share of lower-case words; below = table / list / contents page
ACTIVE_SHARE = 0.6             # of the budget left after high-risk projects: share for other active projects
CHUNKER_VERSION = "v1"
EMBED_BATCH = 8                # texts per embedding request (the workspace limit counts every text as a query)
EMBED_THREADS = 1              # one request at a time already reaches ~15 texts/s, the measured workspace limit
EMBED_GROUP = 5_000            # chunks embedded and saved per checkpoint
SYNC_TIMEOUT_MIN = 75

# COMMAND ----------

# MAGIC %md
# MAGIC **What:** loads the shared constants and helpers (`search()`, `embed_texts()`, `chat()`) from `rag_common`, using the `VECTOR_BACKEND` chosen above.
# MAGIC
# MAGIC **Why:** `%run` executes the other notebook inside this one, so all RAG notebooks share one definition of names, filters and endpoint calls.

# COMMAND ----------

# MAGIC %run ./rag_common

# COMMAND ----------

# MAGIC %md
# MAGIC **What:** imports and a timing dictionary.
# MAGIC
# MAGIC **Why:** every stage records its duration, so the summary shows where the time goes.

# COMMAND ----------

import re, json, time, hashlib, unicodedata
from collections import Counter
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pyspark.sql import functions as F, Window
T = {}
t_start = time.time()
print("backend:", VECTOR_BACKEND)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Measure the candidates
# MAGIC
# MAGIC **What:** candidates are (a) the 2,505 active projects scored by the champion model in `gold.active_project_scores` and (b) the test-period projects (rated, approved FY2012+, `split = 'test'` in `gold.project_features`). We keep those with a downloaded pre-approval document and print how many there are, their document types, and the words-per-document distribution. Readable region / sector / country labels come from `silver.projects` (the gold table stores them one-hot encoded for the model).
# MAGIC
# MAGIC **Why:** the subset decision must rest on numbers. Active projects are what a portfolio manager asks about today; test-period projects have a known outcome, which lets later notebooks check whether the documents' warnings matched what happened. A **risk decile** (1 = the 10% of active projects with the highest predicted risk of failure) is added so the assistant can filter on it.

# COMMAND ----------

act = (spark.table(f"{CATALOG}.gold.active_project_scores")
       .select("project_id", "risk_of_failure", "risk_rank", "risk_band")
       .withColumn("risk_decile", F.ntile(10).over(Window.orderBy(F.desc("risk_of_failure")))))
test_ids = spark.table(f"{CATALOG}.gold.project_features").filter("split = 'test'").select("project_id")
cand = (act.select("project_id").withColumn("is_active", F.lit(True))
        .unionByName(test_ids.withColumn("is_active", F.lit(False))))
proj = spark.table(f"{CATALOG}.silver.projects").select(
    "project_id", "project_name", F.col("country_name").alias("country"), "region",
    F.regexp_replace("major_sector", r"^FY\d\d - ", "").alias("sector"), "approval_fy", "approval_date")
docs_sdf = (cand
    .join(spark.table(f"{CATALOG}.bronze.doc_downloads").filter("status in ('ok','exists')").select("project_id", "doc_id", "path"), "project_id")
    .join(spark.table(f"{CATALOG}.bronze.doc_manifest").select("project_id", "doc_id", "doc_type"), ["project_id", "doc_id"])
    .join(proj, "project_id", "left")
    .join(act.drop("risk_rank"), "project_id", "left")
    .join(spark.table(f"{CATALOG}.silver.doc_text_features").select("project_id", "n_words"), "project_id", "left"))
docs = docs_sdf.toPandas()

n_cand = cand.groupBy("is_active").count().toPandas().set_index("is_active")["count"]
n_high = act.filter("risk_band = 'High'").count()
print(f"candidates: active {n_cand[True]:,}, test {n_cand[False]:,}")
print(f"with a document: active {int(docs.is_active.sum()):,}, test {int((~docs.is_active).sum()):,}")
print(f"high-risk active projects with a document: {int(((docs.risk_band == 'High') & docs.is_active).sum())} of {n_high}")
print("\ndocument types:\n" + docs.groupby(["is_active", "doc_type"]).size().to_string())
print("\nwords per document (quantiles):")
print(docs.groupby("is_active")["n_words"].quantile([.1, .25, .5, .75, .9, 1.0]).unstack().round().to_string())
est = docs.n_words.sum() / (TARGET_WORDS - OVERLAP_WORDS)
print(f"\ntotal words: {docs.n_words.sum() / 1e6:.1f} M -> roughly {est / 1e3:.0f}k chunks before filtering, vs budget {MAX_CHUNKS / 1e3:.0f}k")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Cleaning and chunking rules
# MAGIC
# MAGIC **What:** functions that turn one raw `.txt` file into chunks. The rules come from reading sample documents first (details in "How it was built and tested"):
# MAGIC 1. **Repair mojibake** — text that was UTF-8 but got decoded as Windows-1252 and saved again, so "Peru's" appears as `Peruâ€™s`. Each such run of characters is re-encoded and decoded correctly.
# MAGIC 2. **Normalise line endings** — the files use Windows `\r\n` line ends and form feeds (`\f`) at page breaks; without this, line-based rules silently match nothing.
# MAGIC 3. **Reuse the `03_unstructured_silver` cleaning** — remove table-of-contents dot leaders and page-number lines, collapse spaces — but *keep line breaks* until headings are found (03 flattened each document to one line, fine for features but it loses structure).
# MAGIC 4. **Drop running page headers/footers** — short lines repeated 4+ times in a document (e.g. "The World Bank", "<PROJECT NAME> (P123456)") — and Project Information Document template markers (`@#&OPS~...`). This runs *before* heading detection: in the first trial run the ALL-CAPS running header "TRANSPORT AND CONNECTIVITY SUPPORT PROJECT (P161877)" was taken for a section heading on every page.
# MAGIC 5. **Detect section headings** with cheap rules: Roman-numbered ("III. IMPLEMENTATION"), lettered/numbered short title-case lines ("B. Fiduciary Arrangements", "2.3 Key Risks"), "ANNEX 1: ...", and short ALL-CAPS lines. Not headings: lines ending in a digit (table-of-contents entries) or like a sentence; footnotes ("7 World Bank, 2018, Systematic Country Diagnostic" — a bare number, a year, several commas); table cells with symbols ("10. Overall ⚫ High"). Two levels are tracked, so a section reads like "V. KEY RISKS > A. Overall Risk Rating".
# MAGIC 6. **Split into sentences** at `. ! ?` followed by a capital letter or digit, skipping abbreviations ("e.g.", "No.", "Mr.") and initials; 1–2-word fragments (paragraph numbers like "110.") are glued to the next sentence.
# MAGIC 7. **Pack sentences into chunks** — never cutting a sentence; each chunk starts with the last ~40 words (whole sentences) of the previous one, unless that would repeat more than half of a short chunk; a new section starts a new chunk with no overlap across sections. Only exception: one "sentence" longer than 300 words (a table flattened into a line, no full stops) is cut into 260-word pieces and flagged `forced_split`.
# MAGIC 8. **Prose ratio** = share of words starting with a lower-case letter. Normal prose scores about 0.6–0.75; acronym lists, numeric tables and contents pages score far lower.
# MAGIC
# MAGIC **Why:** retrieval quality is decided here. Junk chunks (acronym tables, page headers) match any query that mentions an acronym, and a chunk cut mid-sentence loses the meaning the model needs to answer.

# COMMAND ----------

DOT_LEADER = re.compile(r"\.{3,}|_{3,}|-{3,}")                                    # from 03
PAGE_LINE = re.compile(r"^\s*(?:page\s*)?\d{1,3}(?:\s*of\s*\d{1,3})?\s*$|^\s*-\s*\d{1,3}\s*-\s*$", re.I)  # 03, extended
WS = re.compile(r"\s+")                                                           # from 03
TEMPLATE_MARK = re.compile(r"@#&OPS~")
# Mojibake = a UTF-8 lead byte shown as a Latin-1 letter (\u00c2-\u00f4), followed by 1-3 continuation bytes shown as
# Latin-1 (\u0080-\u00bf) or as their Windows-1252 look-alikes (curly quotes, dashes, euro sign, ...).
CP1252_EXTRA = "".join(chr(c) for c in [0x152, 0x153, 0x160, 0x161, 0x178, 0x17D, 0x17E, 0x192, 0x2C6, 0x2DC,
                                        0x2013, 0x2014, 0x2018, 0x2019, 0x201A, 0x201C, 0x201D, 0x201E, 0x2020,
                                        0x2021, 0x2022, 0x2026, 0x2030, 0x2039, 0x203A, 0x20AC, 0x2122])
MOJIBAKE = re.compile("[\u00c2-\u00f4][\u0080-\u00bf" + re.escape(CP1252_EXTRA) + "]{1,3}")
ROMAN = r"(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)"
NUMBERED_HEAD = re.compile(rf"^(?P<num>{ROMAN}|[A-H]|\d{{1,2}}(?:\.\d{{1,2}}){{0,2}})[.):]?\s+(?P<title>[A-Z][^.!?;]{{2,90}})$")
ANNEX_HEAD = re.compile(r"^(?:ANNEX|Annex|APPENDIX|Appendix)\s+[0-9IVX]+[A-Z]?\b.{0,90}$")
YEAR = re.compile(r"\b(19|20)\d\d\b")
SMALL_WORDS = {"and", "of", "the", "for", "to", "in", "on", "a", "an", "with", "by", "or", "at", "from", "as"}
ABBREV = {"e.g", "i.e", "mr", "ms", "mrs", "dr", "no", "nos", "para", "paras", "vs", "etc", "fig", "rev", "st", "inc",
          "ltd", "co", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "approx",
          "u.s", "u.k", "u.n", "art", "sec", "op", "cit", "al", "p", "pp", "vol", "ch"}
SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def fix_mojibake(text):
    def repl(m):
        s = m.group(0)
        try:
            return bytes(ord(c) if ord(c) < 256 else c.encode("cp1252")[0] for c in s).decode("utf-8")
        except Exception:
            return s
    return MOJIBAKE.sub(repl, text)


def heading_level(line):
    """(level, text) if the line looks like a section heading, else None."""
    if len(line) < 4 or len(line) > 100 or line[-1].isdigit() or line.endswith((",", ";")):
        return None
    words = line.split()
    if len(words) > 14:
        return None
    if ANNEX_HEAD.match(line):
        return 1, line
    m = NUMBERED_HEAD.match(line)
    if m:
        title, num = m.group("title").strip(), m.group("num")
        tw = [w for w in title.split() if w.lower() not in SMALL_WORDS]
        if (title.endswith(".") or len(title.split()) > 12 or sum(w[:1].isupper() for w in tw) < 0.5 * max(1, len(tw))
                or (num.isdigit() and line[len(num)] not in ".)")        # "7 World Bank, 2018, ..." = a footnote
                or YEAR.search(title) or title.count(",") >= 2           # citations, not headings
                or any(unicodedata.category(c) == "So" for c in title)): # table cells with symbols, e.g. "Overall ⚫ High"
            return None
        return (1 if re.fullmatch(ROMAN, num) else 2), line
    letters = [c for c in line if c.isalpha()]
    if len(words) >= 2 and len(letters) >= 0.6 * len(line.replace(" ", "")) and \
            sum(c.isupper() for c in letters) >= 0.9 * len(letters):
        return 1, line
    return None


def clean_lines(raw):
    text = fix_mojibake(raw.replace("\ufeff", "").replace("\u00ef\u00bb\u00bf", ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    lines = [WS.sub(" ", DOT_LEADER.sub(" ", l)).strip() for l in text.split("\n")]
    counts = Counter(l for l in lines if l and len(l) <= 150)
    repeated = {l for l, c in counts.items() if c >= 4}         # running page headers/footers (even ALL-CAPS ones)
    out, stats = [], Counter()
    for l in lines:
        if not l:
            out.append("")
        elif TEMPLATE_MARK.search(l):
            stats["template_marker_lines"] += 1
        elif PAGE_LINE.match(l):
            stats["page_number_lines"] += 1
        elif l in repeated:
            stats["repeated_header_lines"] += 1
        else:
            out.append(l)
    return out, stats


def split_sentences(par):
    parts, start = [], 0
    for m in SENT_END.finditer(par):
        prev = par[start:m.start()].rsplit(" ", 1)[-1].rstrip(".!?\"')]").lower()
        if prev in ABBREV or (len(prev) == 1 and prev.isalpha()):
            continue
        parts.append(par[start:m.start()].strip())
        start = m.end()
    parts.append(par[start:].strip())
    out = []
    for p in parts:
        if out and len(out[-1].split()) <= 2:
            out[-1] = out[-1] + " " + p
        elif p:
            out.append(p)
    return out


def doc_units(raw):
    """List of ('h', level, heading) and ('s', sentence) units in reading order."""
    lines, stats = clean_lines(raw)
    units, buf = [], []

    def flush_par():
        if buf:
            par = re.sub(r"(\w)- (\w)", r"\1\2", " ".join(buf))    # re-join words hyphenated at line ends
            units.extend(("s", s) for s in split_sentences(par))
            buf.clear()

    for l in lines:
        if not l:
            flush_par()
            continue
        h = heading_level(l)
        if h:
            flush_par()
            units.append(("h", h[0], h[1]))
            stats["headings"] += 1
        else:
            buf.append(l)
    flush_par()
    return units, stats


def prose_ratio(text):
    toks = text.split()
    return sum(t[:1].islower() for t in toks) / max(1, len(toks))


def chunk_document(raw, target_words=TARGET_WORDS, max_words=MAX_WORDS, overlap_words=OVERLAP_WORDS,
                   min_section_flush=MIN_SECTION_FLUSH):
    units, stats = doc_units(raw)
    chunks = []
    cur, cur_words, n_carried, carried_words = [], 0, 0, 0
    sec1 = sec2 = None
    cur_section = "Front matter"

    def section_name():
        return (f"{sec1} > {sec2}" if sec1 and sec2 else (sec1 or sec2 or "Front matter"))[:160]

    def emit(carry_overlap):
        nonlocal cur, cur_words, n_carried, carried_words
        text = " ".join(t for t, _ in cur)
        chunks.append({"section": cur_section, "text": text, "n_words": len(text.split()),
                       "forced_split": any(f for _, f in cur)})
        keep, kw = [], 0
        if carry_overlap:                        # last whole sentences, ~overlap_words (at most 2x that)
            for t, f in reversed(cur):
                w = len(t.split())
                if kw + w > 2 * overlap_words:
                    break
                keep.insert(0, (t, f)); kw += w
                if kw >= overlap_words:
                    break
            if kw > 0.5 * cur_words:             # a short chunk would be repeated almost whole: no overlap
                keep, kw = [], 0
        cur, cur_words, n_carried, carried_words = keep, kw, len(keep), kw

    def add(t, forced):
        nonlocal cur_words, cur_section
        if len(cur) == n_carried:                # the first new unit decides the chunk's section
            cur_section = section_name()
        cur.append((t, forced)); cur_words += len(t.split())

    for u in units:
        if u[0] == "h":
            if cur_words - carried_words >= min_section_flush:
                emit(carry_overlap=False)        # new section -> new chunk, no overlap across sections
            elif len(cur) == n_carried:
                cur, cur_words, n_carried, carried_words = [], 0, 0, 0
            if u[1] == 1:
                sec1, sec2 = u[2], None
            else:
                sec2 = u[2]
            if len(cur) > n_carried:             # a small preamble stays with this section
                cur_section = section_name()
            add(u[2], False)
            continue
        words = u[1].split()
        pieces = [u[1]] if len(words) <= max_words else \
            [" ".join(words[i:i + target_words]) for i in range(0, len(words), target_words)]
        for piece in pieces:
            pw = len(piece.split())
            if cur_words + pw > max_words and cur_words - carried_words > 0:
                emit(carry_overlap=True)
            add(piece, len(pieces) > 1)
            if cur_words >= target_words:
                emit(carry_overlap=True)
    new_units = cur[n_carried:]
    if new_units:
        new_words = sum(len(t.split()) for t, _ in new_units)
        if chunks and new_words < min_section_flush and chunks[-1]["section"] == cur_section:
            chunks[-1]["text"] += " " + " ".join(t for t, _ in new_units)     # a short tail joins the previous chunk
            chunks[-1]["n_words"] = len(chunks[-1]["text"].split())
        else:
            emit(carry_overlap=False)
    return chunks, stats

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a. Try the rules on one document
# MAGIC
# MAGIC **What:** chunks the document of the highest-risk active project that has a Project Appraisal Document, and prints some chunks with their section, word count and prose ratio.
# MAGIC
# MAGIC **Why:** a quick human check that headings, sentence boundaries and overlap look right before processing thousands of files.

# COMMAND ----------

ex = docs[(docs.risk_band == "High") & (docs.doc_type == "Project Appraisal Document")].sort_values("risk_of_failure", ascending=False).iloc[0]
ex_chunks, ex_stats = chunk_document(open(ex.path, "rb").read().decode("utf-8", errors="replace"))
print(ex.project_id, ex.project_name, "|", ex.doc_type, "|", len(ex_chunks), "chunks |", dict(ex_stats))
for i, c in enumerate(ex_chunks[:48]):
    if i < 3 or i % 8 == 0:
        print(f"\n#{i} [{c['section']}] {c['n_words']} words, prose {prose_ratio(c['text']):.2f}\n   {c['text'][:350]} ... {c['text'][-120:]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Chunk every candidate document (in parallel)
# MAGIC
# MAGIC **What:** runs `chunk_document` on all candidate files with Spark `mapInPandas` (the 4 cores each process a slice of the documents) and caches the result. Each document also returns one summary row (`chunk_index = -1`) with its cleaning statistics, so documents that produce zero chunks are still visible.
# MAGIC
# MAGIC **Why:** we need the *actual* number of usable chunks per document to choose the subset — word counts over-estimate, because table-heavy annexes are filtered out later. A plain Python loop over the 3,777 files ran for more than 15 minutes on one core in a trial; 4 cores in parallel cut that roughly 4x. Chunking everything is cheap (CPU only); only the chosen subset is embedded, which is the slow part.

# COMMAND ----------

t0 = time.time()
chunk_schema = ("project_id string, doc_id string, chunk_index int, section string, chunk_text string, "
                "n_words int, forced_split boolean, prose double, doc_stats string")
chunk_cols = [c.split()[0] for c in chunk_schema.split(", ")]

def chunk_files(batches):
    for pdf in batches:
        rows = []
        for r in pdf.itertuples(index=False):
            try:
                chunks, stats = chunk_document(open(r.path, "rb").read().decode("utf-8", errors="replace"))
            except Exception:
                chunks, stats = [], Counter({"error": 1})
            stats["n_chunks_raw"] = len(chunks)
            rows.append((r.project_id, r.doc_id, -1, None, None, 0, False, None, json.dumps(stats)))
            for i, c in enumerate(chunks):
                rows.append((r.project_id, r.doc_id, i, c["section"], c["text"], c["n_words"],
                             c["forced_split"], prose_ratio(c["text"]), None))
        yield pd.DataFrame(rows, columns=chunk_cols)

all_chunks = (docs_sdf.select("project_id", "doc_id", "path").repartition(32)
              .mapInPandas(chunk_files, chunk_schema).cache())
n_rows = all_chunks.count()
T["chunk_all_s"] = round(time.time() - t0)
ds = all_chunks.filter("chunk_index = -1").select("project_id", "doc_stats").toPandas()
doc_stats = pd.concat([ds[["project_id"]], pd.json_normalize(ds.doc_stats.map(json.loads)).fillna(0)], axis=1)
print(f"chunked {len(doc_stats):,} documents into {n_rows - len(doc_stats):,} raw chunks in {T['chunk_all_s']} s")
print("cleaning totals:", doc_stats.drop(columns="project_id").sum().astype(int).to_dict())
print("documents with 0 headings detected:", int((doc_stats.get("headings", pd.Series(0, index=doc_stats.index)) == 0).sum()))
print("documents with 0 chunks:", int((doc_stats.n_chunks_raw == 0).sum()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Calibrate and apply the quality filter
# MAGIC
# MAGIC **What:** prints the prose-ratio and word-count distributions of the raw chunks, with examples just below and just above `PROSE_MIN`, then keeps chunks with at least `MIN_CHUNK_WORDS` words and prose ratio ≥ `PROSE_MIN`.
# MAGIC
# MAGIC **Why:** a threshold should be checked against real examples, not picked blind. Dropped chunks are mostly tables, acronym lists and contents pages: they rarely answer a question but often match one by accident.

# COMMAND ----------

raw_chunks = all_chunks.filter("chunk_index >= 0")
print("prose ratio quantiles (1,5,10,20,30,50,75%):", [round(x, 3) for x in raw_chunks.approxQuantile("prose", [.01, .05, .1, .2, .3, .5, .75], 0.001)])
# probabilities must be floats: Python ints 0 and 1 reach the JVM as Integer -> ClassCastException
print("words/chunk quantiles (0,1,5,25,50,75,95,99,100%):",
      raw_chunks.approxQuantile("n_words", [0.0, .01, .05, .25, .5, .75, .95, .99, 1.0], 0.001))
for lo, hi in [(PROSE_MIN - 0.1, PROSE_MIN), (PROSE_MIN, PROSE_MIN + 0.1)]:
    print(f"\n--- examples with prose ratio in [{lo:.2f}, {hi:.2f}) ---")
    for r in raw_chunks.filter(f"prose >= {lo} and prose < {hi}").orderBy(F.rand(11)).limit(3).collect():
        print(f"  [{r.project_id} #{r.chunk_index} {r.section[:50]!r}] {r.chunk_text[:300]}")
good = raw_chunks.filter((F.col("n_words") >= MIN_CHUNK_WORDS) & (F.col("prose") >= PROSE_MIN))
per_doc = good.groupBy("project_id").agg(F.count("*").alias("n_chunks"), F.sum("n_words").alias("chunk_words")).toPandas()
n_raw, n_good = n_rows - len(doc_stats), int(per_doc.n_chunks.sum())
print(f"\nkept {n_good:,} of {n_raw:,} raw chunks ({n_good / n_raw:.1%}); dropped {n_raw - n_good:,} (too short or prose < {PROSE_MIN})")
print("usable chunks per document (quantiles):", per_doc.n_chunks.quantile([.1, .25, .5, .75, .9, 1]).tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Choose the subset (budget `MAX_CHUNKS`)
# MAGIC
# MAGIC **What:** orders documents and takes them while the budget lasts:
# MAGIC 1. **All high-risk active projects** (top 20% predicted risk of failure), highest risk first — notebook 12 writes "why this project was flagged" notes for them.
# MAGIC 2. The remaining budget is split: `ACTIVE_SHARE` (60%) for other active projects, the rest for test-period projects (any budget the first pool leaves unused flows to the second).
# MAGIC 3. Within each pool, documents are **stratified by region × sector**: each stratum's documents get a stable pseudo-random order (a hash of `project_id`), and the pool is sorted by *position within its stratum ÷ stratum size*. Taking documents from the top of that list gives every stratum a share proportional to its size, and every stratum gets its first document early.
# MAGIC 4. A document that does not fit the remaining budget is skipped (a smaller one may still fit).
# MAGIC
# MAGIC **Why:** whole documents rather than slices of many: a question about one project should find all of its relevant passages (risks, fiduciary arrangements, results framework). Stratifying stops one region or sector from dominating the index. The hash order makes the selection **deterministic** — a re-run picks the same documents, so nothing is re-embedded.

# COMMAND ----------

sel = docs.merge(per_doc, on="project_id", how="inner")
sel["h"] = sel.project_id.map(lambda s: int(hashlib.md5(s.encode()).hexdigest()[:8], 16))
sel["stratum"] = sel.region.fillna("?") + " | " + sel.sector.fillna("?")

def stratified_order(pool):
    pool = pool.sort_values("h").copy()
    pool["pos"] = pool.groupby("stratum").cumcount() / pool.groupby("stratum")["h"].transform("count")
    return pool.sort_values(["pos", "h"])

def take(pool, budget):
    chosen_ids, used = [], 0
    for r in pool.itertuples(index=False):
        if used + r.n_chunks <= budget:
            chosen_ids.append(r.project_id); used += r.n_chunks
    return chosen_ids, used

tier0 = sel[sel.is_active & (sel.risk_band == "High")].sort_values("risk_of_failure", ascending=False)
ids0, used0 = take(tier0, MAX_CHUNKS)
left = MAX_CHUNKS - used0
ids1, used1 = take(stratified_order(sel[sel.is_active & (sel.risk_band != "High")]), int(left * ACTIVE_SHARE))
ids2, used2 = take(stratified_order(sel[~sel.is_active]), left - used1)
chosen = sel[sel.project_id.isin(ids0 + ids1 + ids2)].copy()
chosen["tier"] = np.select([chosen.project_id.isin(ids0), chosen.project_id.isin(ids1)],
                           ["0 high-risk active", "1 other active"], "2 test period")
print(f"chosen {len(chosen)} documents, {chosen.n_chunks.sum():,} chunks (budget {MAX_CHUNKS:,})")
print(f"high-risk active documents included: {len(ids0)} of {len(tier0)}\n")
for col in ["tier", "region", "sector", "doc_type"]:
    print(chosen.groupby(col).agg(docs=("project_id", "size"), chunks=("n_chunks", "sum"),
                                  median_chunks_per_doc=("n_chunks", "median")).to_string(), "\n")
pool_share = sel.groupby("region").size() / len(sel)
print("region share: candidates vs chosen\n" + pd.DataFrame({"candidates": pool_share.round(3),
      "chosen": (chosen.groupby("region").size() / len(chosen)).round(3)}).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Build the chunk rows
# MAGIC
# MAGIC **What:** for the chosen documents, adds the metadata and two text columns:
# MAGIC - `chunk_text` — the clean chunk, for display and citation;
# MAGIC - `embed_text` — a **contextual header** plus the chunk: `"<project name> | <country> | <doc type> | <section>: <chunk text>"`. This is what gets embedded.
# MAGIC
# MAGIC `chunk_id = <project_id>_<doc_id>_<chunk_index>` (index from the raw chunk sequence, zero-padded) and `text_hash` = SHA-256 of `embed_text`.
# MAGIC
# MAGIC **Why:**
# MAGIC - **Contextual header:** a chunk saying "the procurement unit lacks experience" cannot match "procurement problems in Kenya" unless its vector also knows it comes from a Kenya project and which section it sits in. Prepending this short context is a cheap, well-known retrieval improvement.
# MAGIC - **Stable, deterministic `chunk_id`:** the same document and rules always give the same ids, so re-runs update rows in place instead of duplicating them, and citations in notebooks 10–12 stay valid. The index is taken *before* filtering, so changing the filter threshold does not renumber the other chunks.
# MAGIC - **`text_hash`:** shows whether a chunk's embedded text changed, so only changed chunks are re-embedded.

# COMMAND ----------

meta_cols = ["project_id", "doc_id", "doc_type", "project_name", "country", "region", "sector", "approval_fy",
             "approval_date", "is_active", "risk_of_failure", "risk_decile", "risk_band"]
# take the metadata from the Spark DataFrame (keeps proper NULLs and types), filtered to the chosen projects
chosen_sdf = (docs_sdf.join(F.broadcast(spark.createDataFrame(chosen[["project_id"]])), "project_id")
              .select(*meta_cols))
rows_sdf = (good.join(F.broadcast(chosen_sdf), ["project_id", "doc_id"])
            .withColumn("chunk_id", F.format_string("%s_%s_%04d", "project_id", "doc_id", "chunk_index"))
            .withColumn("embed_text", F.concat(F.coalesce("project_name", F.lit("?")), F.lit(" | "),
                        F.coalesce("country", F.lit("?")), F.lit(" | "), "doc_type", F.lit(" | "), "section",
                        F.lit(": "), "chunk_text"))
            .withColumn("text_hash", F.sha2("embed_text", 256))
            .withColumn("chunker_version", F.lit(CHUNKER_VERSION))
            .select("chunk_id", "project_id", "doc_id", "doc_type", "project_name", "country", "region", "sector",
                    "approval_fy", "approval_date", "chunk_index", "section", "n_words", "forced_split",
                    F.round("prose", 3).alias("prose_ratio"), "is_active", "risk_of_failure", "risk_decile", "risk_band",
                    "chunk_text", "embed_text", "text_hash", "chunker_version"))
rows_sdf.createOrReplaceTempView("rag_chunks_new")
print("rows to write:", rows_sdf.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write `silver.rag_chunks` (idempotent MERGE)
# MAGIC
# MAGIC **What:** creates the table once (if missing) with:
# MAGIC - **Change Data Feed (CDF)** on — Delta records which rows were inserted, updated or deleted in each table version; the Vector Search **Delta Sync** index reads this feed to update itself incrementally;
# MAGIC - an informational **primary key** on `chunk_id` (Unity Catalog records it but does not enforce it; uniqueness is checked in step 7);
# MAGIC - an `embedding ARRAY<FLOAT>` column, filled in step 8.
# MAGIC
# MAGIC Then it **MERGE**s the new rows: insert new chunk ids, update rows whose content changed (clearing the embedding only if the embedded text changed), delete chunk ids that are no longer produced.
# MAGIC
# MAGIC **Why one table with the vectors in it, not a separate embeddings table:** a Delta Sync index with self-managed embeddings reads the vectors from its source table, so a separate embeddings table would need a third, joined copy. One table also makes "which chunks still need a vector?" a simple `embedding IS NULL` filter. **Why MERGE, not overwrite:** an overwrite rewrites every row, so CDF reports ~50,000 changes and the index re-processes all of them; with MERGE an unchanged re-run changes **zero** rows, keeps every stored embedding, and the sync has nothing to do.

# COMMAND ----------

t0 = time.time()
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE} (
  chunk_id STRING NOT NULL, project_id STRING, doc_id STRING, doc_type STRING, project_name STRING, country STRING,
  region STRING, sector STRING, approval_fy INT, approval_date DATE, chunk_index INT, section STRING, n_words INT,
  forced_split BOOLEAN, prose_ratio DOUBLE, is_active BOOLEAN, risk_of_failure DOUBLE, risk_decile INT, risk_band STRING,
  chunk_text STRING, embed_text STRING, text_hash STRING, chunker_version STRING,
  embedding ARRAY<FLOAT>, embedding_model STRING, updated_at TIMESTAMP,
  CONSTRAINT rag_chunks_pk PRIMARY KEY (chunk_id))
COMMENT 'Full-document chunks of pre-approval World Bank project documents for RAG (built by 09_rag_index). embed_text = contextual header + chunk_text; embedding = databricks-gte-large-en vector of embed_text.'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
data_cols = [c for c in rows_sdf.columns if c != "chunk_id"]
changed = " OR ".join(f"NOT (t.{c} <=> s.{c})" for c in data_cols)
merge_res = spark.sql(f"""
MERGE INTO {CHUNKS_TABLE} t USING rag_chunks_new s ON t.chunk_id = s.chunk_id
WHEN MATCHED AND ({changed}) THEN UPDATE SET
  {", ".join(f"t.{c} = s.{c}" for c in data_cols)},
  t.embedding = CASE WHEN t.text_hash = s.text_hash THEN t.embedding END,
  t.embedding_model = CASE WHEN t.text_hash = s.text_hash THEN t.embedding_model END,
  t.updated_at = current_timestamp()
WHEN NOT MATCHED THEN INSERT ({", ".join(rows_sdf.columns)}, updated_at)
  VALUES ({", ".join("s." + c for c in rows_sdf.columns)}, current_timestamp())
WHEN NOT MATCHED BY SOURCE THEN DELETE
""").toPandas()
T["write_s"] = round(time.time() - t0)
print("MERGE result:", merge_res.iloc[0].to_dict(), f"({T['write_s']} s)")
all_chunks.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Quality checks on the chunk table
# MAGIC
# MAGIC **What:** chunk count, words-per-chunk distribution, empty and very short chunks, duplicate `chunk_id`s, chosen documents with zero chunks, forced splits, how many chunks have no detected section, and a few random sample chunks.
# MAGIC
# MAGIC **Why:** catches silent failures (a regex that matched nothing, a join that duplicated rows) before the embedding step.

# COMMAND ----------

rc = spark.table(CHUNKS_TABLE)
q = rc.agg(F.count("*").alias("chunks"), F.countDistinct("chunk_id").alias("distinct_ids"),
           F.countDistinct("project_id").alias("projects"),
           F.sum((F.col("n_words") < MIN_CHUNK_WORDS).cast("int")).alias("very_short"),
           F.sum((F.length(F.trim("chunk_text")) == 0).cast("int")).alias("empty"),
           F.sum(F.col("forced_split").cast("int")).alias("forced_split"),
           F.sum((F.col("section") == "Front matter").cast("int")).alias("no_section_front_matter"),
           F.expr("percentile(n_words, array(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))").alias("words_p01_p05_p25_p50_p75_p95_p99"),
           F.min("n_words").alias("min_words"), F.max("n_words").alias("max_words"),
           F.round(F.avg("n_words"), 1).alias("mean_words")).first().asDict()
q["duplicate_ids"] = q["chunks"] - q["distinct_ids"]
present = {r.project_id for r in rc.select("project_id").distinct().collect()}
q["chosen_docs_with_zero_chunks"] = len(set(chosen.project_id) - present)
q["approx_tokens_per_chunk_p50"] = round(q["words_p01_p05_p25_p50_p75_p95_p99"][3] * TOKENS_PER_WORD)
for k, v in q.items():
    print(f"{k:>36}: {v}")
assert q["duplicate_ids"] == 0 and q["empty"] == 0 and q["chosen_docs_with_zero_chunks"] == 0, "quality check failed"
print("\nmost common sections:")
print(rc.groupBy("section").count().orderBy(F.desc("count")).limit(15).toPandas().to_string(index=False))
for r in rc.orderBy(F.rand(5)).limit(4).collect():
    print(f"\n[{r.chunk_id}] {r.n_words} words, prose {r.prose_ratio}\n  {r.embed_text[:600]} ...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Embed new chunks (self-managed embeddings, incremental)
# MAGIC
# MAGIC **What:** selects the chunks whose `embedding` is still empty, sends their `embed_text` to `databricks-gte-large-en` in batches of 8 texts, one request at a time, scales each vector to length 1, and saves the vectors back into the table every 5,000 chunks (a **checkpoint**). Prints progress, throughput, throttling retries, and an estimate of the tokens used.
# MAGIC
# MAGIC **Rate limit (measured, see Step 5 of "How it was built and tested"):** this workspace's pay-per-token GTE endpoint accepts about **15 texts per second**, counting each text in a batch as one query; bigger batches do not help (a batch of 150 is rejected outright with HTTP 429 "Exceeded workspace QPS rate limit"). One request of 8 texts at a time runs right at that limit (~15 texts/s measured, a few 429s per minute), and `_invoke` backs off and retries whenever a 429 comes back. ~50,000 chunks therefore take about an hour. A *provisioned throughput* endpoint would remove the limit but bills per hour of reserved capacity — not worth it for a one-off job.
# MAGIC
# MAGIC **Why self-managed (we compute and store the vectors) instead of managed embeddings (the index computes them during sync):**
# MAGIC - the vectors are **computed once** and kept in Delta; deleting the Vector Search endpoint and recreating it later only re-indexes stored vectors — minutes, with no new embedding calls;
# MAGIC - the same vectors feed the FAISS backend, so both backends search the identical vector space;
# MAGIC - checkpoints make the step resumable: if it fails halfway, a re-run embeds only what is missing.

# COMMAND ----------

from concurrent.futures import ThreadPoolExecutor
t0 = time.time()
todo = rc.filter("embedding IS NULL").select("chunk_id", "embed_text").toPandas()
print(f"chunks needing an embedding: {len(todo):,} (already embedded: {rc.filter('embedding IS NOT NULL').count():,})")
dims = set()
for g in range(0, len(todo), EMBED_GROUP):
    part = todo.iloc[g:g + EMBED_GROUP]
    texts = part.embed_text.tolist()
    batches = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
    tg = time.time()
    with ThreadPoolExecutor(max_workers=EMBED_THREADS) as pool:
        vecs = np.vstack(list(pool.map(lambda b: embed_texts(b, batch_size=EMBED_BATCH), batches)))
    rate = len(part) / (time.time() - tg)
    dims.add(vecs.shape[1])
    upd = spark.createDataFrame(pd.DataFrame({"chunk_id": part.chunk_id.values, "embedding": [v.tolist() for v in vecs]}),
                                "chunk_id string, embedding array<float>")
    upd.createOrReplaceTempView("emb_upd")
    spark.sql(f"""MERGE INTO {CHUNKS_TABLE} t USING emb_upd s ON t.chunk_id = s.chunk_id
                  WHEN MATCHED THEN UPDATE SET t.embedding = s.embedding, t.embedding_model = '{EMBED_ENDPOINT}',
                  t.updated_at = current_timestamp()""")
    print(f"  embedded and saved {g + len(part):,}/{len(todo):,} ({time.time() - t0:.0f} s, {rate:.1f} texts/s, "
          f"throttling retries so far {CALL_STATS['retries']})")
T["embed_s"] = round(time.time() - t0)
n_tokens_est = int(todo.embed_text.str.split().str.len().sum() * TOKENS_PER_WORD) if len(todo) else 0
print(f"embedded {len(todo):,} chunks in {T['embed_s']} s; vector sizes returned: {dims or 'none (nothing new)'}; "
      f"~{n_tokens_est / 1e6:.1f} M tokens ≈ {n_tokens_est / 1e6 * 1.857:.1f} DBU")
assert not dims or dims == {EMBED_DIM}, f"unexpected embedding size {dims}"
missing = spark.table(CHUNKS_TABLE).filter(f"embedding IS NULL OR size(embedding) != {EMBED_DIM}").count()
print("rows without a valid embedding:", missing)
assert missing == 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9a. FAISS index (always built, used by the `faiss` backend)
# MAGIC
# MAGIC **What:** reads all vectors from the table (already unit length; re-normalised defensively), builds a FAISS `IndexFlatIP` (exact inner-product search; for length-1 vectors the inner product *is* the **cosine similarity**), and saves it with the ordered list of chunk ids and a small manifest to the Unity Catalog volume `gold.rag_index`.
# MAGIC
# MAGIC **Why:** an exact ("flat") index compares the question with all ~50,000 vectors — milliseconds at this size, with no approximation error. Approximate structures such as **HNSW** (hierarchical navigable small world graphs, which Vector Search uses internally) only pay off at millions of vectors. A volume makes the index a governed file that notebooks 10–12 can load without any endpoint.

# COMMAND ----------

import faiss, os, shutil
t0 = time.time()
spark.sql(f"CREATE VOLUME IF NOT EXISTS {FAISS_VOLUME} COMMENT 'FAISS index files for the RAG notebooks (built by 09_rag_index)'")
vec = spark.table(CHUNKS_TABLE).select("chunk_id", "embedding").orderBy("chunk_id").toPandas()
X = np.vstack(vec.embedding.values).astype("float32")
norms = np.linalg.norm(X, axis=1)
print(f"stored vector lengths: min {norms.min():.4f}, max {norms.max():.4f} (should be 1: embed_texts normalises; raw GTE vectors have length ~24)")
X /= norms[:, None]
fidx = faiss.IndexFlatIP(X.shape[1])
fidx.add(X)
local = "/local_disk0/tmp/rag_faiss_build"
os.makedirs(local, exist_ok=True)
faiss.write_index(fidx, f"{local}/index.faiss")
os.makedirs(FAISS_DIR, exist_ok=True)
shutil.copy(f"{local}/index.faiss", f"{FAISS_DIR}/index.faiss")
with open(f"{FAISS_DIR}/chunk_ids.json", "w") as f:
    json.dump(vec.chunk_id.tolist(), f)
table_version = spark.sql(f"DESCRIBE HISTORY {CHUNKS_TABLE} LIMIT 1").first()["version"]
with open(f"{FAISS_DIR}/manifest.json", "w") as f:
    json.dump({"n_vectors": int(fidx.ntotal), "dim": int(X.shape[1]), "metric": "inner product on unit vectors (cosine)",
               "embedding_model": EMBED_ENDPOINT, "source_table": CHUNKS_TABLE, "source_table_version": int(table_version),
               "built_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
T["faiss_s"] = round(time.time() - t0)
print(f"FAISS index: {fidx.ntotal:,} vectors x {X.shape[1]} dims -> {FAISS_DIR} "
      f"({os.path.getsize(f'{FAISS_DIR}/index.faiss') / 1e6:.0f} MB, {T['faiss_s']} s)")
del X, vec

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9b. Vector Search endpoint and Delta Sync index (only when `vector_backend = vector_search`)
# MAGIC
# MAGIC **What:**
# MAGIC 1. Creates the **endpoint** `wb-rag` (type STANDARD) if it does not exist — the serving compute that hosts indexes.
# MAGIC 2. Creates the **Delta Sync index** `gold.rag_chunks_index` if missing — an index that keeps itself in step with a Delta table by reading its Change Data Feed. It reads our stored vectors (`embedding_vector_column="embedding"`, 1,024 dimensions) and uses **TRIGGERED** sync: it updates only when we call `sync()`. If the index already exists, we call `sync()` instead.
# MAGIC 3. Polls the index status every 30 s (with a timeout) until it is online and holds every row. Measured on the first run: ~22 min of endpoint provisioning (even though the endpoint already reported ONLINE), ~2 min index provisioning, ~6 min to load 50,000 vectors.
# MAGIC
# MAGIC **Why:** STANDARD (up to ~2 million vectors per unit) is the smallest option and our ~50k vectors fit in one unit. TRIGGERED is the lightest sync mode — CONTINUOUS keeps a streaming job running to apply changes within seconds, which data that changes a few times a year does not need. `columns_to_sync` copies the metadata used for filters and display into the index, so one search call returns everything the assistant needs.

# COMMAND ----------

vs_info = {"endpoint": None, "index": None, "action": None}
if VECTOR_BACKEND == "vector_search":
    from databricks.vector_search.client import VectorSearchClient
    from datetime import timedelta
    vsc = VectorSearchClient(disable_notice=True)
    t0 = time.time()
    endpoints = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
    if VS_ENDPOINT not in endpoints:
        print(f"creating endpoint {VS_ENDPOINT} ...")
        vsc.create_endpoint_and_wait(name=VS_ENDPOINT, endpoint_type="STANDARD", timeout=timedelta(minutes=45))
    else:
        print(f"endpoint {VS_ENDPOINT} exists")
    T["endpoint_s"] = round(time.time() - t0)
    print("endpoint state:", vsc.get_endpoint(VS_ENDPOINT).get("endpoint_status", {}).get("state"), f"({T['endpoint_s']} s)")

    t0 = time.time()
    indexes = [i["name"] for i in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", [])]
    sync_cols = RETURN_COLUMNS + ["embed_text", "embedding"]
    if VS_INDEX not in indexes:
        print(f"creating Delta Sync index {VS_INDEX} ...")
        index = vsc.create_delta_sync_index(
            endpoint_name=VS_ENDPOINT, index_name=VS_INDEX, source_table_name=CHUNKS_TABLE,
            primary_key="chunk_id", pipeline_type="TRIGGERED",
            embedding_dimension=EMBED_DIM, embedding_vector_column="embedding", columns_to_sync=sync_cols)
        action = "create"
    else:
        index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
        for attempt in range(20):                # sync() is refused while an earlier update is still running
            try:
                index.sync(); break
            except Exception as e:
                print("  sync not accepted yet:", str(e)[:150]); time.sleep(30)
        action = "sync"
    expected = spark.table(CHUNKS_TABLE).count()
    last = None
    while True:
        st = index.describe().get("status", {})
        state, n_idx = st.get("detailed_state"), st.get("indexed_row_count", 0)
        if (state, n_idx) != last:
            print(f"  {time.time() - t0:5.0f} s  {state}  indexed {n_idx:,}/{expected:,}  {st.get('message', '')[:120]}")
            last = (state, n_idx)
        if st.get("ready") and state == "ONLINE_NO_PENDING_UPDATE" and n_idx == expected:
            break
        if "FAILED" in (state or ""):
            raise RuntimeError(f"index sync failed: {st}")
        if time.time() - t0 > SYNC_TIMEOUT_MIN * 60:
            raise TimeoutError(f"index not ready after {SYNC_TIMEOUT_MIN} min: {st}")
        time.sleep(30)
    T[f"index_{action}_s"] = round(time.time() - t0)
    vs_info = {"endpoint": VS_ENDPOINT, "index": VS_INDEX, "action": action, "indexed_rows": n_idx}
    print(f"index {action} finished in {T[f'index_{action}_s']} s")
else:
    print("vector_backend = faiss: no endpoint created or used")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Smoke test: the same questions on every backend and mode
# MAGIC
# MAGIC **What:** four realistic questions, each run as
# MAGIC - **FAISS** — vector only, exact cosine similarity;
# MAGIC - Vector Search **ANN** — vector only (ANN = approximate nearest neighbour);
# MAGIC - Vector Search **HYBRID** — vector search *and* keyword search, with the two ranked lists merged. FAISS has no keyword index, so it cannot do hybrid.
# MAGIC
# MAGIC One question uses a **metadata filter** (Africa regions only) to show that filters behave the same on both backends. Prints the top 5 with project, country, document type, section, score and a snippet, then how much the top-5 lists overlap.
# MAGIC
# MAGIC **Why:** a quick check by eye that retrieval returns on-topic passages, and a first look at where keyword matching helps (exact terms such as "procurement") or hurts. The systematic measurement is in `11_rag_evaluation`.

# COMMAND ----------

AFRICA = ["Eastern and Southern Africa", "Western and Central Africa", "Africa"]
QUESTIONS = [
    ("Which projects mention weak procurement capacity in fragile or conflict-affected states?", None),
    ("What are the main risks of rural road projects in Africa?", {"region": AFRICA}),
    ("How do education projects plan to measure learning outcomes of primary school students?", None),
    ("What measures mitigate macroeconomic or exchange rate risks in budget support operations?", None),
]
runs = [("faiss", "vector")] + ([("vector_search", "vector"), ("vector_search", "hybrid")] if VECTOR_BACKEND == "vector_search" else [])
smoke = []
for qtext, flt in QUESTIONS:
    print("\n" + "=" * 110 + f"\nQ: {qtext}" + (f"   filters={flt}" if flt else ""))
    for backend, mode in runs:
        t0 = time.time()
        res = search(qtext, k=5, filters=flt, mode=mode, backend=backend)
        ms = (time.time() - t0) * 1000
        print(f"\n  -- {backend} / {mode}  ({ms:.0f} ms incl. query embedding)")
        for r in res.itertuples():
            print(f"   {r.rank}. {float(r.score):.4f} {r.project_id} {str(r.country)[:18]:<18} {str(r.doc_type)[:22]:<22} "
                  f"[{str(r.section)[:50]}]\n        {r.chunk_text[:170]}")
        smoke.append({"q": qtext, "backend": backend, "mode": mode, "ms": round(ms),
                      "regions": res.region.tolist(), "chunk_ids": res.chunk_id.tolist()})
sm = pd.DataFrame(smoke)
if VECTOR_BACKEND == "vector_search":
    print()
    for qtext in sm.q.unique():
        s = sm[sm.q == qtext].set_index(["backend", "mode"]).chunk_ids
        f_, a_, h_ = s[("faiss", "vector")], s[("vector_search", "vector")], s[("vector_search", "hybrid")]
        print(f"top-5 overlap  FAISS vs ANN {len(set(f_) & set(a_))}/5 | ANN vs HYBRID {len(set(a_) & set(h_))}/5 | {qtext[:70]}")
flt_ok = all(set(x) <= set(AFRICA) for x in sm[sm.q == QUESTIONS[1][0]].regions)
print("filter check (all Africa results are in Africa regions):", flt_ok)
assert flt_ok

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading the smoke test
# MAGIC
# MAGIC Observations from the first full run (job run 677913325617981); the numbers above are from the latest run:
# MAGIC - **FAISS and Vector Search ANN return the same top 5** for every question. That is expected and is a useful check: the stored vectors have length 1, so ranking by L2 distance (Vector Search) is the same as ranking by cosine (FAISS). The scores differ only in scale: Vector Search reports `1 / (1 + squared distance)` — e.g. cosine 0.8055 → 1 / (1 + 2 − 2 × 0.8055) = 0.720.
# MAGIC - **Hybrid changes most of the list** (0–2 of 5 in common with vector-only), with mixed effect. It helps when the question has a distinctive term: for macroeconomic / exchange-rate risk in budget support it finds the Tonga and Rwanda *Program Documents*, while vector-only returns financial-management tables about "budgeting" (a near-miss on one word). It hurts when keywords match a table: for learning outcomes its first hit is a results-framework table. It also adds diversity: vector-only returned five chunks of the same Uganda roads PAD for the Africa roads question.
# MAGIC - **Filters work on both backends** (all results of the Africa question are in Africa regions).
# MAGIC - **Weak spots visible already:** some section labels are noise (table headers such as "CONTROL RISK S S", PAD template codes such as "RESULT_FRAME_TBL_ PD O"), and two-column tables flattened by PDF-to-text read as interleaved text. Retrieval still lands on the right projects and sections in most cases, but a table-aware text extraction would be the next improvement.
# MAGIC - Hybrid hit scores are on a rank-fusion scale (several 5th-ranked results score exactly 0.5), so the assistant uses the separately computed `cosine` for its "is the evidence strong enough?" rule, never the hybrid score. `11_rag_evaluation` measures all three configurations on 40 questions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Summary
# MAGIC
# MAGIC **What:** key counts, names and timings as the notebook's exit value.
# MAGIC
# MAGIC **Why:** a job run (and the person reviewing it) can see what was built without opening the notebook.

# COMMAND ----------

T["total_s"] = round(time.time() - t_start)
summary = {
    "backend": VECTOR_BACKEND, "candidates_with_docs": int(len(docs)), "raw_chunks_all_candidates": int(n_raw),
    "usable_chunks_all_candidates": int(n_good), "docs_chosen": int(len(chosen)),
    "docs_by_tier": chosen.tier.value_counts().to_dict(), "chunks": int(q["chunks"]), "projects": int(q["projects"]),
    "words_per_chunk_p50": q["words_p01_p05_p25_p50_p75_p95_p99"][3], "words_per_chunk_mean": q["mean_words"],
    "chunks_embedded_this_run": int(len(todo)), "embedding_model": EMBED_ENDPOINT, "embedding_dim": EMBED_DIM,
    "chunks_table": CHUNKS_TABLE, "faiss_dir": FAISS_DIR, "vs_endpoint": vs_info.get("endpoint"),
    "vs_index": vs_info.get("index"), "vs_action": vs_info.get("action"), "timings_s": T,
}
print(json.dumps(summary, indent=2, default=str))
dbutils.notebook.exit(json.dumps(summary, default=str))
