*This project has been created as part of the 42 curriculum by rapoggi.*

# RAG Against The Machine

## Description

This is my implementation of a Retrieval-Augmented Generation (RAG) pipeline.
It allows a small LLM to answer very specific questions about a given codebase
(here, the vLLM repository) that it was never trained on. To do so, the codebase
is split into small pieces called chunks, the chunks are indexed, and when a
question comes in, the few most relevant chunks are retrieved and injected into
the model's prompt as context.

This avoids having to train the model on the codebase, keeps the prompt within
the model's context window, and is the foundation of any coding assistant that
"knows" your project.

The pipeline has four stages:

1. **Indexing**: split every file of the codebase into chunks and store them in
   a searchable index ([Chunking Strategy](#chunking-strategy)).
2. **Retrieving**: given a question, find the most relevant chunks
   ([Retrieval Method](#retrieval-method)).
3. **Augmenting**: put those chunks into the prompt, next to the question.
4. **Generating**: let Qwen3-0.6B read that prompt and write the answer.

## Instructions

A Makefile is provided, and this project uses uv to manage its virtual
environment automatically.
To install the dependencies needed by the project, run:
```bash
make install
```
or
```bash
uv sync
```

The vLLM repository must be extracted under `data/raw/`, and the datasets under
`data/datasets/`.

To build the index, run:
```bash
make index
```
or
```bash
uv run python3 -m src index [--max_chunk_size 2000]
```
The first run also downloads and quantizes the embedding model (see
[Semantic retrieval](#semantic-retrieval-embeddings-and-faiss)), so it needs an
internet connection.

To retrieve the top-k chunks for a single question, run:
```bash
uv run python3 -m src search "How do I enable LoRA?" --k 5
```

To retrieve the top-k chunks for a dataset of questions, run:
```bash
make search_[code/docs]
```
or
```bash
uv run python3 -m src search_dataset \
       --dataset_path [dataset_path] \
       --k [k] \
       --save_directory [save_directory]
```

To measure your own recall@k against a ground-truth dataset, run:
```bash
make eval_[code/docs]
```
or
```bash
uv run python3 -m src evaluate \
       --student_search_results_path [student_search_results_path] \
       --dataset_path [answered_questions_dataset_path]
```

To answer a single question, run:
```bash
uv run python3 -m src answer "How do I enable LoRA?" --k 5
```

To generate an answer for every question of a dataset previously run through
retrieval, run:
```bash
uv run python3 -m src answer_dataset \
       --student_search_results_path [student_search_results_path] \
       --save_directory [save_directory]
```
The first run downloads Qwen3-0.6B and converts it to OpenVINO, which takes a
few minutes and needs an internet connection.

To run this project as a local server and trigger indexing and answering over
HTTP, run:
```bash
make server
```
or
```bash
uv run flask --app src.rag_http_api run
```

The exposed endpoints are `/index` and `/answer`.
See [Example Usage](#example-usage) for more information.

To clean the repository of all generated files, run:
```bash
make clean
```

To run the linters (flake8 and mypy), run:
```bash
make lint
```

## Resources

#### References

- [Okapi BM25 (Wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25): the
  ranking function used for lexical retrieval.
- [Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf):
  where BM25's formula, and the role of `k1` and `b`, come from.
- [Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* (SIGIR 2009)](https://dl.acm.org/doi/10.1145/1571941.1572114):
  the paper behind the fusion step.
- [Sentence-Transformers documentation](https://sbert.net/) and the
  [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2):
  sentence embeddings.
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki): vector search.
- [OpenVINO documentation](https://docs.openvino.ai/),
  [Optimum Intel](https://huggingface.co/docs/optimum/intel/index) and
  [OpenVINO GenAI](https://openvinotoolkit.github.io/openvino.genai/): running and
  quantizing models on CPU.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B): the answering
  model.
- [Python `ast` module](https://docs.python.org/3/library/ast.html): Python code
  chunking.
- [Python Fire](https://github.com/google/python-fire),
  [pydantic](https://docs.pydantic.dev/latest/),
  [Flask](https://flask.palletsprojects.com/en/stable/) and
  [uv](https://docs.astral.sh/uv/) documentation.

#### Use of AI

I used Claude (Anthropic) as a reviewer and assistant during this project, for:

- **Code review**: auditing the project against the subject and listing what was
  left to fix.
- **Robustness**: making every CLI command handle degenerate inputs (empty
  query, `k` of the wrong type, missing or malformed files, missing index)
  without a traceback.
- **Lint**: fixing the flake8 and mypy findings.
- **Performance**: rewriting `BM25Retriever` around an inverted index, and
  measuring which chunks were worth embedding to bring indexing under 5 minutes.
- **Learning**: guidance and documentation pointers for the HTTP API, which I
  implemented myself, and explanations of BM25 and Reciprocal Rank Fusion.
- **This README**: drafting the Retrieval Method, Performance Analysis, Design
  Decisions, Challenges Faced and Example Usage sections, which I then reviewed.

Every AI suggestion was read, tested and measured on the public datasets before
being kept.

## System Architecture

The system is driven by 4 commands, two of which have a batch variant
(`search_dataset`, `answer_dataset`). These commands are implemented as
standalone functions that create the relevant objects and call their methods.
They must run in a specific order, since each one uses what the previous one
produced. Here is each command, in that order:

#### Index

`def index(max_chunk_size: int = MAX_CHUNK_SIZE_LIMIT) -> Dict[str, str | int]:`

This function does a lot of the heavy lifting. To understand how it works, you
first need to know what "the index" is. Here it means the actual data saved on
disk and reused by the rest of the pipeline, not the function. This system uses
**two** indexes: a lexical index and a vector (embeddings) index.

The **lexical index** is a list of chunks. In the code, each chunk is a
`MinimalSource` object: a piece of one of the codebase's files, whether
documentation or code. It holds the file path, the first and last character
index of the piece in that file, and the piece's text. For example:

```json
{
    "file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
    "first_character_index": 2071,
    "last_character_index": 3833,
    "content": "## Serving LoRA Adapters\n..."
}
```

This plain list is what BM25 searches through (see
[Lexical retrieval: BM25](#lexical-retrieval-bm25)).

The **vector index** is built from the lexical one: each chunk goes through a
small embedding model, which turns its text into a list of 384 numbers
representing its meaning. It is also called a semantic index, because it works
on meaning rather than on exact words. It can find a chunk that paraphrases the
question with completely different words, which would fool the lexical index
entirely (see
[Semantic retrieval](#semantic-retrieval-embeddings-and-faiss)).

Here is how the function builds them:

1. **Load the cache.** If `index()` already ran successfully, both indexes are on
   disk, along with a cache holding a SHA-256 fingerprint of every file that was
   indexed.
2. **Find what needs (re)indexing.** A cache doesn't mean there is nothing to do:
   the codebase may have changed, or the `max_chunk_size` parameter may be
   different. Each file's current fingerprint is compared with the cached one.
   With no cache, or a different `max_chunk_size`, the whole codebase is
   re-indexed. Otherwise, only the new or modified files are, and deleted files
   are removed from the indexes.
3. **Chunk.** A `LexicalIndexer` receives the list of files and its
   `create_index()` method splits each of them into chunks (see
   [Chunking Strategy](#chunking-strategy)). Every chunk gets a unique ID derived
   from its file path and character range.
4. **Embed.** The chunks are passed to a `VectorIndexer`, which encodes the
   Markdown ones with its embedding model and stores the vectors in a FAISS
   index under each chunk's ID. That ID is what allows a file's vectors to be
   removed when the file changes or disappears.
5. **Save.** On success, each indexer writes its index to `data/processed/` and
   the cache is updated.

The function returns a short statistics dictionary describing what just
happened. You only see it when the function is called through the HTTP API:

```json
{"status": "done", "time_elapsed": 34.1, "files_indexed": 1941, "chunks": 16493}
```

#### Search

`def search(query: Any, k: Any = 10) -> None:`

This function uses BM25, FAISS similarity search, and a fusion step to get the
top-k most relevant chunks for a question. `search()` itself only validates its
arguments, turns the query into an `UnansweredQuestion` pydantic model, and
calls the `HybridRetriever`'s `search()` method.

The `HybridRetriever` does the actual work. It asks each retriever
(`BM25Retriever.search()` and `VectorIndexer.search()`) for 4 times more results
than `k`, then fuses the two ranked lists into one, giving each result a score
based on its position in each list. BM25's opinion weighs twice as much as
FAISS's, because its results are more reliable, especially on code questions.
All of this is detailed in [Retrieval Method](#retrieval-method).

The final ranked list is printed with each chunk's location:

```
1. data/raw/vllm-0.10.1/docs/features/lora.md [2071:3833]
2. data/raw/vllm-0.10.1/tests/lora/test_transformers_model.py [1515:3341]
3. data/raw/vllm-0.10.1/docs/features/lora.md [0:1882]
```

`search_dataset()` does the same for every question of a dataset in one go
(embedding all the questions in a single batch), and writes the results as a
`StudentSearchResults` JSON file.

#### Evaluate

`def evaluate(student_search_results_path: Any, dataset_path: Any) -> None:`

This function, which is not exposed through the API, compares the sources
retrieved by `search_dataset()` with a ground-truth dataset and prints the
`recall@k` score for k = 10, 8, 5, 3 and 1. It works like the official
moulinette and is meant for my own testing. Recall@k is explained in
[Performance Analysis](#performance-analysis).

#### Answer

```python
def answer(
    query: Any, k: Any = 10, save_directory: Any = SINGLE_QUERY_ANSWER_DIR
) -> str:
```

This function runs the same retrieval as `search()`, then builds a prompt from
the 5 best retrieved chunks (the context) and the question, and sends it to
Qwen3-0.6B. This lets the model give an informed answer about a codebase it
has never seen. The prompt looks like this:

```
[system] Answer the question using the provided context. If the context does
         not relate to the answer, say so explicitly. Be concise and factual,
         answer in 2-3 phrases.
[user]   Context:
         [data/raw/vllm-0.10.1/docs/features/lora.md]
         ## Serving LoRA Adapters ...

         Question: How do I enable LoRA?
```

The answer is printed, saved as a `StudentSearchResultsAndAnswer` JSON file, and
returned as a `MinimalAnswer` JSON string for the HTTP API.
`answer_dataset()` does the same for every question of a `search_dataset()`
output, reusing the sources that were already retrieved.

## Chunking Strategy

As said above, chunking is handled by `LexicalIndexer`. As requested by the
subject, it has two methods, one per kind of file: `chunk_py()` and `chunk_md()`.
No chunk is ever wider than `max_chunk_size` characters (2000 by default, the
maximum the moulinette accepts).

#### Python files

```python
@validate_call
def chunk_py(
    self,
    file_path: str,
    content: Optional[str] = None,
    max_chunk_size: int = Field(lt=2001, default=2000),
) -> List[MinimalSource]:
```

This method uses Python's `ast` module to parse the file. AST (Abstract Syntax
Tree) turns source code into a tree of objects such as `ClassDef` (a class) and
`FunctionDef` (a function), each knowing on which lines it starts and ends. So
the code can be cut along its real structure instead of searching for keywords
like `def` in the text, which would be slower, more complex, and fooled by
strings or comments containing those words.

If the file can't be parsed (invalid Python), the method falls back to the
text chunker below. Otherwise, it hands the file's top-level statements to its
recursive helper, `_chunk_ast_nodes()`, which:

- **groups consecutive statements** (imports, small functions, constants...) into
  chunks as large as possible without going over `max_chunk_size`;
- **recurses into any class or function too big for one chunk**, splitting it by
  its own inner statements (methods, blocks...);
- **repeats the enclosing signature** at the top of each chunk produced by that
  recursion, so a chunk deep inside a class still says which class it belongs
  to. For example, a chunk holding the middle of a huge method starts with:

  ```python
  class LLMEngine:
      def add_request(self, request_id: str, prompt: PromptType, ...):
  ```

  followed by the actual lines of that part of the method. Without that header,
  the chunk would contain code but none of the names a question would mention.
  The header is capped to a quarter of the chunk size, so deeply nested code
  still leaves room for actual content;
- **keeps every character covered**: comments, blank lines and decorators
  between two statements are attached to the statement that follows them, so no
  part of the file is left out of the index.

#### Markdown and text files

```python
@validate_call
def chunk_md(
    self,
    file_path: str,
    content: Optional[str] = None,
    sep_index: int = 0,
    sep_list: List[str] = Field(
        min_length=1,
        default=["\n# ", "\n## ", "\n### ", "\n\n", "\n", " "],
    ),
    max_chunk_size: int = Field(lt=2001, default=2000),
    offset: int = 0,
) -> List[MinimalSource]:
```

This method relies on separators, because Markdown has no syntax tree to walk:
its structure is made of headings, paragraphs and lines. It takes a list of
separators (`sep_list`), ordered from the biggest unit to the smallest, and
works recursively:

1. Split the text on the first separator (top-level headings, `"\n# "`).
2. Merge consecutive pieces back together as long as they fit within
   `max_chunk_size`.
3. Any piece still too big is split again with the next separator
   (`"\n## "`, then `"\n### "`, then paragraphs, lines, and finally spaces).

For example, a long section is first split by its sub-headings. A sub-section
still too long is split by paragraphs, and so on. This keeps related text
together as much as possible: a chunk only cuts through a paragraph when the
paragraph alone is bigger than the limit. If a piece still doesn't fit after the
last separator (a very long URL or encoded blob with no spaces), it is cut into
fixed-size slices as a last resort.

Consecutive chunks also **overlap** by up to 150 characters, cut on a word
boundary. If the answer to a question sits right on the border between two
chunks, the overlap gives it a chance to be fully contained in one of them.

When Python parsing fails and the file falls back to this method, it uses a
separator list adapted to code instead:
`["\nclass ", "\ndef ", "\nfor ", "\n\n\n", "\n\n", "\n", " "]`.

## Retrieval Method

Retrieval is the part of the pipeline that, given a question, has to find the
handful of chunks (out of ~16 500) that actually contain the answer. It is the
part the project is graded on the most, so this is where most of my time went.

The system uses two retrievers that work in completely different ways, then
merges their opinions into a single ranked list. Here is each of them, then the
merging step.

#### Lexical retrieval: BM25

```python
class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
    def fit(self, sources: List[MinimalSource]) -> None:
    def search(self, query: str, top_k: int = 5) -> List[MinimalSource]:
```

BM25 is a "bag of words" method: it doesn't understand anything, it only counts
words. A chunk scores high if it contains the words of the question, and even
higher if those words are rare in the rest of the corpus. Despite how simple
that sounds, it is still the backbone of most search engines, and on a codebase
it is very strong, because questions about code often quote identifiers
(`trust_remote_code`, `LLMEngine`, `max_num_seqs`...) that appear word for word
in the right chunk.

##### Step 1: tokenization

Before counting words, we need to decide what a "word" is. A classic tokenizer
would split `trust_remote_code` into nothing useful, or keep `Gemma2DecoderLayer`
as one big unique word that nobody will ever type in a question.

My tokenizer extracts every identifier-like sequence (letters, digits and
underscores), lowercases it, and **also** adds its parts:

- `trust_remote_code` → `trust_remote_code`, `trust`, `remote`, `code`
- `Gemma2DecoderLayer` → `gemma2decoderlayer`, `gemma2`, `decoder`, `layer`

This way, a question that quotes the exact identifier matches the full token
(which is very rare, so very valuable, see IDF below), and a question that only
says "the gemma 2 decoder layer" still gets partial credit through the parts.
The camelCase split has to happen on the original text, before lowercasing,
since the capital letters are the only thing telling us where the boundaries are.

Code repeats the same identifiers constantly (`self`, `torch`, `hidden_states`...),
so the result of splitting each identifier is cached: the corpus has ~55 000
distinct terms for millions of occurrences, and each one is only split once.

##### Step 2: the BM25 score

For a question Q and a chunk D, the score is a sum over every word t of the
question:

```
score(D, Q) = Σ  IDF(t) × tf × (k1 + 1) / (tf + k1 × (1 − b + b × |D| / avgdl))
```

It looks scary, but it is made of three simple ideas.

**1. IDF (Inverse Document Frequency): rare words matter more.**

```
IDF(t) = ln( (N − df + 0.5) / (df + 0.5) + 1 )
```

`N` is the number of chunks and `df` the number of chunks containing the word.
With our ~16 500 chunks:

| Word | Appears in | IDF |
|---|---|---|
| `self` | ~10 000 chunks | 0.50 |
| `lora` | ~200 chunks | 4.41 |
| a very specific identifier | 3 chunks | 8.46 |

So matching one rare identifier is worth as much as matching `self` 17 times.
That is exactly the behaviour we want: common words say nothing about where
the answer is.

**2. TF saturation: repeating a word helps, but less and less.**

`tf` is how many times the word appears in the chunk. A chunk mentioning `lora`
5 times is probably more about LoRA than one mentioning it once, but not 5 times
more. The `k1` parameter (1.5 here) caps that effect. For a chunk of average
length, the term frequency part gives:

| tf | 1 | 2 | 5 | 20 | ∞ |
|---|---|---|---|---|---|
| contribution | 1.00 | 1.43 | 1.92 | 2.33 | 2.50 |

**3. Length normalization: long chunks don't win just by being long.**

A 2000-character chunk naturally contains more words than a 300-character one,
so it would match more question words by pure chance. The `|D| / avgdl` part
compares the chunk's length to the average chunk length, and `b` (0.75 here)
decides how much we penalize long chunks.

##### A worked example

Four tiny chunks, and the question **"the lora"**:

| Chunk | Text | Words |
|---|---|---|
| D1 | load the lora adapter | 4 |
| D2 | the model loads the weights | 5 |
| D3 | lora lora lora rank | 4 |
| D4 | the server starts | 3 |

`the` is in 3 chunks out of 4 (IDF 0.357, a weak clue), `lora` in 2 out of 4
(IDF 0.693, twice as good). The average length is 4 words.

| Chunk | `the` | `lora` | Total |
|---|---|---|---|
| D1 | 0.357 | 0.693 | **1.050** |
| D2 | 0.472 (twice, but long) | 0 | **0.472** |
| D3 | 0 | 1.155 (three times) | **1.155** |
| D4 | 0.402 (once, short) | 0 | **0.402** |

D3 wins because it says the good clue three times. D1, with both words but one
`lora`, comes close second. D2 has `the` twice and still loses, because `the` is
worth little and D2 is the longest chunk. Also notice that `loads` in D2 did not
count for `load`: BM25 compares exact words, which is the blind spot semantic
search covers.

##### Step 3: making it fast

My first version was the textbook one: for every question, loop over all 16 500
chunks and compute their score in pure Python. That took ~150 ms per question.

The trick is that **nothing in the score above depends on the question except
which words are summed**. For a given word and a given chunk, the whole
`IDF × tf × ... / ...` value is fixed once the corpus is known. So `fit()`
computes it once, for every (word, chunk) pair, and stores it in an *inverted
index*: for each word, the list of chunks containing it, and the precomputed
score contribution for each of them.

Searching then becomes: for each word of the question, take its list and add
its contributions to the scores of those chunks (one NumPy operation per word).
Chunks that don't share a single word with the question are never even looked
at. Only chunks with a score above 0 are sorted at the end.

Result: ~2.5 ms per question instead of ~150 ms, with rankings identical to the
textbook version on every public question (same formula, same operation order,
same tie-breaking).

#### Semantic retrieval: embeddings and FAISS

```python
class VectorIndexer:
    def create_index(self, chunks, deleted_paths, batch_size=BATCH_SIZE) -> None:
    def search(self, query: str, k: int = 10) -> List[MinimalSource]:
```

BM25's weakness is the opposite of its strength: it only sees words. If the
documentation says "serve several fine-tuned adapters at once" and the question
says "use multiple LoRA models simultaneously", BM25 finds almost nothing in
common. Semantic search fixes that.

##### What is an embedding?

An embedding model (here `all-MiniLM-L6-v2`, a small 22-million-parameter
model) reads a piece of text and outputs a list of 384 numbers: a *vector*,
which you can picture as a point in a 384-dimensional space. The model has been
trained so that texts with a similar meaning land close to each other in that
space, even when they share no words. "How do I stop the server?" and "shutting
down the engine" end up near each other, far away from "quantization kernels".

To compare a question with a chunk, we compare their vectors with the **cosine
similarity**: the cosine of the angle between the two vectors. 1 means they
point in exactly the same direction (same meaning), 0 means unrelated. All
vectors are normalized to a length of 1 when they are created, and for vectors
of length 1 the cosine is simply the *inner product* (multiply the two lists
element by element and sum), which is very cheap to compute.

##### FAISS

FAISS is a library from Meta that stores vectors and finds the closest ones to
a given vector. I use `IndexFlatIP`: "flat" means it compares the question to
every stored vector (exact search, no approximation), "IP" means it uses the
inner product. With under a thousand vectors this takes a fraction of a
millisecond, so there is no need for the approximate index types FAISS offers
for millions of vectors. It is wrapped in an `IndexIDMap`, so each vector
carries the chunk's ID and can be removed when its file changes.

##### Only the documentation is embedded

Embedding is by far the most expensive operation of the whole pipeline. Running
the model on all ~16 500 chunks took more than 6 minutes on a 4-core machine,
over the 5-minute indexing limit. So I measured what each part actually brings,
replaying the public questions with the semantic index restricted to different
subsets of the chunks:

| Semantic index contains | Chunks embedded | recall@5 docs | recall@5 code |
|---|---|---|---|
| nothing (BM25 only) | 0 | 0.780 | 0.727 |
| every chunk | 16 493 | 0.810 | 0.737 |
| Markdown chunks only | 847 | **0.830** | 0.707 |

Code questions mostly quote identifiers, which BM25 already matches exactly:
embedding 15 600 code chunks bought about 1 point of recall for ~95% of the
indexing time. Documentation questions, on the other hand, are paraphrases, and
without semantic search docs recall falls below the 80% requirement. So only the
Markdown chunks get a vector, and BM25 keeps covering everything. Indexing went
from ~7 minutes to ~35 seconds.

The model reads at most 256 tokens of each chunk (`MAX_SEQ_LENGTH`), which is
the length it was trained on. I tried 512: same recall, twice the time.

##### First load

The model is not shipped in the repository. On the first run, it is downloaded
from Hugging Face, its weights are quantized to 8-bit integers with OpenVINO
(4x smaller, faster on CPU, same results in practice), and the result is saved
under `data/models/`. Every later run loads it from there in under a second.

#### Fusion: weighted Reciprocal Rank Fusion

```python
class HybridRetriever:
    def search(self, query: str, k: int = 10) -> List[MinimalSource]:
    def search_batch(self, queries: List[str], k: int = 10) -> List[List[MinimalSource]]:
```

We now have two ranked lists that disagree, and we need one. The obvious idea,
adding the scores, doesn't work: a BM25 score can be 3 or 45 depending on the
question, while a cosine similarity is always between 0 and 1. The numbers
simply don't live on the same scale.

Reciprocal Rank Fusion ignores the scores and only looks at the **positions**
(ranks) in each list. Each chunk receives, from each list it appears in:

```
weight / (rank + constant)
```

and the chunks are sorted by the sum of what they received.

Here is a small example with the actual values (`constant` = 10, BM25 weight =
1.0, semantic weight = 0.5):

| Chunk | BM25 rank | Semantic rank | Fused score |
|---|---|---|---|
| A | 1 | 3 | 1.0/11 + 0.5/13 = **0.129** |
| C | 2 | absent | 1.0/12 = **0.083** |
| B | absent | 1 | 0.5/11 = **0.045** |

A chunk both retrievers agree on comes first. A chunk only one of them found
can still make it into the top-k, just lower.

Why these values:

- **BM25 weighs twice as much as semantic search**, because it is the more
  reliable of the two on this corpus (see the table above). Semantic search is
  there to rescue the questions BM25 misses, not to outvote a correct BM25 hit.
  Raising the semantic weight to 1.0 made code recall@5 fall from 0.707 to 0.606.
- **The constant is 10, not the usual 60.** The 60 found in the literature comes
  from web-scale search. With 60, being ranked 1st (1/61) instead of 5th (1/65)
  makes a ~6% difference, so a strong first hit barely counts more than an
  average one. With 10, it is a ~36% difference (1/11 vs 1/15).
- **Each retriever is asked for 4× more results than `k`** (at least 20), so a
  chunk ranked 15th by one retriever can still be rescued if the other one also
  liked it.

Finally, an empty or blank question returns no sources at all: any vector is
"close" to the embedding of nothing, so searching it would only return noise.

## Performance Analysis

All figures below were measured on a cloud container with 4 cores of an Intel
Xeon @ 2.80 GHz and 15 GB of RAM, on the public datasets (100 docs questions,
99 code questions), with the default `--max_chunk_size 2000`.

#### Recall@k

For each question, recall@k is the share of its correct sources found in the
top-k results. A correct source counts as found when one of our results is in
the same file and overlaps it with an IoU (intersection over union of the two
character ranges) of at least 0.05.

| Dataset | recall@1 | recall@3 | recall@5 | recall@10 | Required recall@5 |
|---|---|---|---|---|---|
| docs | 0.610 | 0.760 | **0.830** | 0.870 | 0.80 |
| code | 0.374 | 0.616 | **0.707** | 0.818 | 0.50 |

Both requirements are met. Code has the most comfortable margin. Docs is closer
to its threshold: 3 of the 100 docs reference sources live in `.txt` files.

#### Effect of the chunk size

The subject asks to report the effect of smaller chunks. Same pipeline, only
`--max_chunk_size` changes:

| max_chunk_size | Chunks | recall@5 docs | recall@5 code | recall@10 docs | recall@10 code |
|---|---|---|---|---|---|
| 2000 | 16 493 | **0.830** | **0.707** | **0.870** | **0.818** |
| 1000 | 34 819 | 0.800 | 0.687 | 0.840 | 0.758 |
| 500 | 75 579 | 0.780 | 0.606 | 0.820 | 0.687 |

Bigger chunks win on every metric. Two reasons:

- A bigger chunk covers a bigger region of the file, so it has more chances to
  overlap the reference span, and the IoU bar is low (0.05).
- Small chunks lose context: a 500-character piece of a function may contain
  the answer but none of the words of the question, which sat a few lines above.
  This is also why nested code chunks repeat their enclosing class and function
  signature (see [Chunking Strategy](#chunking-strategy)).

At 500 characters, docs recall@5 falls below the 80% requirement, so 2000 (the
maximum allowed) stays the default.

#### Indexing time (limit: 5 minutes)

| Run | Time |
|---|---|
| First ever run (downloads and quantizes the embedding model) | 44 s |
| Full rebuild, model already on disk | 21 to 36 s |
| Same first run, restricted to a single CPU core | 1 min 13 s |
| Incremental, one `.md` and one `.py` file changed | 13 s |

Chunking the whole corpus takes 5 to 10 seconds; the rest is embedding the
Markdown chunks. The single-core run is my stand-in for weak hardware: it still
leaves about 4× of margin under the limit.

#### Retrieval throughput (limit: 90 s for 200 questions)

Running `search_dataset` on both public datasets (199 questions) takes 24 s in
total, including loading the models and the index twice (once per command,
~6 s each). The search itself costs ~2.5 ms per question for BM25 plus one
batched embedding pass for all the questions.

The BM25 optimisation described in
[Lexical retrieval: BM25](#lexical-retrieval-bm25) is what made this possible:
with the original version, BM25 alone took ~150 ms per question, so ~30 s of the
budget for 200 questions before even embedding anything.

#### Answer generation

## Design Decisions

#### Two retrievers instead of one

Neither retriever is good enough alone on this corpus. BM25 is excellent when
the question shares words with the answer, which is almost always the case for
code questions (they quote function and class names), but it is blind to
paraphrases. Semantic search handles paraphrases, which documentation questions
are full of, but it is fuzzy on exact identifiers: to an embedding model,
`max_num_seqs` and `max_num_batched_tokens` look almost the same. Combining
both, with BM25 as the main signal and semantic search as the safety net, gave
the best recall on both datasets (see [Fusion](#fusion-weighted-reciprocal-rank-fusion)).

#### Fusing ranks, not scores

BM25 scores and cosine similarities live on different scales, and BM25's scale
even changes from one question to the next. Any formula mixing the raw numbers
would need constant re-tuning. Reciprocal Rank Fusion only uses positions, which
are always comparable.

#### Embedding only the documentation

This is the decision that made the 5-minute indexing limit reachable, and it
was taken from measurements rather than intuition (see
[Only the documentation is embedded](#only-the-documentation-is-embedded)).
It costs 3 points of code recall@5 (0.737 → 0.707, still far above 0.50), and
it gained 2 points of docs recall@5, the dataset closest to its threshold.

#### Structure-aware chunking

Cutting Python along its syntax tree means a chunk is a complete function, a
complete class, or a meaningful part of one, never half a function glued to
the start of the next. Repeating the enclosing signature in nested chunks
keeps the names a question would mention attached to the code. Both
directly improve what BM25 can match.

#### Chunk IDs derived from the chunk's location

Each chunk's ID is a hash of `file_path:first_character_index:last_character_index`.
The same chunk always gets the same ID, so the lexical and semantic indexes
always agree on which chunk is which, and the ID survives a reload from disk.
An earlier version mixed the file's modification date into the ID, which broke
the link between the two indexes whenever a file was touched.

#### Incremental indexing

Every indexed file's SHA-256 fingerprint is saved in a cache. On the next
`index`, only files whose fingerprint changed are re-chunked and re-embedded,
and the chunks of deleted files are removed. Re-indexing after editing one file
takes seconds instead of a full rebuild. Changing `max_chunk_size` (or the
index format) forces a full rebuild, since every existing chunk was cut
differently.

#### Models quantized and saved on first load

Both models (the embedder and Qwen3-0.6B) run through OpenVINO, Intel's
inference engine for CPUs, with their weights quantized to 8-bit integers:
4 times smaller than 32-bit floats, and faster on a CPU, with no measurable
loss in quality. Neither model is stored in the repository (the subject forbids
committing model weights): they are downloaded, converted and saved under
`data/models/` the first time they are needed.

#### A small, focused prompt

Qwen3-0.6B is a very small model. Only the 5 best chunks are given as context,
even when more are retrieved, to keep the prompt short: the more text a small
model has to read, the more likely it is to drift away from the question, and
the slower it answers. Its "thinking" mode is disabled, and the system prompt
tells it to answer in 2-3 sentences and to say so when the context doesn't
contain the answer, instead of making something up.

#### A CLI that never crashes

The subject requires that the CLI never ends on a traceback, whatever the
input. Every command validates its arguments before loading any model (Fire
passes `--k abc` as a string, `--k 2.5` as a float...), user files are read
through a single function that turns malformed JSON into a readable message,
and every command is wrapped in one error handler that prints a one-line error
and exits with a non-zero status.

## Challenges Faced

#### Indexing took longer than allowed

The first complete version needed almost 7 minutes to index the corpus on a
4-core machine, for a 5-minute limit, and slower hardware would only make it
worse. Profiling showed that chunking took under 10 seconds: everything else
was the embedding model running on 16 500 chunks. Instead of trying to make the
model faster, I measured what its output was worth for each kind of chunk, and
found that embedding code was nearly useless for recall. Embedding only the
documentation brought indexing down to ~35 seconds and slightly improved docs
recall.

#### BM25 was too slow for the throughput requirement

The textbook BM25 loop cost ~150 ms per question, which alone would have used a
third of the 90-second budget for 200 questions. Realizing that every term's
contribution to every chunk can be computed once at indexing time turned the
search into a few array additions (~2.5 ms per question), with identical
results.

#### Invalid sources that would invalidate the whole output

The moulinette rejects the entire output if a single source is wider than 2000
characters. Testing smaller chunk sizes revealed an edge case in the Python
chunker: a statement exactly as long as the limit produced a chunk starting at
index -1 and spanning up to ~20 000 characters. Another file contained a single
"word" (a long URL) larger than any chunk size, which no separator could split.
Both are now handled, the second one by cutting such pieces into fixed-size
slices as a last resort.

#### Docs recall close to the threshold

Docs recall@5 sat right at 0.80 for a long time, where any small change could
make it fail. The fusion constants (weights and RRF constant) and the choice of
what to embed were all decided by replaying the public questions and comparing
recall, which brought it to 0.83.

#### Degenerate inputs

Python Fire guesses argument types: `--k abc` arrives as a string, `--k 2.5` as
a float, a bare `--dataset_path` as `True`, and `--dataset_path 0` as the integer
0, which Python's `open()` treats as standard input, so the program would hang
waiting for keyboard input. Listing these cases one by one and validating every
argument before doing any work was what made the CLI robust.

#### Running the API with Flask

`flask --app src/rag_http_api run` failed with `ModuleNotFoundError: No module
named 'src'`. Given a file path, Flask looks for the package structure through
`__init__.py` files, and `src/` has none, so it imported the file as a
standalone module and the `from src...` imports broke. Passing the module name
instead (`--app src.rag_http_api`) makes Flask import it as part of the `src`
package, like `python -m src` does.

## Example Usage

#### Command line

Build the index:
```bash
$ uv run python3 -m src index
Loading cache...
Looking up files that need indexing...
Indexing 1941 files...
Indexed as 16493 chunks in 5.018s.
Building vector index from the Markdown chunks...
Saving results...
Built lexical + semantic index in 21.090s.
Ingestion complete! Indices saved under data/processed/
```

Search for a single question:
```bash
$ uv run python3 -m src search "How do I enable LoRA?" --k 3
Processing query: How do I enable LoRA?
1. data/raw/vllm-0.10.1/docs/features/lora.md [2071:3833]
2. data/raw/vllm-0.10.1/tests/lora/test_transformers_model.py [1515:3341]
3. data/raw/vllm-0.10.1/docs/features/lora.md [0:1882]
```

Search a whole dataset, then evaluate it:
```bash
$ make search_docs
Treating 100 questions
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json

$ make eval_docs
Evaluation Results
========================================
recall@10: 0.870  recall@8: 0.860  recall@5: 0.830  recall@3: 0.760  recall@1: 0.610
```

Answer a whole dataset from its search results:
```bash
$ uv run python3 -m src answer_dataset \
       --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
       --save_directory data/output/search_results_and_answer/UnansweredQuestions
Loaded 100 questions
Answering: 100%|██████████| 100/100
Processed 100 of 100 questions
Saved student_search_results_and_answer to data/output/search_results_and_answer/UnansweredQuestions/dataset_docs_public.json
```

Degenerate inputs end with a clear message instead of a traceback:
```bash
$ uv run python3 -m src search "LoRA" --k abc
Error: --k must be a non-negative integer, got 'abc'.

$ uv run python3 -m src search_dataset --dataset_path nope.json --save_directory out
Error: [Errno 2] No such file or directory: 'nope.json'
```

#### HTTP API

Start the server (it listens on `http://127.0.0.1:5000` by default):
```bash
make server
```

Build the index:
```bash
$ curl -X POST -d "max_chunk_size=2000" http://127.0.0.1:5000/index
{"chunks":16493,"files_indexed":1941,"status":"done","time_elapsed":21.09}
```

Answer a question from the top 5 retrieved chunks:
```bash
$ curl -X POST -d "query=How do I enable LoRA?" -d "k=5" http://127.0.0.1:5000/answer
{"question_id":"...","question":"How do I enable LoRA?","retrieved_sources":[{"file_path":"data/raw/vllm-0.10.1/docs/features/lora.md","first_character_index":2071,"last_character_index":3833,...}, ...],"answer":"..."}
```
