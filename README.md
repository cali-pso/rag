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
environment automatically. Python 3.10 or later is required (uv downloads a
suitable interpreter if none is installed).
To install the dependencies needed by the project, run:
```bash
make install
```
or
```bash
uv sync
```

The corpus and the datasets are not shipped with the repository. Extract the
vLLM repository and the datasets so that the layout is:

```
data/raw/vllm-0.10.1/...                       # the indexed corpus
data/datasets/UnansweredQuestions/*.json       # questions to search / answer
data/datasets/AnsweredQuestions/*.json         # ground truth, for evaluate
```

For example:
```bash
unzip vllm-0.10.1.zip -d data/raw/
unzip datasets_public.zip -d data/datasets/
```

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
`--save_directory` defaults to `data/output/search_results/`.

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

To run this project as a local server and trigger indexing, searching and
answering over HTTP, run:
```bash
make server
```
or
```bash
uv run flask --app src.rag_http_api run
```

The server listens on `http://127.0.0.1:5000` and exposes three `POST`
endpoints taking form fields:

| Endpoint | Fields | Returns |
|---|---|---|
| `/index` | `max_chunk_size` | indexing statistics |
| `/search` | `query`, `k` | the top-k sources (`MinimalSearchResults`) |
| `/answer` | `query`, `k` | the answer and its sources (`MinimalAnswer`) |

Invalid requests get a JSON `{"error": ...}` with status 400. See
[Example Usage](#example-usage) and the [Bonus Part](#bonus-part).

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
- **Testing**: running the full correction process (exam scripts, moulinette,
  edge cases, single-core runs) on fresh clones and benchmarking every step;
  the robustness fixes that came out of it (prompt token budget, no answer
  without sources, interruption-safe model export, cached models and JSON
  errors in the HTTP API).
- **This README**: drafting the Retrieval Method, Performance Analysis, Design
  Decisions, Challenges Faced, Example Usage and Bonus sections, which I then
  reviewed.

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
happened (`files_indexed` and `chunks` count what was (re)indexed by this run).
You only see it when the function is called through the HTTP API:

```json
{"status": "done", "time_elapsed": 26.3, "files_indexed": 1969, "chunks": 16592}
```

#### Search

`def search(query: Any, k: Any = 10) -> Dict[str, Any]:`

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

The final ranked list is printed with each chunk's location (and returned as a
`MinimalSearchResults` dictionary for the HTTP API). For "How to configure
OpenAI server?":

```
1. data/raw/vllm-0.10.1/docs/deployment/frameworks/dstack.md [1788:3170]
2. data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md [0:1952]
3. data/raw/vllm-0.10.1/examples/online_serving/openai_transcription_client.py [0:1350]
4. data/raw/vllm-0.10.1/docs/getting_started/quickstart.md [6293:8078]
5. data/raw/vllm-0.10.1/examples/online_serving/openai_chat_completion_client_with_tools_required.py [0:566]
```

`search_dataset()` does the same for every question of a dataset, 32 questions
at a time behind a progress bar (each group of questions is embedded in a single
batch), and writes the results as a `StudentSearchResults` JSON file.

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
) -> Optional[Dict[str, Any]]:
```

This function runs the same retrieval as `search()`, then builds a prompt from
the 5 best retrieved chunks (the context) and the question, and sends it to
Qwen3-0.6B, the default (and only) answering model. This lets the model give an
informed answer about a codebase it has never seen. The prompt looks like this:

```
[system] Answer the question using the provided context. If the context does
         not relate to the answer, say so explicitly. Be concise and factual,
         answer in 2-3 phrases.
[user]   Context:
         [data/raw/vllm-0.10.1/docs/features/lora.md]
         ## Serving LoRA Adapters ...

         Question: How do I enable LoRA?
```

The prompt always fits a fixed token budget, counted with Qwen's own
tokenizer: the context is capped to 3072 tokens (sources are added best first,
the last one truncated to fit) and the question to 256 tokens. Five sources of
up to 2000 characters take about 2500 tokens, so a normal prompt is never cut;
the cap only protects the CPU from pathological inputs (an 88 000-character
question used to keep the model busy for more than 10 minutes).

When retrieval finds nothing (for example with `--k 0`), the model is not
called at all: the answer says that no relevant source was found, instead of
letting the model make one up from its own knowledge.

The answer is printed, saved as a `StudentSearchResultsAndAnswer` JSON file, and
returned as a `MinimalAnswer` dictionary for the HTTP API.
`answer_dataset()` does the same for every question of a `search_dataset()`
output, reusing the sources that were already retrieved. The model is loaded
once per process, so a running HTTP server answers successive questions
without reloading it.

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

##### Step 1b: the file path is indexed too

The text that gets tokenized is not only the chunk's content: it is the file
path followed by the content. `docs/cli/bench/throughput.md` adds the words
`docs`, `cli`, `bench`, `throughput` and `md` to every chunk of that file.

This matters because a path often says what a file is about better than its
text does. Some documentation pages hold almost nothing: `docs/cli/serve.md` is
116 characters long (a title and an include directive that the documentation
website expands later), so a question like "What is the CLI command to start
vLLM serve?" had nothing to match except the path. And code questions often
name the module they are about ("the fused MoE layer", "the metrics reader").

Only the searched text changes: the chunk itself, its character range and its
`content` field stay exactly the same, so the 2000-character limit and the
reported offsets are unaffected. Measured on the public datasets, this took
recall@5 from 0.860 to 0.900 on docs and from 0.707 to 0.818 on code. The path
is also prepended to the text given to the embedding model (see below).

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
the model on all ~16 600 chunks takes almost 7 minutes on a 4-core machine,
over the 5-minute indexing limit on its own. So I measured what each part
actually brings, replaying the public questions with the semantic index
restricted to different subsets of the chunks (current code, file path
indexed):

| Semantic index contains | Chunks embedded | Embedding time | recall@5 docs | recall@5 code |
|---|---|---|---|---|
| nothing (BM25 only) | 0 | 0 s | 0.830 | 0.828 |
| every chunk | 16 592 | 407 s | 0.880 | 0.828 |
| Markdown chunks only | 847 | 18 s | **0.900** | 0.818 |

Code questions mostly quote identifiers, which BM25 already matches exactly:
embedding the 15 700 code chunks brings nothing on code and costs 22 times
more time. Documentation questions, on the other hand, are paraphrases, and
semantic search adds 7 points of docs recall@5 on top of BM25 (before the file
path was indexed, BM25 alone even fell below the 80% requirement, at 0.780).
So only the Markdown chunks get a vector, and BM25 keeps covering everything.
Indexing went from ~7 minutes to well under a minute.

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
99 code questions), with the default `--max_chunk_size 2000`, by running the
same commands and scripts as the correction (the moulinette for recall). Times
include starting Python and loading the models, as a user would experience
them.

#### Recall@k

For each question, recall@k is the share of its correct sources found in the
top-k results. A correct source counts as found when one of our results is in
the same file and overlaps it with an IoU (intersection over union of the two
character ranges) of at least 0.05.

| Dataset | recall@1 | recall@3 | recall@5 | recall@10 | Required recall@5 |
|---|---|---|---|---|---|
| docs | 0.660 | 0.830 | **0.900** | 0.920 | 0.80 |
| code | 0.465 | 0.758 | **0.818** | 0.869 | 0.50 |

Both requirements are met with a margin: 10 points on docs, 32 points on code.
How each part of the retriever contributes is measured in
[Retrieval Method](#retrieval-method): indexing the file path (+4 points docs,
+11 points code) and semantic search on the documentation (+7 points docs over
BM25 alone).

#### Effect of the chunk size

The subject asks to report the effect of smaller chunks. Same pipeline, only
`--max_chunk_size` changes (indexing time is a full rebuild with the embedding
model already on disk):

| max_chunk_size | Chunks | Indexing | recall@5 docs | recall@5 code | recall@10 docs | recall@10 code |
|---|---|---|---|---|---|---|
| 2000 | 16 592 | 36 s | **0.900** | **0.818** | **0.920** | **0.869** |
| 1000 | 35 008 | 53 s | 0.850 | 0.768 | 0.890 | 0.859 |
| 500 | 75 996 | 84 s | 0.830 | 0.707 | 0.860 | 0.788 |

Bigger chunks win on every metric. Two reasons:

- A bigger chunk covers a bigger region of the file, so it has more chances to
  overlap the reference span, and the IoU bar is low (0.05).
- Small chunks lose context: a 500-character piece of a function may contain
  the answer but none of the words of the question, which sat a few lines above.
  This is also why nested code chunks repeat their enclosing class and function
  signature (see [Chunking Strategy](#chunking-strategy)).

Smaller chunks also cost more: more chunks to embed and to score, so indexing
takes longer. 2000 (the maximum allowed) stays the default.

#### Indexing time (limit: 5 minutes)

| Run | Time |
|---|---|
| First ever run: no index, embedding model downloaded and quantized | 44 s |
| Full rebuild, embedding model already on disk | 36 s |
| Nothing changed since the last run | 12.5 s (2 s of actual indexing) |
| One file changed | 13.4 s (1 file re-indexed) |
| First ever run, restricted to a **single CPU core** | 81 s |

Chunking the whole corpus takes 6 to 8 seconds and embedding the 847 Markdown
chunks about 18 seconds; the rest is starting Python and loading the libraries.
The single-core run is my stand-in for weak hardware: it still leaves more than
3 times the margin under the limit.

#### Retrieval throughput (limit: 90 s for 200 questions)

Running `search_dataset` on two 100-question datasets, exactly like the
correction script, takes 32 s in total (33 s on a single core), so about 16 s
per dataset, of which ~6 s is loading the index and the embedding model. A
single `search` command takes 15 s, for the same reason.

The search itself is fast: BM25 scores a question in 2.2 ms (16 592 chunks;
fitting the index at load time takes ~3.5 s), and the semantic side embeds a
whole batch of questions at once. The BM25 optimisation described in
[Lexical retrieval: BM25](#lexical-retrieval-bm25) is what made this possible:
the textbook version took ~150 ms per question, ~30 s of the budget for 200
questions before even embedding anything.

#### Answer generation

| Run | Time |
|---|---|
| First `answer` ever: Qwen3-0.6B downloaded and exported to OpenVINO int8 | 122 s |
| `answer` from the command line, model on disk | 25 to 28 s |
| `/answer` through the HTTP API, first request (models loaded) | 14 s |
| `/answer` through the HTTP API, next requests (models in memory) | 7 s |
| `answer_dataset` on the 100 public docs questions | 760 s (7.6 s per question) |

Most of a command-line `answer` is loading the models; the generation itself
takes a few seconds, which is what the HTTP API shows once they stay in memory.

On the 100 public docs questions, I judged about 85 answers satisfactory:
coherent, grounded in the retrieved sources and answering the question (for
example, the exact endpoint `/v1/load_lora_adapter`, the environment variable
`VLLM_ATTENTION_BACKEND` with its possible values, or the `--runner pooling`
flag). The failures are mostly reasoning limits of a 0.6B model rather than
retrieval failures: inverted yes/no answers ("Can vLLM serve multiple models on
a single port?" answered "Yes" although the context says no), lists cut short,
answers that restate the question, and occasionally an invented link.

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
Compared with embedding every chunk, it costs 1 point of code recall@5
(0.828 → 0.818, far above the 0.50 requirement), gains 2 points of docs
recall@5 (0.880 → 0.900), and divides the embedding time by 22.

#### Indexing the file path

The path of a file is searched together with its content (see
[Step 1b](#step-1b-the-file-path-is-indexed-too)), because it carries meaning
the text sometimes lacks: some documentation pages are little more than a
title, and code questions often name their module. It was the single most
effective change of the project (+4 points of docs recall@5, +11 points of code
recall@5) and costs nothing: only the searched text changes, never the chunk's
character range or its `content`.

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

The prompt also has a hard token budget (3072 tokens of context, 256 of
question), so no input can make generation run away, and a question for which
nothing was retrieved gets a fixed "no relevant sources" answer without calling
the model at all: with no context, anything it said would come from its own,
outdated knowledge rather than from the codebase.

#### Models kept in memory, rebuilt when the index changes

A CLI command runs once and exits, but the HTTP server lives on. Within one
process, the retriever (the BM25 index and the embedding model) and Qwen3-0.6B
are loaded once and reused. The retriever remembers the modification times of
the index files it was built from and is rebuilt automatically when they
change, so an `/index` request is picked up by the next `/search` without
restarting the server. Requests are served one at a time, since they share
these models and `/index` rewrites the files they read.

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
recall, which brought it to 0.83. Indexing `.txt` files (some reference sources
are text files) brought it to 0.86. The real fix came from looking at the
questions that still failed: several asked for a CLI command whose
documentation page is almost empty, and only its path said what it was about.
Indexing the file path brought docs recall@5 to 0.90 and code recall@5 from
0.707 to 0.818.

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
Indexing 1969 files...
Indexed as 16592 chunks in 7.379s.
Building vector index from the Markdown chunks...
Saving results...
Built lexical + semantic index in 26.346s.
Ingestion complete! Indices saved under data/processed/
```

Search for a single question:
```bash
$ uv run python3 -m src search "How to configure OpenAI server?" --k 3
Processing query: How to configure OpenAI server?
1. data/raw/vllm-0.10.1/docs/deployment/frameworks/dstack.md [1788:3170]
2. data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md [0:1952]
3. data/raw/vllm-0.10.1/examples/online_serving/openai_transcription_client.py [0:1350]
```

Search a whole dataset, then evaluate it:
```bash
$ make search_docs
Treating 100 questions
Searching: 100%|██████████| 100/100
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json

$ make eval_docs
Evaluation Results
========================================
recall@10: 0.920  recall@8: 0.910  recall@5: 0.900  recall@3: 0.830  recall@1: 0.660
```

Answer a single question:
```bash
$ uv run python3 -m src answer "How do you specify LoRA modules when starting a vLLM server?" --k 10
Processing query: How do you specify LoRA modules when starting a vLLM server?
To specify LoRA modules when starting a vLLM server, use the `--lora-modules` command line argument with the `name` and `path` of each LoRA module.
Saved answer to data/output/search_results_and_answer/single_query/single_query.json
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

$ uv run python3 -m src search_dataset --dataset_path /nonexistent/dataset.json
Error: [Errno 2] No such file or directory: '/nonexistent/dataset.json'

$ uv run python3 -m src answer "What is vLLM?" --k 0
Processing query: What is vLLM?
No relevant sources were found in the indexed codebase for this question, so it cannot be answered from it.
Saved answer to data/output/search_results_and_answer/single_query/single_query.json
```

#### HTTP API

Start the server (it listens on `http://127.0.0.1:5000`):
```bash
make server
```

Update the index (here nothing changed since the last run):
```bash
$ curl -X POST -d "max_chunk_size=2000" http://127.0.0.1:5000/index
{"chunks":0,"files_indexed":0,"status":"done","time_elapsed":1.52}
```

Search (sources shortened here):
```bash
$ curl -X POST -d "query=How do I enable LoRA?" -d "k=3" http://127.0.0.1:5000/search
{"question":"How do I enable LoRA?","question_id":"e9c0fb6c-...","retrieved_sources":[{"content":"\n## Serving LoRA Adapters\n\nLoRA adapters can be used ...","file_path":"data/raw/vllm-0.10.1/docs/features/lora.md", ...}, ...]}
```

Answer from the top 5 retrieved chunks:
```bash
$ curl -X POST -d "query=What is PagedAttention?" -d "k=5" http://127.0.0.1:5000/answer
{"answer":"PagedAttention is a type of attention mechanism used in the vLLM library ...","question":"What is PagedAttention?","question_id":"...","retrieved_sources":[...]}
```

Invalid requests get a JSON error with status 400:
```bash
$ curl -X POST -d "query=What is vLLM?" -d "k=abc" http://127.0.0.1:5000/answer
{"error":"'k' must be an integer, got 'abc'"}
```

## Bonus Part

All five bonuses are implemented. Each one can be demonstrated with the
commands below (figures measured on the same machine as the
[Performance Analysis](#performance-analysis)).

#### 1. Semantic embeddings

The Markdown chunks are embedded with `all-MiniLM-L6-v2`, a 22-million
parameter CPU model run by OpenVINO with int8 weights, and searched with a
FAISS inner-product index (see
[Semantic retrieval](#semantic-retrieval-embeddings-and-faiss)). Embedding the
847 chunks takes about 18 seconds.

Demo: a question that shares no useful keyword with its answer.
```bash
uv run python3 -m src search "How do I serve a model to many users over the network?" --k 5
```
BM25 alone finds nothing relevant in its top 5; with semantic search,
`docs/serving/openai_compatible_server.md` comes second. Over the docs dataset,
semantic search adds 7 points of recall@5 (0.830 → 0.900).

#### 2. Hybrid retrieval

The BM25 and FAISS rankings are fused with weighted Reciprocal Rank Fusion (see
[Fusion](#fusion-weighted-reciprocal-rank-fusion)); every `search`,
`search_dataset` and `answer` uses it.

| Retrieval | recall@5 docs | recall@5 code |
|---|---|---|
| BM25 only | 0.830 | 0.828 |
| Hybrid (BM25 + semantic) | **0.900** | 0.818 |

#### 3. Incremental indexing

Each indexed file's SHA-256 fingerprint is cached; `index` only re-chunks and
re-embeds the files whose fingerprint changed, and drops the chunks of deleted
files.

```bash
uv run python3 -m src index      # nothing changed: "Indexing 0 files..."
echo "a new line" >> data/raw/vllm-0.10.1/docs/features/lora.md
uv run python3 -m src index      # "Indexing 1 files..."
```

| Run | Work done | Time |
|---|---|---|
| Full rebuild | 1969 files, 16 592 chunks | 36 s |
| Nothing changed | 0 files | 12.5 s (2 s of indexing, the rest is start-up) |
| One file changed | 1 file | 13.4 s (2.4 s of indexing) |

#### 4. Caching

Three levels of caching remove repeated work:

- **The index is cached on disk** (`data/processed/`): a cold start loads it
  instead of rebuilding it, and incremental indexing (bonus 3) builds on it.
- **Both models are exported once** to `data/models/`: the quantized embedder
  (about 10 s the first time) and Qwen3-0.6B (122 s the first time, download
  included). Every later run loads them from disk, without any network access.
- **Inside a running process**, the retriever and Qwen3-0.6B stay in memory, and
  the retriever is rebuilt only when the index files change on disk. Through
  the HTTP API, the first `/answer` takes 14 s (models loaded) and the next ones
  7 s.

#### 5. Local HTTP API

`make server` starts a Flask server on `http://127.0.0.1:5000` with three
`POST` endpoints: `/index`, `/search` and `/answer` (see
[Instructions](#instructions) and the [HTTP API examples](#http-api)). Results
are JSON documents following the project's pydantic models, and invalid
requests get a JSON error with status 400 instead of a crash.

```bash
curl -X POST -d "query=How do I enable LoRA?" -d "k=3" http://127.0.0.1:5000/search
curl -X POST -d "query=How do I enable LoRA?" -d "k=5" http://127.0.0.1:5000/answer
```
