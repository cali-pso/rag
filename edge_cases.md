# CLI edge cases: results and pending fixes

Run on 2026-09-30 against the full vLLM 0.10.1 corpus, 4-core container.
The subject says the CLI "is tested with such edge cases and must never crash
with an unhandled traceback", so every **CRASH** row below would fail a defense.

Status column:
- **fixed**: fixed in the "Match subject CLI interface, honour --max_chunk_size" commit
- **open**: still to do; the proposed fix is in the right-hand column

## Results

| # | Command / input | Observed | Status | Proposed fix |
|---|---|---|---|---|
| 1 | `search "How do I enable LoRA?" --k 3` | OK, 3 sources, ~6 s (mostly model + index load) | ok | none |
| 2 | `search "" --k 5` | Returns 5 arbitrary sources (semantic neighbours of the empty string) | open | `query.strip()` empty: print a warning and return `[]` without touching the retrievers |
| 3 | `search "zzqx blorf %%% ???" --k 5` | Returns 5 semantic neighbours, no crash | ok | Acceptable. Optionally add a minimum-score cutoff |
| 4 | `search "LoRA" --k 0` / `--k -2` | 0 sources, no crash | open (neg.) | Reject `k < 0` with a clear message. `k=0` is fine |
| 5 | `search "LoRA" --k abc` | **CRASH** `TypeError: '<=' not supported between 'str' and 'int'` | open | Fire passes `abc` as `str`. Validate `isinstance(k, int) and not isinstance(k, bool)` at every command entry |
| 6 | `search "LoRA" --k 2.5` | **CRASH** `TypeError: slice indices must be integers` | open | Same as #5 |
| 7 | `search_dataset --dataset_path nope.json` | **CRASH** `FileNotFoundError` | open | Wrap the command in one error handler (see below) |
| 8 | `search_dataset --dataset_path <directory>` | **CRASH** `IsADirectoryError` | open | Same (it's an `OSError`) |
| 9 | `search_dataset` on `{bad` / an empty file | **CRASH** pydantic `ValidationError` (json_invalid) | open | Catch `pydantic.ValidationError`, print "malformed JSON in <path>" |
| 10 | `search_dataset` on `{"foo": 1}` (wrong schema) | **CRASH** `ValidationError` (missing field) | open | Same as #9 |
| 11 | `search_dataset` on `{"rag_questions": []}` | OK, writes an empty result | ok | none |
| 12 | `search_dataset` with blank questions `""`, `"   "` | OK, no crash | ok | Should go through the same empty-query path as #2 |
| 13 | `search_dataset` on an AnsweredQuestions file whose `sources` point to a missing path | **CRASH** `FileNotFoundError` in `os.path.getmtime`, raised from the `MinimalSource` validator | open | `MinimalSource` must not touch the filesystem. Compute `faiss_custom_id` in the indexer (e.g. sha256 of `path:start:end`) and keep the pydantic model pure. This also hits `evaluate`, and `answer_dataset` whenever files have moved |
| 14 | `search_dataset --save_directory /proc/nope` | **CRASH** `FileNotFoundError` from `makedirs` | open | Covered by the handler (`OSError`) |
| 15 | `search_dataset --save_directory <existing file>` | **CRASH** `FileExistsError` | open | Same |
| 16 | `search_dataset` with no args | Fire usage message, exit 2 | ok | none |
| 17 | `search_dataset --k 0` | OK, empty `retrieved_sources` | ok | none |
| 18 | `search` with no `data/processed/` | Before: **CRASH** `FileNotFoundError: Vector index files missing.` After the fix commit: no crash, but silently returns nothing | open | Check that the index files exist in `_build_retriever()`, print "run `index` first" and exit 1 |
| 19 | `search` with a corrupt `lexical_index.json` | **CRASH** `ValidationError` | open | `load_index` should catch `ValidationError` and report "index corrupted, run index again" |
| 20 | `index --max_chunk_size 0` / `5000` | Silently ignored: the value was never passed to the chunker, and the hash cache skipped every file | fixed | Value is now passed through and validated (100-2000). Changing it triggers a full rebuild because the cache stores it |
| 21 | `index --max_chunk_size 300` (found while fixing #20) | **CRASH** `ErrorChunking` on an unsplittable token (`RELEASE.md`) | fixed | Hard character split as the last-resort separator |
| 22 | `index --max_chunk_size 800` (found while fixing #20) | Invalid chunks with `first_character_index = -1` and spans up to ~20k characters. **The moulinette would reject the whole output** | fixed | `_chunk_ast_nodes` flushed an empty buffer when a node was exactly `effective_max` long |
| 23 | `evaluate --student_search_results_path … --dataset_path …` | Command did not exist (Fire usage, exit 2) | fixed | Added with the subject's signature |
| 24 | `answer "What is vLLM?" --k 3` | Wrong signature (took a results path), never saved | fixed | Now retrieve, generate, print and save |
| 25 | `evaluate` with results from another dataset (ids don't match) | **CRASH** `UnknownQuestionID` | open | Catch it and print it. Better: skip unmatched questions and report how many were skipped |
| 26 | `evaluate` on a dataset with 0 questions | predicted **CRASH** `StopIteration` (`next(iter({}))`) | open | Return an empty result / print "no questions" |
| 27 | `python -m src` (bare / `make run`) | `fire.Fire()` exposed `os`, `time`, `fire`, `BM25Retriever`… as commands. `python -m src os system …` would run anything | fixed | `fire.Fire({...})` with only the six commands |
| 28 | `answer_dataset` on 1 question | Runs, but the answer is "The context provided does not relate to the answer." The user turn of the prompt is **empty** | open | Oracle bug: `content` is a tuple (stray comma). See the explanation in the chat |
| 29 | `answer_dataset` on 100 questions | Before: all questions in one `generate()` call, padded to the longest prompt, which risks OOM on CPU | fixed | Mini-batches of 4 with a tqdm bar |

### Not tested yet (predicted from reading the code)

- `index` with a non-UTF-8 file in `data/raw`: `open(file_path, "r")` has no `encoding=` and no
  try/except, so one bad file aborts indexing. Fix: `encoding="utf-8", errors="replace"`, or skip
  the file with a warning. With `errors="replace"`, character offsets stay valid for all
  valid-UTF-8 text.
- `index` with `data/raw` missing: `os.walk` yields nothing, so it builds an empty index. It should
  print an error instead.
- Ctrl-C during `index`: the cache is saved only at the end, so a re-run starts over. That's fine,
  but `KeyboardInterrupt` should print a message rather than a traceback.

## Proposed shape of the remaining fix

A single wrapper applied to every command in `src/__main__.py`:

```python
def _cli(func: Callable[..., None]) -> Callable[..., None]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            func(*args, **kwargs)
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr)
            sys.exit(130)
        except (
            OSError,
            ValueError,
            ValidationError,
            ErrorChunking,
            UnknownQuestionID,
        ) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    return wrapper
```

Add a `_check_k(k)` helper that raises `ValueError` for anything that isn't a non-negative `int`
(`bool` excluded). Together with the `MinimalSource` change (#13) and the index-existence check
(#18), this closes every open **CRASH** row above.
