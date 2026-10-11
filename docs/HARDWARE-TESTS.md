# Hardware tests for FLE Studio 0.2

The automatic tests run against a stand-in engine, so none of them proves anything about a real GPU. This list is what
needs real hardware before a release. Each item says what to run, what to look for and where to write it down. Nothing
here has been run yet; the results column is for whoever runs it.

Every number measured must be noted with the machine (CPU, RAM, GPU, driver, OS), the engine build and the date.

| # | Test | How | Pass when | Result |
|---|---|---|---|---|
| 1 | Tuner on the FLE engine, 32 GB GPU | Engine page, stop the engine, Best for my machine, 20 minutes, 3 runs, threads and cache | finishes, saves `Tuned: ...`, the profile starts and decodes at least as fast as automatic (A/B/A says better or within noise) | |
| 2 | Tuner on stock llama.cpp, 16 GB GPU | same, with `--n-cpu-moe` and batch; llama-bench screening on | the screen step picks a value, the trial confirms or rejects it; no setting that runs out of memory takes the run down | |
| 3 | Out-of-memory handling | tune with a context of 262,144 on a small GPU | the failing settings are listed as failed and skipped; the run still ends with a profile | |
| 4 | Time budget | budget 10 minutes with a model that loads in 3 minutes | the tuner stops before a trial it cannot finish and says so | |
| 5 | KL divergence with real tools | Experiments, Quality, reference Q8_0 against a Q4 file of the same model | `llama-perplexity` runs twice then once (base file cached), mean KLD is between 0.001 and 0.5 and falls with more bits per weight | |
| 6 | KL base file size | check `data/quality/` after test 5 | about 0.3 MB per token of text at a 150k vocabulary (the text has about 1,400 tokens); note the size for the docs | |
| 7 | Knowledge probe on a real model | load a model, Ask the running model | 48 answers; a 30B+ model scores clearly above 0; interval shown | |
| 8 | A/B/A on a warming machine | A = automatic, B = the same profile, 8 runs per block | verdict "within noise"; if the A blocks differ by more than 3 % the page says drift | |
| 9 | Expert cache page with the real engine | run the FLE engine for 5 minutes of chat | the hit-rate chart fills from the `moe-cache:` lines; the page lists the two missing line types | |
| 10 | Expert cache page after the engine prints the new lines | engine build that prints `moe-layer` / `moe-hot` (docs/EXPERT-CACHE-DATA.md) | layer grid, fill and hot experts appear and match the engine's own counters | |
| 11 | gpt-oss on stock llama.cpp | Models, download gpt-oss-20b (12 GB), start with automatic settings, speed test | loads, answers, decode speed measured; note it in the catalog entry | |
| 12 | gpt-oss on the FLE engine | same file with the FLE engine | either runs (note the speed) or fails with a clear message; the catalog note says which | |
| 13 | gpt-oss-120b | needs about 64 GB RAM; tuner with the cache-less engine path | loads; note RAM and GPU memory; tuner finds a `--n-cpu-moe` | |
| 14 | Windows and Linux | e2e (`tests/e2e/run_e2e.py`) on both plus one tuner run | passes | |
| 15 | Simple view on a fresh install | fresh data folder, no profiles | simple view is the default; the Best for my machine card explains itself; nothing in it needs a profile | |

## Notes for the person running them

- Stop other GPU work first; the tuner trusts that the machine is otherwise idle. A background job that starts halfway is
  the usual cause of an A/B/A "drift" warning.
- The tuner restarts the engine many times. Keep the machine awake.
- Results go into the table above and, for speed numbers, as an Experiments result with a label (they stay in the
  data folder with the exact command).
