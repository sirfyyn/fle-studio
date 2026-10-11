# Tuning, repeated measurements and quality

How FLE Studio 0.2 finds settings and how to read its numbers.

## Best for my machine

On the Engine page (simple view). The tuner needs the engine stopped; it starts and stops the engine itself.

1. **Probe.** The starting settings (automatic) run once and the first run must generate tokens. If not, the run stops
   and says why; nothing else is tried.
2. **Search.** One parameter at a time, from the values around the current one:
   CPU threads, the GPU expert cache (FLE engine) or `--n-cpu-moe` (stock llama.cpp, GPU needed), batch size,
   optionally the context size. Every value is a full trial: the engine restarts and answers the speed requests 2 to 10
   times (default 3).
3. **Decision.** A value replaces the current best only if it is faster and the 95 % interval of the difference
   excludes 0. A difference inside the noise keeps the current value. A value that does not load (out of memory) is
   listed as failed and skipped. Context size is the exception: a larger context is kept when it is not slower.
4. **Budget.** Before each trial the tuner checks that the longest trial so far still fits into the time left; it never
   starts one it cannot finish. Budget: 10 minutes to 4 hours.
5. **Result.** Saved as the launch profile `Tuned: <model>` with the machine it was measured on. "Start with the tuned
   settings" starts it. A profile made on other hardware is not offered.

With stock llama.cpp and `llama-bench` next to the server, thread count, batch and `--n-cpu-moe` are first screened
with one `llama-bench` call each (no restarts); the best value then still has to win a normal trial.

## Median and interval

Every speed is one number per run (median over the run's requests). From 3 runs on the page shows the median with a 95 %
interval; with fewer it shows the median only. Intervals: Student's t on the mean for fewer than 8 runs, a percentile
bootstrap of the median for 8 or more.

Comparing two settings: the 95 % interval of the difference (Welch's t for fewer than 8 runs per group, bootstrap for 8
or more). "Faster" / "slower" only when the interval excludes 0, otherwise **within noise**; with fewer than 2 runs
per group, "not enough runs".

## A/B/A

A, then B, then A again, same number of runs. B is compared with both A blocks pooled. It must lie on the same side of
the first and the second A, and if the two A blocks differ significantly (drift) both single comparisons must be
significant too; otherwise the verdict is "within noise".

## Quality

* **KL divergence** (`llama-perplexity --kl-divergence`): the reference file runs once and writes its probabilities to
  a base file; the file to measure is then compared with it. 0 means the same. The base file is cached per reference,
  text and settings (about 0.3 MB per token of text for a 150k vocabulary; the built-in text has about 1,400 tokens).
  Needs `llama-perplexity` next to `llama-server`.
* **Knowledge probe:** 24 short facts in English and German (48 questions), original text in the public domain
  (CC0-1.0, `quality_data/knowledge_cc0.json`). The running model answers each; an answer counts when it contains an
  accepted answer. The score has a Wilson 95 % interval. It is a smoke test: 48 questions cannot rank two models that
  are close.
* **MKQA** can be downloaded on request (Apple's multilingual QA set). Its licence is stated inconsistently (CC BY-SA 3.0
  in the repository, CC BY 3.0 on the Hugging Face card), so FLE Studio does not ship it; it is fetched into your data
  folder after you confirm, with the attribution shown.
* Results are stored in `data/quality/`. The speed test can ask the knowledge questions after the speed requests; its
  Experiments row then shows the score and the newest KL divergence of that file next to the speed.
