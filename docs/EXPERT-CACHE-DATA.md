# Expert cache data: what FLE Studio reads and what the engine must print

The **Expert cache** page of FLE Studio shows how the GPU expert cache of the FLE engine behaves. It uses nothing but
the engine's log (for a systemd service, the journal). It does not change the engine. This page lists every line it
understands. Two of the four exist in the engine today; two do not and are specified here so the engine side can add them.

All counters are **cumulative since the engine started**. The page subtracts neighbouring lines to get the hit rate of
an interval. Log verbosity must be 4 (`-lv 4`, which FLE Studio adds for the engines it starts).

| Line | Today | What the page shows |
|---|---|---|
| `MoE expert cache enabled: 40 layers x 96 slots, 1 inserts/step, 14984.0 MiB device memory` | printed | cache size (layers x slots) |
| `moe-cache: steps=512 hits=123 misses=45 hit-rate=73.2%` | printed, every 512 decode steps | hit rate of the whole run and per interval, as a chart |
| `moe-layer: steps=512 layer=7 hits=1234 misses=210 filled=88 slots=96` | **not printed yet** | one square per layer coloured by hit rate; cache fill over time |
| `moe-hot: steps=512 layer=7 top=311:812,27:790,402:655` | **not printed yet** | the most used experts of a layer |

## What the engine would have to print

1. **`moe-layer:`** once per layer, right after each `moe-cache:` line, with the same `steps` value.
   `hits` and `misses` are this layer's cumulative counts (their sums over all layers equal the totals of the
   `moe-cache:` line). `filled` is the number of slots of this layer that currently hold an expert, `slots` the number
   the layer has.
2. **`moe-hot:`** once per layer, right after its `moe-layer:` line: the (up to) 16 experts that were routed to most
   often since the start, as `expert:count` pairs sorted by count, most used first.

Cost: `layers x 2` lines per 512 decode steps, no extra work in the decode path beyond two counters per layer and
the expert-use counts the profile saving (`FLE_MOE_PROFILE`) already keeps.

A line that does not match its format is ignored. A new `MoE expert cache enabled` line, or a `steps` value that goes
backwards, starts a new run (the page clears its charts). Until the two new lines exist the page shows the overall hit
rate and says in words which lines are missing.

The stand-in engine of the test suite (`tests/fake_llama_server.py`, FLE mode) prints exactly these lines, so the page
and its tests can be tried without a GPU: `FAKE_ENGINE=fle FLE_MOE_CACHE_BUDGET_MIB=1024 FAKE_ANSWER_TOKENS=1300`.
`FAKE_CACHE_DETAIL=0` leaves the two new lines out, to see the page the way it looks with today's engine.

The parser is `src/fle_studio/cachewatch.py`; the API is `GET /api/cache` (optionally `?layer=N` for the time series of
one layer).
