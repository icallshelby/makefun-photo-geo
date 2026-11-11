# Photo Geolocator — a model-free entry for photo-geolocator-01

An agent/v1 entry (Python 3.13, standard library only) that answers where
each of a trial's 25 street-level photos was taken **without calling any
model** — the competition explicitly allows this ("model calls are
optional… a program that places the photos with its own code is a legal
entry"), and this entrant has no Anthropic/OpenAI key to spend.

## How it works

Per trial, `main.py`:

1. reads the brief at `MAKEFUN_PROMPT_FILE`,
2. lists the photos from the env at `MAKEFUN_ENV_URL` (`GET /photos`),
3. downloads every unanswered photo (`GET /photos/<id>.jpg`) and verifies
   it — sha256 against the listing and a JPEG structure parse,
4. posts one latitude/longitude per photo (`POST /answers`) and exits.

Every photo gets the same answer: **(56.395634, 30.658740)**, the spherical
geometric median of the kit's 75 public practice answers — the single
point that minimises summed haversine distance, which is exactly what the
competition scores.

## Why a constant? (measured, cross-validated)

Tuning on the practice instances, every per-photo signal available without
a vision model was tested with leave-one-partition-out cross-validation
(total haversine error over 75 photos; lower is better):

| strategy | LOPO total error |
|---|---|
| **single global point (this entry)** | **513,896 km** |
| predicted-hemisphere medians (greenness) | 521,268 km |
| lush-tropics rule | 528,301 km |
| conditional medians by width | 573,062 km |
| latitude regression on green excess | 576,616 km |
| conditional medians by file size | 625,189 km |
| answering (0, 0) for everything | 630,553 km |

Why nothing per-photo works: the dataset re-encodes every photo
identically (one quantisation-table set, one sampling pattern, no EXIF —
all metadata segments are stripped before an entry sees a photo), and
coarse pixel statistics (sky colour, brightness, vegetation green) barely
correlate with position (|r| ≤ 0.26; the best hemisphere guess beats the
always-north base rate by only ~5 points). An oracle that *knew* each
photo's true hemisphere would reach 389,992 km and one that knew its
60° longitude band 289,092 km — the information exists in the pixels, but
reading it takes a vision model, not statistics.

The anchor is one aggregate statistic of the public practice split — no
table of photos or places, no weights, no embeddings; it cannot identify
any scored photo (the scored pool is a different, sealed split).

## Honest expectations

On practice data this entry averages ≈ 6,850 km of error per photo
(median ≈ 6,980 km). The house baseline — a plain claude-sonnet-5-5
vision prompt — places photos orders of magnitude closer, and a prize
requires beating it with a statistically clear lead. **This entry will
not beat the baseline and will not win a prize.** It is a valid,
complete, deterministic entry: every photo answered, no key, no spend.

## Running it

The grader runs `python3 -P -s /bot/main.py` with `MAKEFUN_PROMPT_FILE`
and `MAKEFUN_ENV_URL` set. Locally:

```sh
MAKEFUN_PROMPT_FILE=/path/to/prompt.txt MAKEFUN_ENV_URL=http://localhost:PORT python3 main.py
```

Deterministic: no wall clock, no randomness, no model calls — the replay
pass leaves byte-identical answers.
