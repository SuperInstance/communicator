# Real-audio corpus — 8 cells, public domain only

Built 2026-10-18 per EXPERIMENTS-AUDIO.md §2 (agent-selected items with
license verification; excerpts 75 s from ~25 s in, past LibriVox intros;
16-bit PCM mono 16 kHz WAVs via `_downloads/make_excerpt.py`).

| cell | file | source item | voiced | F0 med | segs |
|---|---|---|---|---|---|
| solo narration (LOCAL-ONLY — gitignored, regenerate via measure_excerpt.py) | ../alice01_excerpt.wav | aliceinwonderland_drama_1310_librivox | 50% | 229 Hz | 91 |
| dramatic multi-voice (LOCAL-ONLY) | ../alice01_excerpt.wav (full ch.) | same | 50% | 229 Hz | 91 |
| poetry | poetry_weeds_by_the_wall_1702_librivox.wav | Cawein, *Weeds by the Wall* (collab.) | 39% | 196 Hz | 14 |
| essay/lecture | essay_perpetual_peace_librivox.wav | Kant, *Perpetual Peace* (single reader) | 31% | 90 Hz | 66 |
| short story w/ dialogue | shortstory_shortstory058_1405_librivox.wav | Short Story Collection 058 | 55% | 109 Hz | 70 |
| shakespeare | shakespeare_midsummernightsdream_1201_librivox.wav | *Midsummer Night's Dream*, full-cast dramatic (2012) | 53% | 154 Hz | 124 |
| children's | children_english_fairy_tales_joy_librivox.wav | Jacobs, *English Fairy Tales* | 53% | 181 Hz | 156 |
| oration | oration_jfks19620912.wav | JFK Rice University address 1962-09-12 — US-government work, public domain | 64% | 196 Hz | 5 |

Notes: poetry file is 20 s (short source); LibriVox items are PD
dedications; oration's 5 segments = sustained oratory with long phrases —
exactly the cadence contrast the matrix wanted. Sanity = `src.score.measure`
on each file, no degenerate outputs.
