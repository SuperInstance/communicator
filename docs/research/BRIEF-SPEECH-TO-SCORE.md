# BRIEF - Speech to score (Casey via Muse, 2026-10-20) - your next lane, after the harvester

## Caseys spec, in his words as relayed
Use a speech-to-text model to translate everything said, with timestamps.
SEPARATELY measure the tones: where the notes are in the speech. That makes
a melody out of everything said, and it lives in MIDI. Then those little
notes become where the syllables are - the MIDI notes ARE the syllable
skeleton of the utterance.

## Build (this worktree): experiments/speech_to_score.py
Run it on the real free session that landed tonight:
/mnt/c/proart_capture/corpus/free_sessions/1792553040/ (4m19s, audio.wav 48k,
video + luma alongside). Pipeline:
1. TRANSCRIBE: a LOCAL Whisper-class model with word timestamps
   (faster-whisper preferred; the RTX 4050 is available; no paid APIs -
   subscriptions and local iron first). Fall back to CPU if the GPU path
   fights you, and say which you used.
2. PITCH: track F0 over the WAV (pyin/CREPE/aubio - what is installable
   locally; report the choice), segment the contour into NOTE EVENTS
   (onset, offset, median pitch -> MIDI note number), export a real .mid.
3. ALIGN: map note events to transcript words and syllables by timestamp
   overlap; syllable counts from a pronouncing dictionary if available,
   else a stated heuristic. Output an alignment table (JSON + printed):
   per syllable - start, end, MIDI note, word.
Deliverables: transcript (timestamped), session .mid, alignment JSON, and
a short report docs/research/SPEECH-TO-SCORE-REPORT.md with validation:
notes-vs-syllables count agreement, spot-checks of the melody contour
against known speech moments in the session, and honest failure modes.
Commit explicit paths on this branch, push. This composes with
free_session_analyze (face/turn segments) - do not duplicate its work.
