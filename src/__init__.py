"""Phase 1: the five MEASURED acoustic tracks of a spoken utterance.

Five tracks come from the waveform (PITCH, TEMPO, VELOCITY, INFLECTION,
SILENCE); four are judged by a model later (TIMBRE, INTENTION, ATTENTION,
GESTURE). This package is only the measured half.

Standard library only, deliberately: numpy/scipy/soundfile are all absent from
the interpreters available here, and forge-soniqo already showed what happens
when the analysis rate is not carried alongside the samples.
"""
