"""Server-side machine re-verification of VBB person sightings.

Phase A (offline): load a production export, embed crops with candidate
models, evaluate against human labels and an adjudication set, emit shadow
machine labels as files. No backend writes.
"""

__version__ = "0.2.3"
