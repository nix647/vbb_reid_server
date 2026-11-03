"""Model adapters. Every adapter exposes ``model_id`` (weights file name + sha256
prefix, or the HF repo + revision) so cached embeddings of two models are never mixed."""

from .base import BodyEmbedder, FaceDetection, FaceDetector, FaceEmbedder, file_model_id
from .registry import ModelSet, load_model_set

__all__ = [
    "BodyEmbedder", "FaceDetection", "FaceDetector", "FaceEmbedder", "ModelSet",
    "file_model_id", "load_model_set",
]
