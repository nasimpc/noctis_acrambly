"""Field names shared by recognition observations, references, and detections."""

from enum import Enum
from pathlib import Path

# %% inference payload fields


class InferenceField(str, Enum):
    """Name fields in the inference contract.

    .. note::
        Mixing ``str`` and ``Enum`` keeps these keys compatible with Python 3.8.
    """

    IMAGE = "image"
    """Normalized observation image."""
    SCENE_IDENTITY = "scene_id"
    """Source scene identity."""
    FRAME_IDENTITY = "frame_id"
    """Source observation identity."""
    BOXES = "boxes"
    """Pixel rectangles in xyxy order."""
    MASKS = "masks"
    """Full image segmentation masks."""
    BOX_CONFIDENCE = "boxes_scores"
    """Proposal rectangle confidences."""
    MASK_CONFIDENCE = "masks_scores"
    """Proposal segmentation confidences."""
    SCORES = "scores"
    """Retained recognition scores."""
    OBJECT_IDENTITIES = "object_ids"
    """Recognized reference object identities."""
    SCORE_DISTRIBUTION = "score_distribution"
    """Recognition scores across reference objects."""
    DETECTIONS = "detections"
    """Recognized object detections."""
    PROPOSAL_RUNTIME = "runtime_proposal_stage"
    """Time spent proposing and describing image regions."""
    MATCHING_RUNTIME = "runtime_matching_stage"
    """Time spent matching image regions to references."""
    TEMPLATES = "templates"
    """Reference template images."""
    TEMPLATE_MASKS = "template_masks"
    """Reference template segmentations."""
    CLASS_DESCRIPTORS = "cls_descriptors"
    """Global image descriptors in the model's reference format."""
    PATCH_DESCRIPTORS = "patch_descriptors"
    """Local image descriptors in the model's reference format."""
    CATEGORY_IDENTITY = "category_id"
    """Saved object identities in the prediction format."""
    BOUNDING_RECTANGLE = "bbox"
    """Saved rectangles in the prediction format."""
    SEGMENTATION = "segmentation"
    """Saved segmentation masks."""
    IMAGE_IDENTITY = "image_id"
    """Saved source observation identity."""


# %% reference assets


class ReferenceFile(Enum):
    """Locate cached reference assets relative to their template directory."""

    DESCRIPTORS = Path("descriptors.pth")
    """Descriptor cache for the selected reference templates."""
