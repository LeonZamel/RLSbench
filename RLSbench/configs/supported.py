# See algorithms/initializer.py

from RLSbench.algorithms.initializer import ALGORITHMS_REGISTRY

algorithms = ALGORITHMS_REGISTRY.available

label_shift_adapt = ["MLLS", "true", "RLLS", "None", "baseline"]


# See transforms.py
transforms = [
    "image_base",
    "to_tensor",
    "image_resize_and_center_crop",
    "image_none",
    "rxrx1",
    "clip",
    "bert",
    "None",
]

additional_transforms = ["randaugment", "weak", "flip_crop_jitter"]
collate_functions = ["mimic_readmission", "mimic_mortality", "None"]


# See models/initializer.py
# We register all models that are added to any of the following lists of architectures
from RLSbench.models.initializer import MODEL_REGISTRY

models = MODEL_REGISTRY

# Pre-training type
pretrainining_options = ["clip", "imagenet", "swav", "rand", "bert"]

# See optimizer.py
optimizers = ["SGD", "Adam", "AdamW"]

# See scheduler.py
schedulers = [
    "linear_schedule_with_warmup",
    "cosine_schedule_with_warmup",
    "ReduceLROnPlateau",
    "StepLR",
    "FixMatchLR",
    "MultiStepLR",
    "OneCycleLR",
]

# See losses.py
losses = ["cross_entropy", "cross_entropy_logits"]

from RLSbench.model_modifiers import MODEL_MODIFIERS_REGISTRY

model_modifiers = MODEL_MODIFIERS_REGISTRY.available
