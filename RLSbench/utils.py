import argparse
import csv
import logging
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError as e:
    pass


logger = logging.getLogger("label_shift")


def concat_input(labeled_x, unlabeled_x):
    if isinstance(labeled_x, torch.Tensor):
        x_cat = torch.cat((labeled_x, unlabeled_x), dim=0)
    else:
        raise TypeError("x must be Tensor or Batch")
    return x_cat


def cross_entropy_with_logits_loss(input, soft_target):
    """
    Implementation of CrossEntropy loss using a soft target. Extension of BCEWithLogitsLoss to MCE.
    Normally, cross entropy loss is
        \sum_j 1{j == y} -log \frac{e^{s_j}}{\sum_k e^{s_k}} = -log \frac{e^{s_y}}{\sum_k e^{s_k}}
    Here we use
        \sum_j P_j *-log \frac{e^{s_j}}{\sum_k e^{s_k}}
    where 0 <= P_j <= 1
    Does not support fancy nn.CrossEntropy options (e.g. weight, size_average, ignore_index, reductions, etc.)

    Args:
    - input (N, k): logits
    - soft_target (N, k): targets for softmax(input); likely want to use class probabilities
    Returns:
    - losses (N, 1)
    """
    return torch.sum(-soft_target * torch.nn.functional.log_softmax(input, 1), 1)


def update_average(prev_avg, prev_counts, curr_avg, curr_counts):
    denom = prev_counts + curr_counts
    if isinstance(curr_counts, torch.Tensor):
        denom += (denom == 0).float()
    elif isinstance(curr_counts, int) or isinstance(curr_counts, float):
        if denom == 0:
            return 0.0
    else:
        raise ValueError("Type of curr_counts not recognized")
    prev_weight = prev_counts / denom
    curr_weight = curr_counts / denom
    return prev_weight * prev_avg + curr_weight * curr_avg


# Taken from https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value_str = value.split("=")
            if value_str.replace("-", "").isnumeric():
                processed_val = int(value_str)
            elif value_str.replace("-", "").replace(".", "").isnumeric():
                processed_val = float(value_str)
            elif value_str in ["True", "true"]:
                processed_val = True
            elif value_str in ["False", "false"]:
                processed_val = False
            else:
                processed_val = value_str
            getattr(namespace, self.dest)[key] = processed_val


def parse_bool(v):
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def save_model(algorithm, epoch, path):
    state = {}
    state["algorithm"] = algorithm.state_dict()
    state["epoch"] = epoch
    torch.save(state, path)


class ResultsLogger:
    def __init__(self, csv_path, mode="w", use_wandb=False):
        self.path = csv_path
        self.mode = mode
        self.file = open(csv_path, mode)
        self.is_initialized = False

        # Use Weights and Biases for logging
        self.use_wandb = use_wandb
        if use_wandb:
            self.split = Path(csv_path).stem

    def setup(self, log_dict):
        columns = log_dict.keys()

        # Move epoch and batch to the front if in the log_dict
        for key in ["epoch"]:
            if key in columns:
                columns = [key] + [k for k in columns if k != key]

        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if (
            self.mode == "w"
            or (not os.path.exists(self.path))
            or os.path.getsize(self.path) == 0
        ):
            self.writer.writeheader()
        self.is_initialized = True

    def log(self, log_dict):
        if self.is_initialized is False:
            self.setup(log_dict)
        self.writer.writerow(log_dict)
        self.flush()

        if self.use_wandb:
            results = {}
            for key in log_dict:
                new_key = f"{self.split}/{key}"
                results[new_key] = log_dict[key]
            wandb.log(results)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_config(config, logger):
    for name, val in vars(config).items():
        logger.info(f'{name.replace("_"," ").capitalize()}: {val}')
    logger.info("\n")


def initialize_wandb(config):
    if config.wandb_api_key_path is not None:
        with open(config.wandb_api_key_path, "r") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()

    wandb.init(**config.wandb_kwargs)
    wandb.config.update(config)


def move_to(obj, device):
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif (
        isinstance(obj, float)
        or isinstance(obj, int)
        or isinstance(obj, str)
        or isinstance(obj, np.int_)
    ):
        return obj
    else:
        # Assume obj is a Tensor or other type
        # (like Batch, for MolPCBA) that supports .to(device)
        return obj.to(device)


def detach_and_clone(obj):
    if torch.is_tensor(obj):
        return obj.detach()
    elif isinstance(obj, dict):
        return {k: detach_and_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_and_clone(v) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        raise TypeError("Invalid type for detach_and_clone")


def collate_list(vec):
    """
    If vec is a list of Tensors, it concatenates them all along the first dimension.

    If vec is a list of lists, it joins these lists together, but does not attempt to
    recursively collate. This allows each element of the list to be, e.g., its own dict.

    If vec is a list of dicts (with the same keys in each dict), it returns a single dict
    with the same keys. For each key, it recursively collates all entries in the list.
    """
    if not isinstance(vec, list):
        raise TypeError("collate_list must take in a list")
    elem = vec[0]
    if torch.is_tensor(elem):
        return torch.cat(vec)
    elif isinstance(elem, list):
        return [obj for sublist in vec for obj in sublist]
    elif isinstance(elem, dict):
        return {k: collate_list([d[k] for d in vec]) for k in elem}
    else:
        raise TypeError("Elements of the list to collate must be tensors or dicts.")


def remove_key(key):
    """
    Returns a function that strips out a key from a dict.
    """

    def remove(d):
        if not isinstance(d, dict):
            raise TypeError("remove_key must take in a dict")
        return {k: v for (k, v) in d.items() if k != key}

    return remove


class InfiniteDataIterator:
    """
    Adapted from https://github.com/thuml/Transfer-Learning-Library

    A data iterator that will never stop producing data
    """

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            logger.info("Reached the end, resetting data loader...")
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


def load_one_from_dir(module, path, device=None):
    """
    Given a directory, loads exactly one model from it
    """
    # If path is a directory, get exactly one model, if it is a path to a model file, load that one
    if os.path.isdir(path):
        model_paths = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".pth"):
                    model_path = os.path.join(root, file)
                    model_paths.append(model_path)

        assert len(model_paths) == 1, "Exactly one model should be in the directory"
        model_path = model_paths[0]
    else:
        model_path = path

    logger.info(f"Loading model from {model_path}")

    return load(module, model_path, device=device)


def load(module, path, device=None, tries=2, delete_keys=[]):
    """
    Handles loading weights saved from this repo/model into an algorithm/model.
    Attempts to handle key mismatches between this module's state_dict and the loaded state_dict.
    Args:
        - module (torch module): module to load parameters for
        - path (str): path to .pth file
        - device: device to load tensors on
        - tries: number of times to run the match_keys() function
    """
    if device is not None:
        state = torch.load(path, map_location=device)
    else:
        state = torch.load(path)

    if "algorithm" in state:
        prev_epoch = state["epoch"]
        state = state["algorithm"]

    # Loading from a pretrained SwAV model
    elif "state_dict" in state:
        state = state["state_dict"]
        prev_epoch = None
    else:
        prev_epoch = None

    if "state_dict" in state:
        state = state["state_dict"]

    # Loading from MosaicML model
    if "state" in state:
        # TODO: Make this more general
        logger.info("MosaicML model detected")
        state = state["state"]["model"]
        state = {
            k.replace("module", "model"): v for k, v in state.items() if "module" in k
        }
        del state["model.classifier.2.weight"]
        del state["model.classifier.2.bias"]
        module.load_state_dict(state, strict=False)
        return None

    # If keys match perfectly, load_state_dict() will work
    try:
        module.load_state_dict(state)
    except:
        # Otherwise, attempt to reconcile mismatched keys and load with strict=False
        module_keys = module.state_dict().keys()
        for _ in range(tries):
            state = match_keys(state, list(module_keys))
            module.load_state_dict(state, strict=False)
            leftover_state = {
                k: v for k, v in state.items() if k in list(state.keys() - module_keys)
            }
            leftover_module_keys = module_keys - state.keys()
            if len(leftover_state) == 0 or len(leftover_module_keys) == 0:
                break
            state, module_keys = leftover_state, leftover_module_keys
        if len(module_keys - state.keys()) > 0:
            print(
                f"Some module parameters could not be found in the loaded state: {module_keys-state.keys()}"
            )

    return prev_epoch


def match_keys(d, ref):
    """
    Matches the format of keys between d (a dict) and ref (a list of keys).

    Helper function for situations where two algorithms share the same model, and we'd like to warm-start one
    algorithm with the model of another. Some algorithms (e.g. FixMatch) save the featurizer, classifier within a sequential,
    and thus the featurizer keys may look like 'model.module.0._' 'model.0._' or 'model.module.model.0._',
    and the classifier keys may look like 'model.module.1._' 'model.1._' or 'model.module.model.1._'
    while simple algorithms (e.g. ERM) use no sequential 'model._'
    """
    # hard-coded exceptions
    d = {re.sub("model.1.", "model.classifier.", k): v for k, v in d.items()}
    d = {k: v for k, v in d.items() if "pre_classifier" not in k}  # this causes errors

    # probe the proper transformation from d.keys() -> reference
    # do this by splitting d's first key on '.' until we get a string that is a strict substring of something in ref
    success = False
    probe = list(d.keys())[0].split(".")
    for i in range(len(probe)):
        probe_str = ".".join(probe[i:])
        matches = list(
            filter(
                lambda ref_k: len(ref_k) >= len(probe_str)
                and probe_str == ref_k[-len(probe_str) :],
                ref,
            )
        )
        matches = list(
            filter(lambda ref_k: not "layer" in ref_k, matches)
        )  # handle resnet probe being too simple, e.g. 'weight'
        if len(matches) == 0:
            continue
        else:
            success = True
            append = [m[: -len(probe_str)] for m in matches]
            remove = ".".join(probe[:i]) + "."
            break
    if not success:
        raise Exception("These dictionaries have irreconcilable keys")

    return_d = {}
    for a in append:
        for k, v in d.items():
            return_d[re.sub(remove, a, k)] = v

    # hard-coded exceptions
    if "model.classifier.weight" in return_d:
        return_d["model.1.weight"], return_d["model.1.bias"] = (
            return_d["model.classifier.weight"],
            return_d["model.classifier.bias"],
        )
    return return_d


def multiclass_logits_to_pred(logits, alignment_dist=None):
    """
    Takes multi-class logits of size (batch_size, ..., n_classes) and returns predictions
    by taking an argmax at the last dimension
    """
    assert logits.dim() > 1

    if alignment_dist is not None:
        return (F.softmax(logits, dim=-1) * alignment_dist[None]).argmax(-1)
    else:
        return logits.argmax(-1)


def pseudolabel_multiclass_logits(logits, confidence_threshold, alignment_dist=None):
    """
    Input:
        logits (Tensor): Multi-class logits of size (batch_size, ..., n_classes).
        confidence_threshold (float): In [0,1]
    Output:
        unlabeled_y_pred (Tensor): Filtered version of logits, discarding any rows (examples) that
                                   have no predictions with confidence above confidence_threshold.
        unlabeled_y_pseudo (Tensor): Corresponding hard-pseudo-labeled version of logits. All
                                     examples with confidence below confidence_threshold are discarded.
        pseudolabels_kept_frac (float): Fraction of examples not discarded.
        mask (Tensor): Mask used to discard predictions with confidence under the confidence threshold.
    """
    mask = torch.max(F.softmax(logits, -1), -1)[0] >= confidence_threshold
    unlabeled_y_pseudo = multiclass_logits_to_pred(logits, alignment_dist)
    unlabeled_y_pseudo = unlabeled_y_pseudo[mask]
    unlabeled_y_pred = logits[mask]
    pseudolabels_kept_frac = mask.sum() / mask.numel()  # mask is bool, so no .mean()
    return unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac, mask
