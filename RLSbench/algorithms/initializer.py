import logging

from RLSbench import nn

from RLSbench.algorithms.BN_adapt import BN_adapt
from RLSbench.algorithms.BN_adapt_adv import BN_adapt_adv
from RLSbench.algorithms.CDAN import CDAN
from RLSbench.algorithms.COAL import COAL
from RLSbench.algorithms.CORAL import CORAL
from RLSbench.algorithms.DANN import DANN
from RLSbench.algorithms.ERM import ERM
from RLSbench.algorithms.ERM_Adv import ERM_Adv
from RLSbench.algorithms.fixmatch import FixMatch
from RLSbench.algorithms.noisy_student import NoisyStudent
from RLSbench.algorithms.pseudolabel import PseudoLabel
from RLSbench.algorithms.SENTRY import SENTRY
from RLSbench.algorithms.TENT import TENT


from RLSbench.losses import initialize_loss
from RLSbench.models.initializer import initialize_model
from RLSbench.models.model_utils import linear_probe

logger = logging.getLogger("label_shift")


def initialize_algorithm(config, model, datasets, dataloader):
    logger.info(f"Initializing algorithm {config.algorithm} ...")

    source_dataset = datasets["source_train"]
    trainloader_source = dataloader["source_train"]

    # Other config
    n_train_steps = (
        len(trainloader_source) * config.n_epochs // config.gradient_accumulation_steps
    )

    if config.algorithm in (
        "ERM-rand",
        "ERM-imagenet",
        "ERM-clip",
        "ERM-bert",
        "ERM-aug-rand",
        "ERM-aug-imagenet",
        "ERM-swav",
        "ERM-oracle-rand",
        "ERM-oracle-imagenet",
        "IS-ERM-rand",
        "IS-ERM-imagenet",
        "IS-ERM-clip",
        "IS-ERM-aug-rand",
        "IS-ERM-aug-imagenet",
        "IS-ERM-swav",
        "IS-ERM-oracle-rand",
        "IS-ERM-oracle-imagenet",
    ):
        logger.info("Initializing model...")

        if config.algorithm.startswith("IW"):
            use_target_marginal = True
        else:
            use_target_marginal = False

        if config.source_balanced or use_target_marginal:
            loss = initialize_loss(config.loss_function, reduction="none")
        else:
            loss = initialize_loss(config.loss_function)

        assert not config.featurize

        model = initialize_model(
            model_name=config.model,
            dataset_name=config.dataset,
            num_classes=config.num_classes,
            featurize=config.featurize,
            pretrained=config.pretrained,
            pretrained_path=config.pretrained_path,
            config=config,
        )

        if config.pretrained and "clip" in config.model:
            assert False, "Currently not supported"
            model = linear_probe(
                model,
                dataloader,
                device=config.device,
                progress_bar=config.progress_bar,
            )

        algorithm = ERM(
            config=config,
            model=model,
            loss=loss,
            n_train_steps=n_train_steps,
            use_marginal=use_target_marginal,
        )

    elif config.algorithm in ("BN_adapt", "IS-BN_adapt"):
        logger.info("Initializing model...")

        assert not config.featurize
        assert not config.pretrained

        model = initialize_model(
            model_name=config.model,
            dataset_name=config.dataset,
            num_classes=config.num_classes,
            featurize=config.featurize,
            pretrained=config.pretrained,
        )

        model.to(config.device)
        algorithm = BN_adapt(config=config)

    elif True:
        # Rework currently not supporting other methods
        raise ValueError(
            f"Algorithm {config.algorithm} not recognized or currently not supported"
        )

    elif config.algorithm in ("ERM-adv"):
        algorithm = ERM_Adv(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
        )

    elif config.algorithm in ("DANN", "IW-DANN", "IS-DANN"):
        algorithm = DANN(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            n_domains=2,
            **config.dann_kwargs,
        )

    elif config.algorithm in ("CDANN", "IW-CDANN", "IS-CDANN"):
        algorithm = CDAN(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            n_domains=2,
            **config.cdan_kwargs,
        )

    elif config.algorithm in ("FixMatch", "IS-FixMatch"):
        algorithm = FixMatch(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            **config.fixmatch_kwargs,
        )

    elif config.algorithm in ("PseudoLabel", "IS-PseudoLabel"):
        algorithm = PseudoLabel(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            **config.pseudolabel_kwargs,
        )

    elif config.algorithm in ("NoisyStudent", "IS-NoisyStudent"):
        algorithm = NoisyStudent(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            **config.noisystudent_kwargs,
        )

    elif config.algorithm in ("COAL", "IW-COAL"):
        algorithm = COAL(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            **config.coal_kwargs,
        )

    elif config.algorithm in ("SENTRY", "IW-SENTRY"):
        algorithm = SENTRY(
            config=config,
            dataloader=trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            **config.sentry_kwargs,
        )

    elif config.algorithm in ("CORAL", "IS-CORAL"):
        algorithm = CORAL(config=config)

    elif config.algorithm in ("BN_adapt-adv", "IS-BN_adapt-adv"):
        algorithm = BN_adapt_adv(config=config)

    elif config.algorithm in ("TENT", "IS-TENT"):
        algorithm = TENT(config=config)

    else:
        raise ValueError(f"Algorithm {config.algorithm} not recognized")

    return algorithm
