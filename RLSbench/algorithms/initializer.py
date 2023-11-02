import logging

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
from RLSbench.model_modifiers import MODEL_MODIFIERS_REGISTRY
from RLSbench.utils import load_one_from_dir

logger = logging.getLogger("label_shift")


from RLSbench.registry import Registry


ALGORITHMS_REGISTRY = Registry("algorithms")


def init_BN_adapt(config, datasets, dataloader, n_train_steps):
    logger.info("Initializing model...")

    assert not config.featurize
    # assert not config.pretrained

    model = initialize_model(
        model_name=config.model,
        dataset_name=config.dataset,
        num_classes=config.num_classes,
        featurize=config.featurize,
        pretrained=config.pretrained,
    )

    model.to(config.device)
    algorithm = BN_adapt(config, model)
    return algorithm


ALGORITHMS_REGISTRY.register("BN_adapt", init_BN_adapt)
ALGORITHMS_REGISTRY.register("IS-BN_adapt", init_BN_adapt)


def init_ERM(config, datasets, dataloader, n_train_steps):
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
        use_target_marginal=use_target_marginal,
    )

    return algorithm


ALGORITHMS_REGISTRY.register_multiname(
    [
        "ERM",
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
    ],
    init_ERM,
)


def initialize_algorithm(config, datasets, dataloader):
    logger.info(f"Initializing algorithm {config.algorithm} ...")

    # source_dataset = datasets["source_train"]
    # trainloader_source = dataloader["source_train"]

    # Other config
    # n_train_steps = (
    #     len(trainloader_source) * config.n_epochs // config.gradient_accumulation_steps
    # )

    initializer = ALGORITHMS_REGISTRY.get(config.algorithm)
    if initializer is None:
        raise ValueError(f"Algorithm {config.algorithm} not recognized")

    algorithm = initializer(config, datasets, dataloader, 1)  # n_train_steps)

    # TODO: Make these initializable
    # if config.algorithm in ("ERM-adv"):
    #     algorithm = ERM_Adv(
    #         config=config,
    #         dataloader=trainloader_source,
    #         loss_function=config.loss_function,
    #         n_train_steps=n_train_steps,
    #     )

    # elif config.algorithm in ("DANN", "IW-DANN", "IS-DANN"):
    #     algorithm = DANN(
    #         config=config,
    #         dataloader=trainloader_source,
    #         loss_function=config.loss_function,
    #         n_train_steps=n_train_steps,
    #         n_domains=2,
    #         **config.dann_kwargs,
    #     )

    # elif config.algorithm in ("CDANN", "IW-CDANN", "IS-CDANN"):
    #     algorithm = CDAN(
    #         config=config,
    #         dataloader=trainloader_source,
    #         loss_function=config.loss_function,
    #         n_train_steps=n_train_steps,
    #         n_domains=2,
    #         **config.cdan_kwargs,
    #     )

    # elif config.algorithm in ("FixMatch", "IS-FixMatch"):
    #     algorithm = FixMatch(
    #         config=config,
    #         dataloader=trainloader_source,
    #         loss_function=config.loss_function,
    #         n_train_steps=n_train_steps,
    #         **config.fixmatch_kwargs,
    #     )

    # elif config.algorithm in ("PseudoLabel", "IS-PseudoLabel"):
    #     algorithm = PseudoLabel(
    #         config=config,
    #         dataloader=trainloader_source,
    #         loss_function=config.loss_function,
    #         n_train_steps=n_train_steps,
    #         **config.pseudolabel_kwargs,
    #     )

    # elif config.algorithm in ("NoisyStudent", "IS-NoisyStudent"):
    #     algorithm = NoisyStudent(
    #         config=config,
    #         dataloader=trainloader_source,
    #         loss_function=config.loss_function,
    #         n_train_steps=n_train_steps,
    #         **config.noisystudent_kwargs,
    #     )

    # elif config.algorithm in ("COAL", "IW-COAL"):
    #     algorithm = COAL(
    #         config=config,
    #         dataloader=trainloader_source,
    #         loss_function=config.loss_function,
    #         n_train_steps=n_train_steps,
    #         **config.coal_kwargs,
    #     )

    # elif config.algorithm in ("SENTRY", "IW-SENTRY"):
    #     algorithm = SENTRY(
    #         config=config,
    #         dataloader=trainloader_source,
    #         loss_function=config.loss_function,
    #         n_train_steps=n_train_steps,
    #         **config.sentry_kwargs,
    #     )

    # elif config.algorithm in ("CORAL", "IS-CORAL"):
    #     algorithm = CORAL(config=config)

    # elif config.algorithm in ("BN_adapt-adv", "IS-BN_adapt-adv"):
    #     algorithm = BN_adapt_adv(config=config)

    # elif config.algorithm in ("TENT", "IS-TENT"):
    #     algorithm = TENT(config=config)

    # else:
    #     raise ValueError(f"Algorithm {config.algorithm} not recognized")

    if config.source_model_path:
        logger.info("Loading from checkpoint...")
        source_model_path = config.source_model_path
        epoch = load_one_from_dir(algorithm, source_model_path, config.device)
        logger.info(f"Loaded at epoch {epoch}")

    model_modifiers = config.model_modifier
    if model_modifiers:
        model = algorithm.model
        logger.info(f"Applying {len(model_modifiers)} model modifiers...")
        for mm_name in model_modifiers:
            mm = MODEL_MODIFIERS_REGISTRY.get(mm_name)
            model = mm(model)
        algorithm.model = model
    return algorithm
