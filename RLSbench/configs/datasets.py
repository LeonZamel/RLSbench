dataset_defaults = {
    "camelyon": {
        "num_classes": 2,
        "model": "densenet121",
        "optimizer": "SGD",
        "batch_size": 96,
        "optimizer_kwargs": {"momentum": 0.9},
        "lr": 0.003,
        "weight_decay": 0.01,
        "scheduler": None,
        "n_epochs": 10,
        "pretrained": False,
        "pretrain_type": "rand",
        "collate_function": "None",
        "transform": "image_base",
        "resize_resolution": 96,
        "target_resolution": 96,
        "default_normalization": True,
        "dann_kwargs": {
            "featurizer_lr": 0.003,
            "classifier_lr": 0.003,
            "discriminator_lr": 0.003,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 0.003,
            "classifier_lr": 0.003,
            "discriminator_lr": 0.003,
            "penalty_weight": 1.0,
        },
        "coal_kwargs": {
            "featurizer_lr": 0.003,
            "classifier_lr": 0.003,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "iwildcam": {
        "num_classes": 182,
        "model": "resenet50",
        "optimizer": "Adam",
        "batch_size": 24,
        "lr": 4.5e-5,
        "weight_decay": 0.0,
        "scheduler": None,
        "n_epochs": 12,
        "pretrained": True,
        "collate_function": "None",
        "pretrain_type": "imagenet",
        "transform": "image_base",
        "resize_resolution": 448,
        "target_resolution": 448,
        "default_normalization": True,
        "dann_kwargs": {
            "featurizer_lr": 4.5e-5,
            "classifier_lr": 4.5e-4,
            "discriminator_lr": 4.5e-4,
            "penalty_weight": 0.1,
        },
        "cdan_kwargs": {
            "featurizer_lr": 4.5e-5,
            "classifier_lr": 4.5e-4,
            "discriminator_lr": 4.5e-4,
            "penalty_weight": 0.1,
        },
        "coal_kwargs": {
            "featurizer_lr": 4.5e-5,
            "classifier_lr": 4.5e-4,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "fmow": {
        "num_classes": 62,
        "model": "densenet121",
        "optimizer": "Adam",
        "batch_size": 64,
        "lr": 0.0001,
        "weight_decay": 0.0,
        "scheduler": "StepLR",
        "scheduler_kwargs": {"gamma": 0.96, "step_size": 1},
        "collate_function": "None",
        "n_epochs": 30,
        "pretrained": True,
        "pretrain_type": "imagenet",
        "transform": "image_base",
        "resize_resolution": 224,
        "target_resolution": 224,
        "default_normalization": True,
        "dann_kwargs": {
            "featurizer_lr": 0.00001,
            "classifier_lr": 0.0001,
            "discriminator_lr": 0.0001,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 0.00001,
            "classifier_lr": 0.0001,
            "discriminator_lr": 0.0001,
            "penalty_weight": 1.0,
        },
        "coal_kwargs": {
            "featurizer_lr": 0.00001,
            "classifier_lr": 0.0001,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "fmow_region": {
        "num_classes": 62,
        "model": "densenet121",
        "optimizer": "Adam",
        "batch_size": 64,
        "lr": 0.0001,
        "weight_decay": 0.0,
        "scheduler": "StepLR",
        "scheduler_kwargs": {"gamma": 0.96, "step_size": 1},
        "collate_function": "None",
        "n_epochs": 30,
        "pretrained": True,
        "pretrain_type": "imagenet",
        "transform": "image_base",
        "resize_resolution": 224,
        "target_resolution": 224,
        "default_normalization": True,
        "dann_kwargs": {
            "featurizer_lr": 0.00001,
            "classifier_lr": 0.0001,
            "discriminator_lr": 0.0001,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 0.00001,
            "classifier_lr": 0.0001,
            "discriminator_lr": 0.0001,
            "penalty_weight": 1.0,
        },
        "coal_kwargs": {
            "featurizer_lr": 0.00001,
            "classifier_lr": 0.0001,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "cifar10": {
        "num_classes": 10,
        "model": "resnet18",
        "optimizer": "SGD",
        "optimizer_kwargs": {"momentum": 0.9},
        "batch_size": 200,
        "n_epochs": 50,
        "lr": 0.01,
        "weight_decay": 0.0001,
        "collate_function": "None",
        "scheduler": "MultiStepLR",
        "scheduler_kwargs": {"milestones": [25, 40], "gamma": 0.1},
        "pretrained": True,
        "pretrain_type": "imagenet",
        "pretrained_path": "./pretrained_models/resnet18_imagenet32.pt",
        "transform": "image_base",
        "resize_resolution": 32,
        "target_resolution": 32,
        "default_normalization": True,
        # 'mean': [0.4914, 0.4822, 0.4465],
        # 'std': [0.2023, 0.1994, 0.2010],
        "dann_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
            "discriminator_lr": 0.01,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
            "discriminator_lr": 0.01,
            "penalty_weight": 1.0,
        },
        "coal_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "cifar100": {
        "num_classes": 100,
        "model": "resnet18",
        "optimizer": "SGD",
        "optimizer_kwargs": {"momentum": 0.9},
        "batch_size": 200,
        "n_epochs": 50,
        "lr": 0.01,
        "weight_decay": 0.0001,
        "collate_function": "None",
        "scheduler": "MultiStepLR",
        "scheduler_kwargs": {"milestones": [25, 40], "gamma": 0.1},
        "pretrained": True,
        "pretrain_type": "imagenet",
        "transform": "image_base",
        "pretrained_path": "./pretrained_models/resnet18_imagenet32.pt",
        "resize_resolution": 32,
        "target_resolution": 32,
        "default_normalization": True,
        # 'mean': [0.5074, 0.4867, 0.4411],
        # 'std': [0.2011, 0.1987, 0.2025],
        "dann_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
            "discriminator_lr": 0.01,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
            "discriminator_lr": 0.01,
            "penalty_weight": 1.0,
        },
        "coal_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "domainnet": {
        "num_classes": 345,
        "model": "resnet50",
        "optimizer": "SGD",
        "optimizer_kwargs": {"momentum": 0.9},
        "batch_size": 96,
        "n_epochs": 15,
        "lr": 0.01,
        "weight_decay": 0.0001,
        # 'scheduler': None,
        "collate_function": "None",
        "scheduler": "StepLR",
        "scheduler_kwargs": {"gamma": 0.96, "step_size": 1},
        "pretrained": True,
        "pretrain_type": "imagenet",
        "transform": "image_base",
        "resize_resolution": 256,
        "target_resolution": 224,
        "default_normalization": True,
        "dann_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
            "discriminator_lr": 0.01,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
            "discriminator_lr": 0.01,
            "penalty_weight": 1.0,
        },
        "coal_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "officehome": {
        "num_classes": 65,
        "model": "resnet50",
        "optimizer": "SGD",
        "optimizer_kwargs": {"momentum": 0.9},
        "batch_size": 96,
        "n_epochs": 50,
        "lr": 0.01,
        "weight_decay": 0.0001,
        # 'scheduler': None,
        "collate_function": "None",
        "scheduler": "StepLR",
        "scheduler_kwargs": {"gamma": 0.96, "step_size": 1},
        "pretrained": True,
        "pretrain_type": "imagenet",
        "transform": "image_base",
        "resize_resolution": 256,
        "target_resolution": 224,
        "default_normalization": True,
        "dann_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
            "discriminator_lr": 0.01,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
            "discriminator_lr": 0.01,
            "penalty_weight": 1.0,
        },
        "coal_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "visda": {
        "num_classes": 12,
        "model": "resnet50",
        "optimizer": "SGD",
        "optimizer_kwargs": {"momentum": 0.9},
        "batch_size": 96,
        "n_epochs": 10,
        "lr": 0.01,
        "weight_decay": 0.0005,
        # 'scheduler': None,
        "collate_function": "None",
        "scheduler": "StepLR",
        "scheduler_kwargs": {"gamma": 0.96, "step_size": 1},
        "pretrained": True,
        "pretrain_type": "imagenet",
        "transform": "image_base",
        "resize_resolution": 256,
        "target_resolution": 224,
        "default_normalization": True,
        "dann_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
            "discriminator_lr": 0.01,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
            "discriminator_lr": 0.01,
            "penalty_weight": 1.0,
        },
        "coal_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "entity13": {
        "num_classes": 13,
        "model": "resnet18",
        "optimizer": "SGD",
        "optimizer_kwargs": {"momentum": 0.9},
        "batch_size": 256,
        "n_epochs": 40,
        "lr": 0.2,
        "collate_function": "None",
        "weight_decay": 0.00005,
        "scheduler": "linear_schedule_with_warmup",
        "scheduler_kwargs": {"warmup_frac": 0.05},
        "pretrained": False,
        "pretrain_type": "rand",
        "transform": "image_base",
        "resize_resolution": 256,
        "target_resolution": 224,
        "default_normalization": True,
        "dann_kwargs": {
            "featurizer_lr": 0.2,
            "classifier_lr": 0.2,
            "discriminator_lr": 0.2,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 0.2,
            "classifier_lr": 0.2,
            "discriminator_lr": 0.2,
            "penalty_weight": 1.0,
        },
        "coal_kwargs": {
            "featurizer_lr": 0.2,
            "classifier_lr": 0.2,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "entity30": {
        "num_classes": 30,
        "model": "resnet18",
        "optimizer": "SGD",
        "optimizer_kwargs": {"momentum": 0.9},
        "batch_size": 256,
        "n_epochs": 40,
        "lr": 0.2,
        "collate_function": "None",
        "weight_decay": 0.00005,
        "scheduler": "linear_schedule_with_warmup",
        "scheduler_kwargs": {"warmup_frac": 0.05},
        "pretrained": False,
        "pretrain_type": "rand",
        "transform": "image_base",
        "resize_resolution": 256,
        "target_resolution": 224,
        "default_normalization": True,
        "dann_kwargs": {
            "featurizer_lr": 0.2,
            "classifier_lr": 0.2,
            "discriminator_lr": 0.2,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 0.2,
            "classifier_lr": 0.2,
            "discriminator_lr": 0.2,
            "penalty_weight": 1.0,
        },
        "coal_kwargs": {
            "featurizer_lr": 0.2,
            "classifier_lr": 0.2,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "living17": {
        "num_classes": 17,
        "model": "resnet18",
        "optimizer": "SGD",
        "optimizer_kwargs": {"momentum": 0.9},
        "batch_size": 256,
        "n_epochs": 40,
        "lr": 0.2,
        "weight_decay": 0.00005,
        "collate_function": "None",
        "scheduler": "linear_schedule_with_warmup",
        "scheduler_kwargs": {"warmup_frac": 0.05},
        "pretrained": False,
        "pretrain_type": "rand",
        "transform": "image_base",
        "resize_resolution": 256,
        "target_resolution": 224,
        "default_normalization": True,
        "dann_kwargs": {
            "featurizer_lr": 0.2,
            "classifier_lr": 0.2,
            "discriminator_lr": 0.2,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 0.2,
            "classifier_lr": 0.2,
            "discriminator_lr": 0.2,
            "penalty_weight": 1.0,
        },
        "coal_kwargs": {
            "featurizer_lr": 0.2,
            "classifier_lr": 0.2,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "nonliving26": {
        "num_classes": 26,
        "model": "resnet18",
        "optimizer": "SGD",
        "optimizer_kwargs": {"momentum": 0.9},
        "batch_size": 256,
        "n_epochs": 40,
        "lr": 0.2,
        "weight_decay": 0.00005,
        "collate_function": "None",
        "scheduler": "linear_schedule_with_warmup",
        "scheduler_kwargs": {"warmup_frac": 0.05},
        "pretrained": False,
        "pretrain_type": "rand",
        "transform": "image_base",
        "resize_resolution": 256,
        "target_resolution": 224,
        "default_normalization": True,
        "dann_kwargs": {
            "featurizer_lr": 0.2,
            "classifier_lr": 0.2,
            "discriminator_lr": 0.2,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 0.2,
            "classifier_lr": 0.2,
            "discriminator_lr": 0.2,
            "penalty_weight": 1.0,
        },
        "coal_kwargs": {
            "featurizer_lr": 0.2,
            "classifier_lr": 0.2,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "mimic_readmission": {
        "num_classes": 2,
        "model": "mimic_network",
        "optimizer": "Adam",
        "batch_size": 128,
        "lr": 5e-4,
        "weight_decay": 0.0,
        "collate_function": "mimic_readmission",
        "scheduler": None,
        # 'scheduler_kwargs': {'gamma': 0.96, 'step_size':1},
        "n_epochs": 100,
        "pretrained": False,
        "pretrain_type": "rand",
        "transform": "None",
        "default_normalization": False,
        "dann_kwargs": {
            "featurizer_lr": 5e-4,
            "classifier_lr": 5e-4,
            "discriminator_lr": 5e-4,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 5e-4,
            "classifier_lr": 5e-4,
            "discriminator_lr": 5e-4,
            "penalty_weight": 1.0,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "retiring_adult": {
        "num_classes": 2,
        "model": "MLP",
        "optimizer": "Adam",
        "batch_size": 200,
        "lr": 0.01,
        "weight_decay": 0.0001,
        "collate_function": "None",
        "scheduler": None,
        # 'scheduler_kwargs': {'gamma': 0.96, 'step_size':1},
        "n_epochs": 50,
        "pretrained": False,
        "pretrain_type": "rand",
        "transform": "None",
        "default_normalization": False,
        "dann_kwargs": {
            "featurizer_lr": 5e-4,
            "classifier_lr": 5e-4,
            "discriminator_lr": 5e-4,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 5e-4,
            "classifier_lr": 5e-4,
            "discriminator_lr": 5e-4,
            "penalty_weight": 1.0,
        },
        "evaluate_every": 1,
    },
    "civilcomments": {
        "num_classes": 2,
        "model": "distilbert-base-uncased",
        "optimizer": "AdamW",
        "batch_size": 32,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "collate_function": "None",
        "scheduler": None,
        "n_epochs": 5,
        "pretrained": True,
        "pretrain_type": "bert",
        "transform": "bert",
        "collate_function": "None",
        "max_token_length": 300,
        "default_normalization": False,
        "dann_kwargs": {
            "featurizer_lr": 2e-6,
            "classifier_lr": 2e-5,
            "discriminator_lr": 2e-5,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 2e-6,
            "classifier_lr": 2e-5,
            "discriminator_lr": 2e-5,
            "penalty_weight": 1.0,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "amazon": {
        "num_classes": 5,
        "model": "distilbert-base-uncased",
        "optimizer": "AdamW",
        "batch_size": 24,
        "lr": 3e-5,
        "weight_decay": 0.01,
        "collate_function": "None",
        "scheduler": None,
        "n_epochs": 3,
        "pretrained": True,
        "pretrain_type": "bert",
        "transform": "bert",
        "collate_function": "None",
        "max_token_length": 512,
        "default_normalization": False,
        "dann_kwargs": {
            "featurizer_lr": 3e-6,
            "classifier_lr": 3e-5,
            "discriminator_lr": 3e-5,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 3e-6,
            "classifier_lr": 3e-5,
            "discriminator_lr": 3e-5,
            "penalty_weight": 1.0,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "imagenet": {
        "num_classes": 1000,
        "model": "resnet50",
        "optimizer": "SGD",
        "optimizer_kwargs": {"momentum": 0.9},
        "batch_size": 256,
        "n_epochs": 90,
        "lr": 0.1,
        "weight_decay": 0.0001,
        # 'scheduler': None,
        "collate_function": "None",
        "scheduler": "StepLR",
        "scheduler_kwargs": {"gamma": 0.1, "step_size": 30},
        "pretrained": True,
        "pretrain_type": "imagenet",
        "transform": "image_base",
        "resize_resolution": 256,
        "target_resolution": 224,
        "default_normalization": True,
        "dann_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
            "discriminator_lr": 0.01,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
            "discriminator_lr": 0.01,
            "penalty_weight": 1.0,
        },
        "coal_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
    "imagenet_c": {
        "num_classes": 1000,
        "model": "resnet50",
        "optimizer": "SGD",
        "optimizer_kwargs": {"momentum": 0.9},
        "batch_size": 256,
        "n_epochs": 90,
        "lr": 0.1,
        "weight_decay": 0.0001,
        # 'scheduler': None,
        "collate_function": "None",
        "scheduler": "StepLR",
        "scheduler_kwargs": {"gamma": 0.1, "step_size": 30},
        "pretrained": True,
        "pretrain_type": "imagenet",
        "transform": "image_base",
        "resize_resolution": 256,
        "target_resolution": 224,
        "default_normalization": True,
        "dann_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
            "discriminator_lr": 0.01,
            "penalty_weight": 1.0,
        },
        "cdan_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
            "discriminator_lr": 0.01,
            "penalty_weight": 1.0,
        },
        "coal_kwargs": {
            "featurizer_lr": 0.001,
            "classifier_lr": 0.01,
        },
        "evaluate_every": 1,
        # 'save_every': 1
    },
}
