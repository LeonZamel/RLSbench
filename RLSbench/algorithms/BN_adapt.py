import logging

import torch
from RLSbench.algorithms.algorithm import Algorithm
from RLSbench.models.initializer import initialize_model
from RLSbench.utils import load, move_to

logger = logging.getLogger("label_shift")


class BN_adapt(Algorithm):
    def __init__(self, config, model):
        # initialize module
        super().__init__(
            device=config.device,
        )

        self.model = model
        self.epoch = "UNKNOWN"

        self.source_balanced = config.source_balanced
        self.num_classes = config.num_classes

    def get_model_output(self, x):
        outputs = self.model(x)
        return outputs

    def process_batch(self, batch):
        """
        A helper function for update() and evaluate() that processes the batch
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - y_pred (Tensor): model output for batch
        """
        x, y_true = batch[:2]
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)

        outputs = self.get_model_output(x)

        results = {
            "y_true": y_true,
            "y_pred": outputs,
        }
        return results

    def evaluate(self, batch):
        """
        Process the batch and update the log, without updating the model
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch, such as:
                - y_true (Tensor)
                - outputs (Tensor)
                - y_pred (Tensor)
        """
        assert not self.is_training
        results = self.process_batch(batch)
        return results

    def adapt(
        self,
        source_loader,
        target_loader,
        target_marginal=None,
        source_marginal=None,
        target_average=None,
        pretrained_path=None,
    ):
        """
        Load the model and adapt it to the new data
        Args:
            - unlabeled_batch (tuple of Tensors): a batch of data yielded by unlabeled data loader
            - target_marginal (Tensor): the marginal distribution of the target
            - source_marginal (Tensor): the marginal distribution of the source
            - target_average (Tensor): the average of the target

        Output:
        """

        if pretrained_path is not None:
            logger.info(f"Loading pretrained model from {pretrained_path}")
            load(self.model, pretrained_path, device=self.device)

        # self.train(True)

        logger.info("Adapting model to BN params ...")

        with torch.no_grad():
            for batch in target_loader:
                inp = batch[0].to(self.device)
                self.model(inp)

    def reset(self):
        pass
