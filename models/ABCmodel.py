import abc
import tensorflow as tf
from typing import List


class ABCmodel(abc.ABC):
    def __init__(self,
                 INPUT_SHAPE: List[int],
                 OUTPUT_SHAPE: List[int]) -> None:
        self.INPUT_SHAPE = INPUT_SHAPE
        self.OUTPUT_SHAPE = OUTPUT_SHAPE

    @abc.abstractmethod
    def model_fn(self,
                 features: tf.Tensor,
                 mode: tf.estimator.ModeKeys) -> tf.Tensor:
        """
            alexnet image classification convolutional network

            :param features: input of the network
            :param mode: standard names for Estimator model modes
            :return: output of the network (except last FC that evaluates the logits)

        """
        pass

    @abc.abstractmethod
    def get_feature_columns(self) -> List[object]:
        pass
