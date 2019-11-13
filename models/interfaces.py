import abc


class modelABC(abc.ABC):
    @abc.abstractmethod
    def model_fn(self, features, mode):
        pass
