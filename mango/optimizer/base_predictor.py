import abc


class BasePredictor(object):

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def get_next_batch(self, X,Y,X_tries):
    """
    Gives the batch of next suggestions to try for the algorithm
    """
    raise NotImplementedError
