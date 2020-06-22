import torch
from typing import List, Tuple, Dict, Union


class ScaleControllerBase(torch.nn.Module):
    """
    The base class for ScaleController.
    ScaleController is a callable class that re-scale input tensor's value.
    Traditional scale method may include:
        soft-max, L2 normalize, relu and so on.
    Advanced method:
        Learnable scale parameter
    """
    def __init__(self):
        super(ScaleControllerBase, self).__init__()

    def forward(self, x:  torch.Tensor, dim: int = 0, p: int = 1):
        """
        Re-scale the input x into proper value scale.
        :param x: the input tensor
        :param dim: axis to scale(mostly used in traditional method)
        :param p: p parameter used in traditional methods
        :return: rescaled x
        """
        raise NotImplementedError


class LearnableScaleController(ScaleControllerBase):
    """
    Scale parameter mentioned in [Tadam: Task dependent adaptive metric for improved few-shot learning. (NIPS2018)]
    """
    def __init__(self, normalizer: ScaleControllerBase = None):
        super(LearnableScaleController, self).__init__()
        self.scale_rate = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        self.normalizer = normalizer

    def forward(self, x:  torch.Tensor, dim: int = 0, p: int = 1):
        x = x * self.scale_rate
        if self.normalizer:
            x = self.normalizer(x, dim=dim, p=p)
        return x


class FixedScaleController(ScaleControllerBase):
    """
    Scale parameter with a fixed value.
    """
    def __init__(self, normalizer: ScaleControllerBase = None, scale_rate: float = 50):
        super(FixedScaleController, self).__init__()
        self.scale_rate = scale_rate
        self.normalizer = normalizer

    def forward(self, x:  torch.Tensor, dim: int = 0, p: int = 1):
        x = x * self.scale_rate
        if self.normalizer:
            x = self.normalizer(x, dim=dim, p=p)
        return x


class NormalizeScaleController(ScaleControllerBase):
    def __init__(self):
        super(NormalizeScaleController, self).__init__()

    def forward(self, x:  torch.Tensor, dim: int = -1, p: int = 1):
        return torch.nn.functional.normalize(x, p=p, dim=dim)


class SoftmaxScaleController(ScaleControllerBase):
    def __init__(self):
        super(SoftmaxScaleController, self).__init__()

    def forward(self, x:  torch.Tensor, dim: int = 0, p: int = 2):
        return torch.nn.functional.softmax(x, dim=dim)


class ExpScaleController(ScaleControllerBase):
    """ rescale to non-negative """
    def __init__(self):
        super(SoftmaxScaleController, self).__init__()

    def forward(self, x:  torch.Tensor, dim: int = 0, p: int = 2):
        return torch.exp(x, dim=dim)


class ReluScaleController(ScaleControllerBase):
    """ rescale to non-negative """
    def __init__(self):
        super(LearnableScaleController, self).__init__()
        self.scale_rate = torch.nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x:  torch.Tensor, dim: int = 0, p: int = 1):
        return torch.nn.functional.relu(x)


class MixedScaleController(torch.nn.Module):
    def __init__(self, controller_lst: List[ScaleControllerBase]):
        """
        A callable tool class that perform a series of scaling operation in given order
        :param controller_lst: ordered scaling operations
        """
        super(MixedScaleController, self).__init__()
        self.controller_lst = controller_lst

    def forward(self, x:  torch.Tensor, dim_lst: List[int] = None, p_lst: List[int] = None):
        ret = x
        for op, dim, p in zip(self.controller_lst, ):
            ret = op(ret, dim=dim, p=p)
        return ret


def build_scale_controller(name: str, kwargs=None) -> Union[ScaleControllerBase, None]:
    """
    A tool function that help to select scale controller easily.
    :param name: name of scale controller, choice now: 'learn', 'fix', 'relu', 'exp', 'softmax', 'norm'
    :param kwargs: necessary controller parameter in dictionary style
    :return:
    """
    if not name or name == 'none':
        return None
    controller_choices = {
        'learn': LearnableScaleController,
        'fix': FixedScaleController,
        'relu': ReluScaleController,
        'exp': ExpScaleController,
        'softmax': SoftmaxScaleController,
        'norm': NormalizeScaleController,
    }
    if name not in controller_choices:
        raise KeyError('Wrong scale controller name.')
    controller_type = controller_choices[name]
    return controller_type(**kwargs) if kwargs else controller_type()
