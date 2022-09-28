import typing

import torch.nn as nn
from torchvision.models.resnet import (
    ResNet,
    BasicBlock,
    Bottleneck,
)


class GreyResNet(ResNet):
    def __init__(
        self,
        block: typing.Type[typing.Union[BasicBlock, Bottleneck]],
        layers: typing.List[int],
        num_classes: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: typing.Optional[typing.List[bool]] = None,
        norm_layer: typing.Optional[typing.Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer
        )
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


def _grey_resnet(block, layers, **kwargs):
    model = GreyResNet(block, layers, **kwargs)
    return model


def grey_resnet18(**kwargs):
    return _grey_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)
