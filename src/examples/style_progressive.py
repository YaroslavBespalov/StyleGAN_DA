import torch
from torch import Tensor, nn
from typing import List, Tuple, Dict
from gan.nn.stylegan.components import EqualLinear
from nn.progressiya.base import ProgressiveWithoutState, StateInjector, LastElementCollector
from nn.progressiya.unet import ProgressiveSequential


class InjectState(StateInjector):

    def inject(self, state: Tensor, args: Tuple[Tensor, ...], kw: Dict[str, Tensor]):
        return (torch.cat([state, args[0]], dim=-1), ), kw


class StyleTransform(nn.Module):

    def __init__(self):
        super().__init__()

        self.style_transform_1 = ProgressiveWithoutState[List[Tensor]](
            [EqualLinear(512, 512, activation='fused_lrelu')] \
            +[EqualLinear(512 * 2, 512, activation='fused_lrelu') for _ in range(13)],
            state_injector=InjectState()
        )

        self.style_transform_2 = ProgressiveWithoutState[List[Tensor]](
            [EqualLinear(512, 512, activation=None)] \
            +[EqualLinear(512 * 2, 512, activation=None) for _ in range(13)],
            state_injector=InjectState()
        )

        self.style_transform = ProgressiveSequential(
            self.style_transform_1,
            self.style_transform_2
        ).cuda()

    def forward(self, styles: Tensor):
        styles = [styles[:, i, ...] for i in range(styles.shape[1])]
        return torch.stack(tensors=self.style_transform(styles), dim=1)


class StyleDisc(nn.Module):

    def __init__(self):
        super().__init__()

        self.progressija = ProgressiveWithoutState[List[Tensor]](
            [EqualLinear(512, 512, activation='fused_lrelu')] \
            +[EqualLinear(512 * 2, 512, activation='fused_lrelu') for _ in range(13)],
            state_injector=InjectState(),
            collector_class=LastElementCollector
        )

        self.head = nn.Sequential(EqualLinear(512, 512, activation='fused_lrelu'),
                             EqualLinear(512, 1, activation=None))

    def forward(self, styles: Tensor):
        styles = [styles[:, i, ...] for i in range(styles.shape[1])]
        return self.head(self.progressija(styles))