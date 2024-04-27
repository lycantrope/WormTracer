import typing as _T
from itertools import count as _Counter
from pathlib import Path

import numpy as np
from attrs import define as _define
from attrs import field as _field
from torch.utils import data as _data

_NP_T: _T.TypeAlias = np.ndarray
_PATH_T: _T.TypeAlias = _T.Union[str, Path]


class Block(_T.NamedTuple):
    index: int
    is_complex: bool
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1

    def __repr__(self) -> str:
        return f"Block(index={self.index:d}, is_complex={self.is_complex:}, start={self.start:d}, end={self.end:d}, size={self.size:d})"


@_define(frozen=True)
class TrainingBlocks:
    blocks: _NP_T = _field(kw_only=True, repr=False)
    complex_block: _NP_T = _field(kw_only=True)
    rigid: float = _field(kw_only=True)
    relaxed: float = _field(kw_only=True)

    @classmethod
    def from_losses(
        cls,
        *,
        losses: _NP_T,
        relaxed: float,
        rigid: float,
    ) -> "TrainingBlocks":
        assert rigid > relaxed, "rigid margin must be greater than relaxed margin"
        complex_area = losses > relaxed
        distinct_from_prev = np.zeros_like(complex_area).astype(bool)
        distinct_from_prev[1:] = complex_area[:-1] ^ complex_area[1:]
        # labeling all blocks in 0-index
        blocks = distinct_from_prev.astype(int).cumsum()
        # filter the block that fulfilled rigid criteria
        complex_block_count = np.bincount(blocks[losses > rigid])
        complex_block = np.where(complex_block_count > 0)[0]

        return cls(
            blocks=blocks,
            complex_block=complex_block,
            relaxed=relaxed,
            rigid=rigid,
        )

    @classmethod
    def from_complex_area(
        cls,
        *,
        complex_area: _NP_T,
    ):

        distinct_from_prev = np.zeros_like(complex_area).astype(bool)
        distinct_from_prev[1:] = complex_area[:-1] ^ complex_area[1:]
        blocks = distinct_from_prev.astype(int).cumsum()
        # filter the block that fulfilled rigid criteria
        complex_block_count = np.bincount(blocks[complex_area])
        complex_block = np.where(complex_block_count > 0)[0]

        return cls(
            blocks=blocks,
            complex_block=complex_block,
            relaxed=np.nan,
            rigid=np.nan,
        )

    def batch_iter(self, batchsize: _T.Optional[int] = None) -> _T.Iterator[Block]:
        """Return an iterator that yields Block(idx, is_complex, start, end) with a batchsize

        Args:
            batchsize (int | None): Defaults to None, if batchsize is set, the blocks will be splitted in batchsize

        Returns:
            _T.Iterator[Block]: _description_

        Yields:
            Iterator[_T.Iterator[Block]]: _description_
        """
        block_sizes = np.bincount(self.blocks)
        # it will return the index of first occurence.
        label, onset = np.unique(self.blocks, return_index=True)
        offset = onset + block_sizes - 1
        mask = np.isin(label, self.complex_block)
        counter = _Counter()
        for m, start, end in zip(mask, onset, offset):
            if batchsize is None:
                yield Block(next(counter), m, start, end)
                continue

            for st in range(start, end, batchsize):
                yield Block(next(counter), m, st, min(st + batchsize - 1, end))

    def get_block_margins(self, use_complex_block: bool = True) -> _NP_T:
        """This method return the inclusive margin of each blocks [start, end].

        Args:
            complex_area (bool, optional): Specify to return the complex area or not. Defaults to True.

        Returns:
            margins: The ndarray of blocks margins with dimestion of (N, 2)
        """
        block_sizes = np.bincount(self.blocks)
        # it will return the index of first occurence.
        label, onset = np.unique(self.blocks, return_index=True)
        offset = onset + block_sizes - 1
        margins = np.array([onset, offset]).T

        mask = np.isin(label, self.complex_block)
        if not use_complex_block:
            mask = ~mask
        return margins[mask]

    def get_block_mask(self, get_complex: bool = True) -> _NP_T:
        mask = np.isin(self.blocks, self.complex_block)
        if not get_complex:
            mask = ~mask
        return mask

    # def calc_unit_length_from_blocks(
    #     self,
    #     x: _NP_T,
    #     y: _NP_T,
    #     use_complex_block: bool,
    # ) -> float:
    #     assert x.shape == y.shape, "shapes of x and y were different"
    #     assert x.shape[:1] == self.blocks.shape[:1], "shapes of x and y were different"

    #     mask = np.isin(self.blocks, self.complex_block)
    #     if not use_complex_block:
    #         mask = ~mask

    #     return np.sqrt(
    #         np.median(
    #             np.power(np.diff(x[mask], n=1, axis=1), 2)
    #             + np.power(np.diff(y[mask], n=1, axis=1), 2)
    #         )
    #     )


def get_use_points(
    image_losses: _NP_T,
    image_loss_max: float,
) -> TrainingBlocks:
    """
    Judge frames complex or not and get span for training.
    """
    # the criteria to filter complex area
    rigid = 0.4 * image_loss_max + 0.6 * np.min(image_losses)
    relaxed = 0.2 * image_loss_max + 0.8 * np.min(image_losses)

    return TrainingBlocks.from_losses(losses=image_losses, relaxed=relaxed, rigid=rigid)


class ImageStack(_data.Dataset):
    def __init__(self, data_folder: _PATH_T, ext: str = "png"):
        super().__init__()
        self.homepath = Path(data_folder)
        # sort data in lexicographic order
        self.images = sorted(data_folder.glob("*.{}".format(ext)), key=lambda x: x.stem)

    def __getitem__(self, idx):
        return self.images.__getitem__(idx)

    def __len__(self) -> int:
        return self.images.__len__()
