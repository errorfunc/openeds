from .path_utils import (
    check_path,
    make_directory,
    glob_by_extension,
)

from .io_utils import (
    read_frame,
    save_frame,
)

from .dataset_utils import (
    strip_and_split,
    make_internal_id,
)

from .aug_utils import gaze_aug
from .tensor_utils import (
    image_to_tensor,
    cast_to_numpy,
)

from .dataset_utils import get_gkfold_split

__all__ = [
    'check_path',
    'make_directory',
    'glob_by_extension',
    'read_frame',
    'save_frame',
    'strip_and_split',
    'make_internal_id',
    'gaze_aug',
    'image_to_tensor',
    'cast_to_numpy',
    'get_gkfold_split',
    'TqdmStream',
]

from tqdm.auto import tqdm


class TqdmStream:
    @classmethod
    def write(cls, s):
        tqdm.write(s)

    @classmethod
    def flush(cls):
        pass
