import typing
import pathlib


def check_path(path: pathlib.Path) -> bool:
    if path.exists():
        return True
    return False


def make_directory(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def glob_by_extension(path: pathlib.Path, extension: str) -> typing.List[pathlib.Path]:
    return sorted(list(path.glob(f'**/*.{extension}')))
