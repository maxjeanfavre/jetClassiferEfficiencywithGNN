import pathlib
import pkgutil


def get_package_names(dir_path: pathlib.Path):
    package_names = tuple(pkg.name for pkg in pkgutil.iter_modules(path=[dir_path]))

    return package_names
