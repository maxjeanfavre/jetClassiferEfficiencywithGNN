import pathlib
import pkgutil


def get_package_names(dir_path: pathlib.Path):
    package_names = tuple(pkg.name for pkg in pkgutil.iter_modules(path=[dir_path]))
    print("package_names pkgutil.iter_modules(path=[dir_path]) ",package_names, " dir_path ",dir_path) #package_names pkgutil.iter_modules(path=[dir_path])  ('QCD_Pt_300_470_MuEnrichedPt5', 'QCD_Pt_300_470_MuEnrichedPt5_test', 'TTTo2L2Nu', 'TTTo2L2Nu_test')  dir_path  /work/krgedia/CMSSW_10_1_0/src/Xbb/python/gnn_b_tagging_efficiency/configs/dataset 
    return package_names
