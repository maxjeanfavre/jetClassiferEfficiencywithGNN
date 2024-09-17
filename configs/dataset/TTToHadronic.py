from utils.configs.dataset import DatasetConfig
import os

def get_filenames(path, file_pattern):
    # Function to get all files in the path
    files = os.listdir(path)

    # Filter the files based on the pattern
    matching_files = [f for f in files if f.startswith(file_pattern)]

    filenames = [os.path.basename(file) for file in matching_files]
    print(filenames)

    return filenames


dataset_config = DatasetConfig(
    name="TTToHadronic",
    path=(
        "/pnfs/psi.ch/cms/trivcat/store/user/kadatta/jetObservables/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16PFNanov2pt3_jetObsSkim_WtopSelnomV7/240213_030112/0000/"
    ),
    key="Events;1",
    file_limit=None,
    filename_pattern=r"^jetObservables_nanoskim_\d+.root$",
    branches_to_simulate=[
        {
            "name": "Jet_Pt",
            "expression": "Jet_pt",
        }
    ],

    filenames = get_filenames(
        path=(
        "/pnfs/psi.ch/cms/trivcat/store/user/kadatta/jetObservables/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16PFNanov2pt3_jetObsSkim_WtopSelnomV7/240213_030112/0000/"
        ),
        file_pattern="jetObservables_nanoskim_")
)
