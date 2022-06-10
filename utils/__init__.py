from utils.filenames import Filenames
from utils.paths_handler import Paths
from utils.settings import Settings


paths = Paths()
filenames = Filenames()
settings = Settings()

flavours_niceify = {
    0: "light-flavour",
    4: "charm",
    5: "bottom",
}

# TODO(high): use config.display_name
datasets_niceify = {
    "QCD_Pt_300_470_MuEnrichedPt5": "QCD",
    "QCD_Pt_300_470_MuEnrichedPt5_test": "QCD (test)",
    "TTTo2L2Nu": r"$\mathrm{t\bar{t}}$",
    "TTTo2L2Nu_test": r"$\mathrm{t\bar{t}}$ (test)",
}

branches_niceify = {
    "Jet_Pt": r"$p_\mathrm{T}$",
    "Jet_eta": r"$\eta$",
    "Jet_phi": r"$\phi$",
    "Jet_mass": r"Jet mass",
    "Jet_area": r"Jet area",
    "Jet_nConstituents": r"Jet constituents",
    "Jet_hadronFlavour": "Flavour",
}

working_points_niceify = {
    "btagWP_Loose_DeepCSV": "Loose",
    "btagWP_Medium_DeepCSV": "Medium",
    "btagWP_Tight_DeepCSV": "Tight",
}
