import ROOT
import numpy as np
import matplotlib
matplotlib.use('agg')  # Use the 'agg' backend
import matplotlib.cm as cm
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import pathlib


def extract_data(root_file_path):
    # Open the ROOT file
    root_file = ROOT.TFile.Open(root_file_path)

    # Access the "Events" tree
    events_tree = root_file.Get("Events")

    # Create empty lists to store the extracted data
    jet_pt = []
    #jet_Pt = [] 
    jet_eta = []
    jet_phi = []
    event_n_jets = []

    # Loop over entries in the tree and extract data
    for event in events_tree:
        # Access and append data to lists
        jet_pt.append(list(event.Jet_pt))
        #jet_Pt.append(list(event.Jet_Pt))
        jet_eta.append([eta for eta in event.Jet_eta])
        jet_phi.append([phi for phi in event.Jet_phi])
        event_n_jets.append(np.array(event.nJet))

    # Close the ROOT file
    root_file.Close()

    return jet_pt, jet_eta, jet_phi, np.array(event_n_jets)


#QCD
# Specify the path to the ROOT file
path_QCD = "/pnfs/psi.ch/cms/trivcat/store/user/kadatta/jetObservables/QCD_Pt-300To470_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18PFNanov2pt2_jetObsSkim_WtopSelnomV7/240213_121041/0000/jetObservables_nanoskim_1.root"

# Extract data from the ROOT file
j_pt, j_eta, j_phi, event_n_jets = extract_data(path_QCD)

#jet_pt_QCD = np.concatenate(j_pt, axis=0)
#jet_eta_QCD = np.concatenate(j_eta, axis=0)
#jet_phi_QCD = np.concatenate(j_phi, axis=0)
#event_n_jets_QCD = event_n_jets

valid_jets_QCD = np.zeros(len(event_n_jets))

for i in range(len(event_n_jets)):
    a = 0
    for j in range(event_n_jets[i]):
        if j_pt[i][j]>=30 and j_pt[i][j]<=1000 and j_eta[i][j]<=2.5 :
            a = a + 1
    valid_jets_QCD[i] = a 

n_min_QCD = int(np.min(valid_jets_QCD))
n_max_QCD = int(np.max(valid_jets_QCD))
bin_QCD = n_max_QCD - n_min_QCD
#plt.show()
#plt.savefig("/work/mjeanfav/jetClassiferEfficiencywithGNN/comparaison_plots/valid_events.pdf")

# Print the extracted data
#print("Jet Pt : ", j_Pt)
#print("Jet pt : ", j_pt)


#Semi_Leptonic
path_Lept = "/pnfs/psi.ch/cms/trivcat/store/user/kadatta/jetObservables/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16PFNanov2pt2_jetObsSkim_WtopSelnomV7/240213_013759/0000/jetObservables_nanoskim_1.root"

# Extract data from the ROOT file
j_pt, j_eta, j_phi, event_n_jets = extract_data(path_Lept)

valid_jets_TT = np.zeros(len(event_n_jets))

for i in range(len(event_n_jets)):
    a = 0
    for j in range(event_n_jets[i]):
        if j_pt[i][j]>=30 and j_pt[i][j]<=1000 and j_eta[i][j]<=2.5 :
            a = a + 1
    valid_jets_TT[i] = a

n_min_TT = int(np.min(valid_jets_TT))
n_max_TT = int(np.max(valid_jets_TT))
#n_min = min(n_min_QCD,n_min_TT)
#n_max = max(n_max_QCD,n_max_TT)
bin_TT = n_max_TT-n_min_TT
print(n_min_QCD)

#Plot the histogram
# Create the histogram for dR_QCD
counts_qcd, bin_edges_qcd, _ = plt.hist(valid_jets_QCD, bins=bin_QCD, histtype='step', label="QCD", normed=False)
bin_centers_qcd = 0.5 * (bin_edges_qcd[1:] + bin_edges_qcd[:-1])-0.5

# Create the histogram for dR_Lept
counts_lept, bin_edges_lept, _ = plt.hist(valid_jets_TT, bins=bin_TT, histtype='step', label="TTTo", normed=False)
bin_centers_lept = 0.5 * (bin_edges_lept[1:] + bin_edges_lept[:-1])-0.5

plt.close()

counts_qcd_norm = counts_qcd/np.sum(counts_qcd)

# Calculate the Poisson errors for dR_QCD
poisson_errors_qcd = np.sqrt(counts_qcd)/np.sum(counts_qcd)

# Plot the histogram and Poisson error bars for dR_QCD
plt.errorbar(bin_centers_qcd, counts_qcd_norm, yerr=poisson_errors_qcd, fmt='_', capsize=4, color='r', label="QCD")

counts_lept_norm = counts_lept/np.sum(counts_lept)

# Calculate the Poisson errors for dR_Lept
poisson_errors_lept = np.sqrt(counts_lept)/np.sum(counts_lept)

# Plot the histogram and Poisson error bars for dR_Lept
plt.errorbar(bin_centers_lept, counts_lept_norm, yerr=poisson_errors_lept, fmt='_', capsize=4, color='b', label="TTTo")

# Add labels and legend
plt.xlabel("Valid jets per event")
plt.ylabel("Normalized Events")
plt.legend()

# Show the plot
plt.show()

# Save the plot
plt.savefig("/work/mjeanfav/jetClassiferEfficiencywithGNN/comparaison_plots/valid_events.pdf")
plt.close()


# plt.hist(valid_jets_QCD, bins=bin_QCD, histtype='step', fill=False, label='QCD', normed=True, alpha=0.5)
# plt.hist(valid_jets_TT, bins=bin_TT, histtype='step', fill=False, label='TTTo', normed=True, alpha=0.5)
# plt.legend()
# plt.xlabel("Valid jets per event")
# plt.ylabel("Normalized number of Events")
# plt.show()
# plt.savefig("/work/mjeanfav/jetClassiferEfficiencywithGNN/comparaison_plots/valid_events.pdf")

