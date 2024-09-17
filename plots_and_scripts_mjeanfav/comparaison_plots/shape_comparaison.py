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
    jet_mass = []
    jet_pt = []
    jet_eta = []
    jet_phi = []
    event_n_jets = []

    # Loop over entries in the tree and extract data
    for event in events_tree:
        # Access and append data to lists
        jet_mass.append(list(event.Jet_mass))
        jet_pt.append(list(event.Jet_pt))
        jet_eta.append([eta for eta in event.Jet_eta])
        jet_phi.append([phi for phi in event.Jet_phi])
        event_n_jets.append(np.array(event.nJet))

    # Close the ROOT file
    root_file.Close()

    return jet_mass, jet_pt, jet_eta, jet_phi, np.array(event_n_jets)


#QCD
# Specify the path to the ROOT file
path_QCD = "/pnfs/psi.ch/cms/trivcat/store/user/kadatta/jetObservables/QCD_Pt-300To470_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18PFNanov2pt2_jetObsSkim_WtopSelnomV7/240213_121041/0000/jetObservables_nanoskim_1.root"

# Extract data from the ROOT file
j_mass, j_pt, j_eta, j_phi, event_n_jets = extract_data(path_QCD)

jet_mass_QCD = np.concatenate(j_mass, axis=0)
jet_pt_QCD = np.concatenate(j_pt, axis=0)
jet_eta_QCD = np.concatenate(j_eta, axis=0)
jet_phi_QCD = np.concatenate(j_phi, axis=0)
event_n_jets_QCD = event_n_jets


# Print the extracted data
#print("Jet Mass : ", jet_mass)
#print("Jet Eta type : " type(jet_eta))
#print("jet pt : ", jet_pt_QCD)


# #di-lepton
# # Specify the path to the ROOT file
# path_2L2N = "/pnfs/psi.ch/cms/trivcat/store/user/kadatta/jetObservables/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16PFNanov2pt2_jetObsSkim_WtopSelnomV7/240213_030219/0000/jetObservables_nanoskim_1.root"

# # Extract data from the ROOT file
# j_mass, j_pt, j_eta, j_phi, event_n_jets = extract_data(path_2L2N)

# jet_mass_2L2N = np.concatenate(j_mass, axis=0)
# jet_pt_2L2N = np.concatenate(j_pt, axis=0)
# jet_eta_2L2N = np.concatenate(j_eta, axis=0)
# jet_phi_2L2N = np.concatenate(j_phi, axis=0)
# event_n_jets_2L2N = event_n_jets


# #Hadronic
# path_Hadro = "/pnfs/psi.ch/cms/trivcat/store/user/kadatta/jetObservables/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16PFNanov2pt3_jetObsSkim_WtopSelnomV7/240213_030112/0000/jetObservables_nanoskim_1.root"

# # Extract data from the ROOT file
# j_mass, j_pt, j_eta, j_phi, event_n_jets = extract_data(path_Hadro)

# jet_mass_Hadro = np.concatenate(j_mass, axis=0)
# jet_pt_Hadro = np.concatenate(j_pt, axis=0)
# jet_eta_Hadro = np.concatenate(j_eta, axis=0)
# jet_phi_Hadro = np.concatenate(j_phi, axis=0)
# event_n_jets_Hadro = event_n_jets


#Semi_Leptonic
path_Lept = "/pnfs/psi.ch/cms/trivcat/store/user/kadatta/jetObservables/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16PFNanov2pt2_jetObsSkim_WtopSelnomV7/240213_013759/0000/jetObservables_nanoskim_1.root"

# Extract data from the ROOT file
j_mass, j_pt, j_eta, j_phi, event_n_jets = extract_data(path_Lept)

jet_mass_Lept = np.concatenate(j_mass, axis=0)
jet_pt_Lept = np.concatenate(j_pt, axis=0)
jet_eta_Lept = np.concatenate(j_eta, axis=0)
jet_phi_Lept = np.concatenate(j_phi, axis=0)
event_n_jets_Lept = event_n_jets



#Definition of environment variables dR and mjj

def compute_delta_r(event_n_jets, eta, phi):
    delta_r_values = np.full(
        shape=np.sum(event_n_jets * (event_n_jets - 1)),
        fill_value=np.nan,
        dtype=np.float64,
    )
    n_jets = np.sum(event_n_jets)
    events_jets_offset = np.concatenate((np.array([0]), np.cumsum(event_n_jets[:-1])))
    
    running_idx = 0
    for event_idx, n_jets in enumerate(event_n_jets):
        n_jets_offset = events_jets_offset[event_idx]
        for primary_jet_idx in range(n_jets):
            for secondary_jet_idx in range(n_jets):
                if primary_jet_idx != secondary_jet_idx:
                    delta_r = compute_delta_r_single_combination_njit(
                        eta[n_jets_offset + primary_jet_idx],
                        phi[n_jets_offset + primary_jet_idx],
                        eta[n_jets_offset + secondary_jet_idx],
                        phi[n_jets_offset + secondary_jet_idx],
                    )
                    delta_r_values[running_idx] = delta_r
                    running_idx += 1

    return delta_r_values



def compute_delta_r_single_combination_njit(eta_1, phi_1, eta_2, phi_2):
    d_eta = eta_1 - eta_2
    d_phi = phi_1 - phi_2

    while d_phi >= np.pi:
        d_phi -= 2 * np.pi
    while d_phi < -np.pi:
        d_phi += 2 * np.pi

    return np.sqrt(d_eta * d_eta + d_phi * d_phi)


# calculate mass_dijet
def get_mass_dijet(event_n_jets, mass):
    n_jets = np.sum(event_n_jets)

    events_jets_offset = np.concatenate((np.array([0]), np.cumsum(event_n_jets[:-1])))
    running_idx = 0

    mass_dijet = np.full(
        shape=np.sum(event_n_jets * (event_n_jets - 1)),
        fill_value=np.nan,
        dtype=np.float64,
    )
    for event_idx, n_jets in enumerate(event_n_jets):
        n_jets_offset = events_jets_offset[event_idx]
        for primary_jet_idx in range(n_jets):
            for secondary_jet_idx in range(n_jets):
                if primary_jet_idx != secondary_jet_idx:
                    #function to link node-node mass (can be modify)
                    mass_dijet_values = mass[n_jets_offset + primary_jet_idx] + mass[n_jets_offset + secondary_jet_idx]

                    mass_dijet[running_idx] = mass_dijet_values
                    running_idx += 1

    return(mass_dijet)



#Environment variables for each dataset
dR_QCD = compute_delta_r(event_n_jets_QCD, jet_eta_QCD, jet_phi_QCD)
mjj_QCD = get_mass_dijet(event_n_jets_QCD, jet_mass_QCD)

# dR_2L2N = compute_delta_r(event_n_jets_2L2N, jet_eta_2L2N, jet_phi_2L2N)
# mjj_2L2N = get_mass_dijet(event_n_jets_2L2N, jet_mass_2L2N)

# dR_Hadro = compute_delta_r(event_n_jets_Hadro, jet_eta_Hadro, jet_phi_Hadro)
# mjj_Hadro = get_mass_dijet(event_n_jets_Hadro, jet_mass_Hadro)

dR_Lept = compute_delta_r(event_n_jets_Lept, jet_eta_Lept, jet_phi_Lept)
mjj_Lept = get_mass_dijet(event_n_jets_Lept, jet_mass_Lept)


#print(np.shape(mjj_QCD))
# print(np.shape(mjj_2L2N))
# print(np.shape(mjj_Hadro))
#print(np.shape(mjj_Lept))
#print(mjj_QCD[:2000])


# Plot shape of dR for QCD and Lept
# Normalized histogram with Poisson error bars



#Plot dR
# Create the histogram for dR_QCD
counts_qcd, bin_edges_qcd, _ = plt.hist(dR_QCD, bins=20, range=(0.5, 4), histtype='step', label="QCD", normed=False)
bin_centers_qcd = 0.5 * (bin_edges_qcd[1:] + bin_edges_qcd[:-1])

# Create the histogram for dR_Lept
counts_lept, bin_edges_lept, _ = plt.hist(dR_Lept, bins=20, range=(0.5, 4), histtype='step', label="TTTo", normed=False)
bin_centers_lept = 0.5 * (bin_edges_lept[1:] + bin_edges_lept[:-1])

plt.close()

counts_qcd_norm = counts_qcd/np.sum(counts_qcd)

# Calculate the Poisson errors for dR_QCD
poisson_errors_qcd = np.sqrt(counts_qcd)/np.sum(counts_qcd)

# Plot the histogram and Poisson error bars for dR_QCD
plt.errorbar(bin_centers_qcd, counts_qcd_norm, yerr=poisson_errors_qcd, fmt='+', capsize=4, color='r', label="QCD")

counts_lept_norm = counts_lept/np.sum(counts_lept)

# Calculate the Poisson errors for dR_Lept
poisson_errors_lept = np.sqrt(counts_lept)/np.sum(counts_lept)

# Plot the histogram and Poisson error bars for dR_Lept
plt.errorbar(bin_centers_lept, counts_lept_norm, yerr=poisson_errors_lept, fmt='+', capsize=4, color='b', label="TTTo")

# Add labels and legend
plt.xlabel(r"$\Delta R$")
plt.ylabel("Normalized Events")
plt.legend()

# Show the plot
plt.show()

# Save the plot
plt.savefig("/work/mjeanfav/jetClassiferEfficiencywithGNN/comparaison_plots/shape_comp_delta_r.pdf")
plt.close()




#Plot mjj

# Create the histogram for mjj_QCD
counts_qcd, bin_edges_qcd, _ = plt.hist(mjj_QCD, bins=20, range=(0,125), histtype='step', label="QCD", normed=False)
bin_centers_qcd = 0.5 * (bin_edges_qcd[1:] + bin_edges_qcd[:-1])

# Create the histogram for dR_Lept
counts_lept, bin_edges_lept, _ = plt.hist(mjj_Lept, bins=20, range=(0,125), histtype='step', label="TTTo", normed=False)
bin_centers_lept = 0.5 * (bin_edges_lept[1:] + bin_edges_lept[:-1])

plt.close()

counts_qcd_norm = counts_qcd/np.sum(counts_qcd)

# Calculate the Poisson errors for dR_QCD
poisson_errors_qcd = np.sqrt(counts_qcd)/np.sum(counts_qcd)

# Plot the histogram and Poisson error bars for dR_QCD
plt.errorbar(bin_centers_qcd, counts_qcd_norm, yerr=poisson_errors_qcd, fmt='+', capsize=4, color='r', label="QCD")

counts_lept_norm = counts_lept/np.sum(counts_lept)

# Calculate the Poisson errors for dR_Lept
poisson_errors_lept = np.sqrt(counts_lept)/np.sum(counts_lept)

# Plot the histogram and Poisson error bars for dR_Lept
plt.errorbar(bin_centers_lept, counts_lept_norm, yerr=poisson_errors_lept, fmt='+', capsize=4, color='b', label="TTTo")

# Add labels and legend
plt.xlabel("Mass dijet [GeV]")
plt.ylabel("Normalized Events")
plt.legend()

# Show the plot
plt.show()

# Save the plot
plt.savefig("/work/mjeanfav/jetClassiferEfficiencywithGNN/comparaison_plots/shape_comp_mass_dijet.pdf")
plt.close()



#Plot pt, eta ,phi

pt_counts_qcd, pt_bin_edges_qcd, _ = plt.hist(jet_pt_QCD, bins=20, range=(0,600), histtype='step', label="QCD", normed=False)
pt_bin_centers_qcd = 0.5 * (pt_bin_edges_qcd[1:] + pt_bin_edges_qcd[:-1])
pt_counts_lept, pt_bin_edges_lept, _ = plt.hist(jet_pt_Lept, bins=20, range=(0,600), histtype='step', label="TTTo", normed=False)
pt_bin_centers_lept = 0.5 * (pt_bin_edges_lept[1:] + pt_bin_edges_lept[:-1])

eta_counts_qcd, eta_bin_edges_qcd, _ = plt.hist(jet_eta_QCD, bins=20, histtype='step', label="QCD", normed=False)
eta_bin_centers_qcd = 0.5 * (eta_bin_edges_qcd[1:] + eta_bin_edges_qcd[:-1])
eta_counts_lept, eta_bin_edges_lept, _ = plt.hist(jet_eta_Lept, bins=20, histtype='step', label="TTTo", normed=False)
eta_bin_centers_lept = 0.5 * (eta_bin_edges_lept[1:] + eta_bin_edges_lept[:-1])

phi_counts_qcd, phi_bin_edges_qcd, _ = plt.hist(jet_phi_QCD, bins=20, histtype='step', label="QCD", normed=False)
phi_bin_centers_qcd = 0.5 * (phi_bin_edges_qcd[1:] + phi_bin_edges_qcd[:-1])
phi_counts_lept, phi_bin_edges_lept, _ = plt.hist(jet_phi_Lept, bins=20, histtype='step', label="TTTo", normed=False)
phi_bin_centers_lept = 0.5 * (phi_bin_edges_lept[1:] + phi_bin_edges_lept[:-1])

mass_counts_qcd, mass_bin_edges_qcd, _ = plt.hist(jet_mass_QCD, bins=20, range=(0,60), histtype='step', label="QCD", normed=False)
mass_bin_centers_qcd = 0.5 * (mass_bin_edges_qcd[1:] + mass_bin_edges_qcd[:-1])
mass_counts_lept, mass_bin_edges_lept, _ = plt.hist(jet_mass_Lept, bins=20, range=(0,60), histtype='step', label="TTTo", normed=False)
mass_bin_centers_lept = 0.5 * (mass_bin_edges_lept[1:] + mass_bin_edges_lept[:-1])

plt.close()

# Normalize the histograms
pt_counts_qcd_norm = pt_counts_qcd/np.sum(pt_counts_qcd)
pt_counts_lept_norm = pt_counts_lept/np.sum(pt_counts_lept)

eta_counts_qcd_norm = eta_counts_qcd/np.sum(eta_counts_qcd)
eta_counts_lept_norm = eta_counts_lept/np.sum(eta_counts_lept)

phi_counts_qcd_norm = phi_counts_qcd/np.sum(phi_counts_qcd)
phi_counts_lept_norm = phi_counts_lept/np.sum(phi_counts_lept)

mass_counts_qcd_norm = mass_counts_qcd/np.sum(mass_counts_qcd)
mass_counts_lept_norm = mass_counts_lept/np.sum(mass_counts_lept)

# Calculate the Poisson errors for pt, eta, phi
pt_poisson_errors_qcd = np.sqrt(pt_counts_qcd)/np.sum(pt_counts_qcd)
pt_poisson_errors_lept = np.sqrt(pt_counts_lept)/np.sum(pt_counts_lept)

eta_poisson_errors_qcd = np.sqrt(eta_counts_qcd)/np.sum(eta_counts_qcd)
eta_poisson_errors_lept = np.sqrt(eta_counts_lept)/np.sum(eta_counts_lept)

phi_poisson_errors_qcd = np.sqrt(phi_counts_qcd)/np.sum(phi_counts_qcd)
phi_poisson_errors_lept = np.sqrt(phi_counts_lept)/np.sum(phi_counts_lept)

mass_poisson_errors_qcd = np.sqrt(mass_counts_qcd)/np.sum(mass_counts_qcd)
mass_poisson_errors_lept = np.sqrt(mass_counts_lept)/np.sum(mass_counts_lept)


# Plot the histograms and Poisson error bars for pt
plt.errorbar(pt_bin_centers_qcd, pt_counts_qcd_norm, yerr=pt_poisson_errors_qcd, fmt='+', capsize=4, color='r', label="QCD")
plt.errorbar(pt_bin_centers_lept, pt_counts_lept_norm, yerr=pt_poisson_errors_lept, fmt='+', capsize=4, color='b', label="TTTo")

plt.xlabel(r"$p_t$ [GeV]")
plt.ylabel("Normalized Events")
plt.legend()

plt.show()
plt.savefig("/work/mjeanfav/jetClassiferEfficiencywithGNN/comparaison_plots/shape_comp_pt.pdf")
plt.close()

# Plot the histograms and Poisson error bars for eta
plt.errorbar(eta_bin_centers_qcd, eta_counts_qcd_norm, yerr=eta_poisson_errors_qcd, fmt='+', capsize=4, color='r', label="QCD")
plt.errorbar(eta_bin_centers_lept, eta_counts_lept_norm, yerr=eta_poisson_errors_lept, fmt='+', capsize=4, color='b', label="TTTo")

plt.xlabel(r"$\eta$")
plt.ylabel("Normalized Events")
plt.legend()

plt.show()
plt.savefig("/work/mjeanfav/jetClassiferEfficiencywithGNN/comparaison_plots/shape_comp_eta.pdf")
plt.close()

# Plot the histograms and Poisson error bars for phi
plt.errorbar(phi_bin_centers_qcd, phi_counts_qcd_norm, yerr=phi_poisson_errors_qcd, fmt='+', capsize=4, color='r', label="QCD")
plt.errorbar(phi_bin_centers_lept, phi_counts_lept_norm, yerr=phi_poisson_errors_lept, fmt='+', capsize=4, color='b', label="TTTo")

plt.xlabel(r"$\phi$")
plt.ylabel("Normalized Events")
plt.legend()

plt.show()
plt.savefig("/work/mjeanfav/jetClassiferEfficiencywithGNN/comparaison_plots/shape_comp_phi.pdf")
plt.close()

# Plot the histograms and Poisson error bars for mass
plt.errorbar(mass_bin_centers_qcd, mass_counts_qcd_norm, yerr=mass_poisson_errors_qcd, fmt='+', capsize=4, color='r', label="QCD")
plt.errorbar(mass_bin_centers_lept, mass_counts_lept_norm, yerr=mass_poisson_errors_lept, fmt='+', capsize=4, color='b', label="TTTo")

plt.xlabel(r"Mass [GeV]")
plt.ylabel("Normalized Events")
plt.legend()

plt.show()
plt.savefig("/work/mjeanfav/jetClassiferEfficiencywithGNN/comparaison_plots/shape_comp_mass.pdf")
plt.close()