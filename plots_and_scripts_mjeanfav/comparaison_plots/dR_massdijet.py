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
    jet_eta = []
    jet_phi = []
    event_n_jets = []

    # Loop over entries in the tree and extract data
    for event in events_tree:
        # Access and append data to lists
        jet_mass.append(list(event.Jet_mass))
        jet_eta.append([eta for eta in event.Jet_eta])
        jet_phi.append([phi for phi in event.Jet_phi])
        event_n_jets.append(np.array(event.nJet))

    #j_mass = []
    #j_mass.append(mass for mass in jet_mass)
    #j_mass = np.array(j_mass)

    #j_eta = []
    #j_eta.append(eta for eta in jet_eta)
    #j_eta = np.array(j_eta)

    #j_phi = []
    #j_phi.append(phi for phi in jet_phi)
    #j_phi = np.array(j_phi)
 
       #jet_mass.append(event.Jet_mass)
        #jet_eta.append(event.Jet_eta)
        #jet_phi.append(event.Jet_phi)

    # Close the ROOT file
    root_file.Close()

    return jet_mass, jet_eta, jet_phi, np.array(event_n_jets)

# Specify the path to the ROOT file
root_file_path = "/pnfs/psi.ch/cms/trivcat/store/user/kadatta/jetObservables/QCD_Pt-300To470_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18PFNanov2pt2_jetObsSkim_WtopSelnomV7/240213_121041/0000/jetObservables_nanoskim_1.root"

# Extract data from the ROOT file
j_mass, j_eta, j_phi, event_n_jets = extract_data(root_file_path)

jet_mass = np.concatenate(j_mass, axis=0)
jet_eta = np.concatenate(j_eta, axis=0)
jet_phi = np.concatenate(j_phi, axis=0)


# Print the extracted data
print("Jet Mass:", jet_mass)
print("Jet Eta:", type(jet_eta))
print("Jet Phi:", jet_phi[0])
#print("Event_n_jets : ", (event_n_jets))


#def compute_delta_r(eta , phi):
#    n = len(eta)
#    print(eta)
#    delta_r = np.full(
#        shape=(n, n),
#        fill_value=np.nan,
#        dtype=np.float64,
#    )
#    for i in range(n):
#        for j in range(n):
#            d_eta = eta[i] - eta[j]
#            d_phi = phi[i] - phi[j]
#            while d_phi >= np.pi:
#                d_phi -= 2 * np.pi
#            while d_phi < -np.pi:
#                d_phi += 2 * np.pi
#            delta_r[i][j] = np.sqrt(d_eta * d_eta + d_phi * d_phi)
#    return delta_r



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
                    # function to link node-node mass (can be modify)
                    m1 = mass[n_jets_offset + primary_jet_idx]
                    m2 = mass[n_jets_offset + secondary_jet_idx]

                    mass_dijet_values = np.sqrt(m1*m1 + m2*m2)

                    # mass_dijet_values = mass[n_jets_offset + primary_jet_idx] + mass[n_jets_offset + secondary_jet_idx]

                    mass_dijet[running_idx] = mass_dijet_values
                    running_idx += 1

    return(mass_dijet)


#    mass_dijet = np.split(
#        mass_dijet,
#        np.cumsum(event_n_jets * (event_n_jets - 1))[:-1],
#    )


delta_r = compute_delta_r(event_n_jets, jet_eta, jet_phi)
mass_dijet = get_mass_dijet(event_n_jets, jet_mass)

#print(np.shape(delta_r))
#print(np.shape(mass_dijet))


#plt.plot(mass_dijet[30000:],delta_r[30000:], 'r.')
plt.hist2d(mass_dijet,delta_r, bins=(50,50))
plt.xlim(0,125)
plt.xlabel("m(jj) [GeV]")
plt.ylabel("$\Delta$R(jj)")
plt.colorbar().set_label('Events')
plt.show()
plt.savefig("/work/mjeanfav/jetClassiferEfficiencywithGNN/comparaison_plots/correlation_mass_dijet_eucl_norm_dR.pdf")

#def plot_3d_scatter(delta_r, jet_mass):
#    fig = plt.figure()
#    ax = fig.add_subplot(projection='3d')

#    ax.scatter(delta_r, jet_mass, jet_mass)
#    ax.set_xlabel('Delta R')
#    ax.set_ylabel('Jet Mass')
#    ax.set_zlabel('Jet Mass')
#    fig.savefig("/work/mjeanfav/jetClassiferEfficiencywithGNN/correlation.pdf")
    #plt.show()
    #save_figure(fig,"/work/mjeanfav/jetClassiferEfficiencywithGNN/", correlation_dR_mass.pdf


#plot_3d_scatter(delta_r, jet_mass)


#def save_figure(fig, path, filename, **kwargs):
#    fig.savefig(fname=path / f"{filename}.pdf", **kwargs)
#
#    fig.savefig(fname=path / f"{filename}.jpg", **kwargs)
#    fig.savefig(fname=path / f"{filename}.png", **kwargs)

#path = pathlib.Path("/work/mjeanfav/jetClassiferEfficiencywithGNN/")

#save_figure(plot_3d_scatter(delta_r, jet_mass[0]),path, "correlation_dR_mass")
