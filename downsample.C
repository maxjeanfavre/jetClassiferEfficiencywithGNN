// root -q downsample.C
#include <iostream>


void downsample() {
    TString src_dir = "/eos/cms/store/group/phys_higgs/hbb/ntuples/VHbbPostNano/2018/V12/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18NanoAODv6-Nano25O83/200302_100732/0000/";
    TString target_dir = "test_data/input/";
    int files = 5;
    int sample_size = 1000;
    TString tree_name = "Events;1";

    assert (src_dir.EndsWith("/"));
    assert (target_dir.EndsWith("/"));

    gSystem->Exec(TString::Format("mkdir -p %s", target_dir.Data()));
    std::cout << "Created target dir: " << target_dir << std::endl;


    int files_ = files;

    for (int i = 1; i <= files_; i++) {
        // format the path/file name
        TString original_filename = TString::Format("%s%s%d.root", src_dir.Data(), "tree_", i);
        // make sure the file exists
        if (gSystem->AccessPathName(original_filename, kFileExists)) { // as it returns False when it can accessed, see https://root.cern.ch/doc/master/classTSystem.html#a849c28ea0dd3b3aa3310a4d447c7b21a
            std::cout << "File does not exist " << original_filename << std::endl;
            files_++; // increase to actually sample the number of files as requested
        } else {
            std::cout << "Sampling " << original_filename << std::endl;
            // if it exists, generate downsampled version
            TFile original_file(original_filename);
            TTree * original_tree;
            original_file.GetObject(tree_name, original_tree);

            // Create a new file + a clone of old tree with the requested number of entries.
            TString downsampled_filename = TString::Format("%s%s%d%s.root", target_dir.Data(), "tree_", i, "_sample");
            TFile downsampled_file(downsampled_filename, "recreate");
            auto downsampled_tree = original_tree -> CloneTree(sample_size);

            downsampled_file.Write();
        }
    }

    std::cout << "Done" << std::endl;
}
