from utils.configs.dataset import DatasetConfig


dataset_config = DatasetConfig(
    name="TTTo2L2Nu",
    path=(
        "root://eoscms.cern.ch//eos/cms/store/group/phys_higgs/hbb/ntuples/"
        "VHbbPostNano/2018/V12/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/"
        "RunIIAutumn18NanoAODv6-Nano25O83/200302_100732/0000/"
    ),
    key="Events;1",
    file_limit=None,
    filename_pattern=r"^tree_\d+.root$",
    branches_to_simulate=None,
    filenames=(
        "tree_1.root",
        "tree_2.root",
        "tree_3.root",
        "tree_4.root",
        "tree_5.root",
        "tree_6.root",
        "tree_7.root",
        "tree_8.root",
        "tree_9.root",
        "tree_10.root",
        "tree_11.root",
        "tree_12.root",
        "tree_13.root",
        "tree_14.root",
        "tree_16.root",
        "tree_17.root",
        "tree_18.root",
        "tree_19.root",
        "tree_20.root",
        "tree_21.root",
        "tree_22.root",
        "tree_23.root",
        "tree_24.root",
        "tree_25.root",
        "tree_26.root",
        "tree_27.root",
        "tree_28.root",
        "tree_29.root",
        "tree_30.root",
        "tree_31.root",
        "tree_32.root",
        "tree_33.root",
        "tree_34.root",
        "tree_35.root",
        "tree_36.root",
        "tree_37.root",
        "tree_38.root",
        "tree_39.root",
        "tree_40.root",
        "tree_41.root",
        "tree_42.root",
        "tree_43.root",
        "tree_44.root",
        "tree_45.root",
        "tree_46.root",
        "tree_47.root",
        "tree_48.root",
        "tree_49.root",
        "tree_50.root",
        "tree_51.root",
        "tree_52.root",
        "tree_53.root",
        "tree_54.root",
        "tree_55.root",
        "tree_56.root",
        "tree_57.root",
        "tree_58.root",
        "tree_59.root",
        "tree_60.root",
        "tree_61.root",
        "tree_62.root",
        "tree_63.root",
        "tree_64.root",
        "tree_65.root",
        "tree_66.root",
        "tree_67.root",
        "tree_68.root",
        "tree_69.root",
        "tree_70.root",
        "tree_71.root",
        "tree_72.root",
        "tree_73.root",
        "tree_74.root",
        "tree_75.root",
        "tree_76.root",
        "tree_77.root",
        "tree_78.root",
        "tree_79.root",
        "tree_80.root",
        "tree_81.root",
        "tree_82.root",
        "tree_83.root",
        "tree_84.root",
        "tree_85.root",
        "tree_86.root",
        "tree_87.root",
        "tree_88.root",
        "tree_89.root",
        "tree_90.root",
        "tree_91.root",
        "tree_92.root",
        "tree_93.root",
        "tree_94.root",
        "tree_95.root",
        "tree_96.root",
        "tree_97.root",
        "tree_98.root",
        "tree_99.root",
        "tree_100.root",
        "tree_101.root",
        "tree_102.root",
        "tree_103.root",
        "tree_104.root",
        "tree_105.root",
        "tree_106.root",
        "tree_107.root",
        "tree_108.root",
        "tree_109.root",
        "tree_110.root",
        "tree_111.root",
        "tree_112.root",
        "tree_113.root",
        "tree_114.root",
        "tree_115.root",
        "tree_116.root",
        "tree_117.root",
        "tree_118.root",
        "tree_119.root",
        "tree_120.root",
        "tree_121.root",
        "tree_122.root",
        "tree_123.root",
        "tree_124.root",
        "tree_125.root",
        "tree_126.root",
        "tree_127.root",
        "tree_128.root",
        "tree_129.root",
        "tree_130.root",
        "tree_131.root",
        "tree_132.root",
        "tree_133.root",
        "tree_134.root",
        "tree_135.root",
        "tree_136.root",
        "tree_137.root",
        "tree_138.root",
        "tree_139.root",
        "tree_140.root",
        "tree_141.root",
        "tree_142.root",
        "tree_143.root",
        "tree_144.root",
        "tree_145.root",
        "tree_146.root",
        "tree_147.root",
        "tree_148.root",
        "tree_149.root",
        "tree_150.root",
        "tree_151.root",
        "tree_152.root",
        "tree_153.root",
        "tree_154.root",
        "tree_155.root",
        "tree_156.root",
        "tree_157.root",
        "tree_158.root",
        "tree_159.root",
        "tree_160.root",
        "tree_161.root",
        "tree_162.root",
        "tree_163.root",
        "tree_164.root",
        "tree_165.root",
        "tree_166.root",
        "tree_167.root",
        "tree_168.root",
        "tree_169.root",
        "tree_170.root",
        "tree_171.root",
        "tree_172.root",
        "tree_173.root",
        "tree_174.root",
        "tree_175.root",
        "tree_176.root",
        "tree_177.root",
        "tree_178.root",
        "tree_179.root",
        "tree_180.root",
        "tree_181.root",
        "tree_182.root",
        "tree_183.root",
        "tree_184.root",
        "tree_185.root",
        "tree_186.root",
        "tree_187.root",
        "tree_188.root",
        "tree_189.root",
        "tree_190.root",
        "tree_191.root",
        "tree_192.root",
        "tree_193.root",
        "tree_194.root",
        "tree_195.root",
        "tree_196.root",
        "tree_197.root",
        "tree_198.root",
        "tree_199.root",
        "tree_200.root",
        "tree_201.root",
        "tree_202.root",
        "tree_203.root",
        "tree_204.root",
        "tree_205.root",
        "tree_206.root",
        "tree_207.root",
        "tree_208.root",
        "tree_209.root",
        "tree_210.root",
        "tree_211.root",
        "tree_212.root",
        "tree_213.root",
        "tree_214.root",
        "tree_215.root",
        "tree_216.root",
        "tree_217.root",
        "tree_218.root",
        "tree_220.root",
        "tree_225.root",
        "tree_226.root",
        "tree_227.root",
        "tree_228.root",
        "tree_230.root",
        "tree_231.root",
        "tree_232.root",
        "tree_233.root",
        "tree_234.root",
        "tree_235.root",
        "tree_236.root",
        "tree_237.root",
        "tree_238.root",
        "tree_239.root",
        "tree_240.root",
        "tree_241.root",
        "tree_242.root",
        "tree_243.root",
        "tree_244.root",
        "tree_245.root",
        "tree_246.root",
        "tree_247.root",
        "tree_248.root",
        "tree_249.root",
        "tree_250.root",
        "tree_251.root",
        "tree_252.root",
        "tree_253.root",
        "tree_254.root",
        "tree_255.root",
        "tree_256.root",
        "tree_257.root",
        "tree_258.root",
    ),
)
