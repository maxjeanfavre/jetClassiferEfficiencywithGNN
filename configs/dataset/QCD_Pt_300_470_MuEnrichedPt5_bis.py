from utils.configs.dataset import DatasetConfig


dataset_config = DatasetConfig(
    name="QCD_Pt_300_470_MuEnrichedPt5",
    path=(
        "/pnfs/psi.ch/cms/trivcat/store/user/kadatta/jetObservables/"
        "QCD_Pt-300To470_MuEnrichedPt5_TuneCP5_13TeV-pythia8/"
        "RunIISummer20UL18PFNanov2pt2_jetObsSkim_WtopSelnomV7/240213_121041/0000/"
    ),
    key="Events;1",
    file_limit=78,
    filename_pattern=r"^jetObservables_nanoskim_\d+.root$",
    branches_to_simulate=[
        {
            "name": "Jet_Pt",
            "expression": "Jet_pt",
        }
    ],
    filenames=(
        "jetObservables_nanoskim_1.root",
        "jetObservables_nanoskim_2.root",
        "jetObservables_nanoskim_3.root",
        "jetObservables_nanoskim_4.root",
        "jetObservables_nanoskim_5.root",
        "jetObservables_nanoskim_6.root",
        "jetObservables_nanoskim_7.root",
        "jetObservables_nanoskim_8.root",
        "jetObservables_nanoskim_9.root",
        "jetObservables_nanoskim_10.root",
        "jetObservables_nanoskim_11.root",
        "jetObservables_nanoskim_12.root",
        "jetObservables_nanoskim_13.root",
        "jetObservables_nanoskim_14.root",
        "jetObservables_nanoskim_15.root",
        "jetObservables_nanoskim_16.root",
        "jetObservables_nanoskim_17.root",
        "jetObservables_nanoskim_18.root",
        "jetObservables_nanoskim_19.root",
        "jetObservables_nanoskim_20.root",
        "jetObservables_nanoskim_21.root",
        "jetObservables_nanoskim_22.root",
        "jetObservables_nanoskim_23.root",
        "jetObservables_nanoskim_24.root",
        "jetObservables_nanoskim_25.root",
        "jetObservables_nanoskim_26.root",
        "jetObservables_nanoskim_27.root",
        "jetObservables_nanoskim_28.root",
        "jetObservables_nanoskim_29.root",
        "jetObservables_nanoskim_30.root",
        "jetObservables_nanoskim_31.root",
        "jetObservables_nanoskim_32.root",
        "jetObservables_nanoskim_33.root",
        "jetObservables_nanoskim_34.root",
        "jetObservables_nanoskim_35.root",
        "jetObservables_nanoskim_36.root",
        "jetObservables_nanoskim_37.root",
        "jetObservables_nanoskim_38.root",
        "jetObservables_nanoskim_39.root",
        "jetObservables_nanoskim_40.root",
        "jetObservables_nanoskim_41.root",
        "jetObservables_nanoskim_42.root",
        "jetObservables_nanoskim_43.root",
        "jetObservables_nanoskim_44.root",
        "jetObservables_nanoskim_45.root",
        "jetObservables_nanoskim_46.root",
        "jetObservables_nanoskim_47.root",
        "jetObservables_nanoskim_48.root",
        "jetObservables_nanoskim_49.root",
        "jetObservables_nanoskim_50.root",
        "jetObservables_nanoskim_51.root",
        "jetObservables_nanoskim_52.root",
        "jetObservables_nanoskim_53.root",
        "jetObservables_nanoskim_54.root",
        "jetObservables_nanoskim_55.root",
        "jetObservables_nanoskim_56.root",
        "jetObservables_nanoskim_57.root",
        "jetObservables_nanoskim_58.root",
        "jetObservables_nanoskim_59.root",
        "jetObservables_nanoskim_60.root",
        "jetObservables_nanoskim_61.root",
        "jetObservables_nanoskim_62.root",
        "jetObservables_nanoskim_63.root",
        "jetObservables_nanoskim_64.root",
        "jetObservables_nanoskim_65.root",
        "jetObservables_nanoskim_66.root",
        "jetObservables_nanoskim_67.root",
        "jetObservables_nanoskim_68.root",
        "jetObservables_nanoskim_69.root",
        "jetObservables_nanoskim_70.root",
        "jetObservables_nanoskim_71.root",
        "jetObservables_nanoskim_72.root",
        "jetObservables_nanoskim_73.root",
        "jetObservables_nanoskim_74.root",
        "jetObservables_nanoskim_75.root",
        "jetObservables_nanoskim_76.root",
        "jetObservables_nanoskim_77.root",
        "jetObservables_nanoskim_78.root",
        "jetObservables_nanoskim_79.root",
        "jetObservables_nanoskim_80.root",
        "jetObservables_nanoskim_81.root",
        "jetObservables_nanoskim_82.root",
        "jetObservables_nanoskim_83.root",
        "jetObservables_nanoskim_84.root",
        "jetObservables_nanoskim_85.root",
        "jetObservables_nanoskim_86.root",
        "jetObservables_nanoskim_87.root",
        "jetObservables_nanoskim_88.root",
        "jetObservables_nanoskim_89.root",
        "jetObservables_nanoskim_90.root",
        "jetObservables_nanoskim_91.root",
        "jetObservables_nanoskim_92.root",
        "jetObservables_nanoskim_93.root",
        "jetObservables_nanoskim_94.root",
        "jetObservables_nanoskim_95.root",
        "jetObservables_nanoskim_96.root",
        "jetObservables_nanoskim_97.root",
        "jetObservables_nanoskim_98.root",
        "jetObservables_nanoskim_99.root",
        "jetObservables_nanoskim_100.root",
        "jetObservables_nanoskim_101.root",
        "jetObservables_nanoskim_102.root",
        "jetObservables_nanoskim_103.root",
        "jetObservables_nanoskim_104.root",
        "jetObservables_nanoskim_105.root",
        "jetObservables_nanoskim_106.root",
        "jetObservables_nanoskim_107.root",
        "jetObservables_nanoskim_108.root",
        "jetObservables_nanoskim_109.root",
        "jetObservables_nanoskim_110.root",
        "jetObservables_nanoskim_111.root",
        "jetObservables_nanoskim_112.root",
        "jetObservables_nanoskim_113.root",
        "jetObservables_nanoskim_114.root",
        "jetObservables_nanoskim_115.root",
        "jetObservables_nanoskim_116.root",
        "jetObservables_nanoskim_117.root",
        "jetObservables_nanoskim_118.root",
        "jetObservables_nanoskim_119.root",
        "jetObservables_nanoskim_120.root",
        "jetObservables_nanoskim_121.root",
        "jetObservables_nanoskim_122.root",
        "jetObservables_nanoskim_123.root",
        "jetObservables_nanoskim_124.root",
        "jetObservables_nanoskim_125.root",
        "jetObservables_nanoskim_126.root",
        "jetObservables_nanoskim_127.root",
        "jetObservables_nanoskim_128.root",
        "jetObservables_nanoskim_129.root",
        "jetObservables_nanoskim_130.root",
        "jetObservables_nanoskim_131.root",
        "jetObservables_nanoskim_132.root",
        "jetObservables_nanoskim_133.root",
        "jetObservables_nanoskim_134.root",
        "jetObservables_nanoskim_135.root",
        "jetObservables_nanoskim_136.root",
        "jetObservables_nanoskim_137.root",
        "jetObservables_nanoskim_138.root",
        "jetObservables_nanoskim_139.root",
        "jetObservables_nanoskim_140.root",
        "jetObservables_nanoskim_141.root",
        "jetObservables_nanoskim_142.root",
        "jetObservables_nanoskim_143.root",
        "jetObservables_nanoskim_144.root",
        "jetObservables_nanoskim_145.root",
        "jetObservables_nanoskim_146.root",
        "jetObservables_nanoskim_147.root",
        "jetObservables_nanoskim_148.root",
        "jetObservables_nanoskim_149.root",
        "jetObservables_nanoskim_150.root",
        "jetObservables_nanoskim_151.root",
        "jetObservables_nanoskim_152.root",
        "jetObservables_nanoskim_153.root",
        "jetObservables_nanoskim_154.root",
        "jetObservables_nanoskim_155.root",
        "jetObservables_nanoskim_156.root",
        "jetObservables_nanoskim_157.root",
        "jetObservables_nanoskim_158.root",
        "jetObservables_nanoskim_159.root",
        "jetObservables_nanoskim_160.root",
        "jetObservables_nanoskim_161.root",
        "jetObservables_nanoskim_162.root",
        "jetObservables_nanoskim_163.root",
        "jetObservables_nanoskim_164.root",
        "jetObservables_nanoskim_165.root",
        "jetObservables_nanoskim_166.root",
        "jetObservables_nanoskim_167.root",
        "jetObservables_nanoskim_168.root",
        "jetObservables_nanoskim_169.root",
        "jetObservables_nanoskim_170.root",
        "jetObservables_nanoskim_171.root",
        "jetObservables_nanoskim_172.root",
        "jetObservables_nanoskim_173.root",
        "jetObservables_nanoskim_174.root",
        "jetObservables_nanoskim_175.root",
        "jetObservables_nanoskim_176.root",
        "jetObservables_nanoskim_177.root",
        "jetObservables_nanoskim_178.root",
        "jetObservables_nanoskim_179.root",
        "jetObservables_nanoskim_180.root",
        "jetObservables_nanoskim_181.root",
        "jetObservables_nanoskim_182.root",
        "jetObservables_nanoskim_183.root",
        "jetObservables_nanoskim_184.root",
        "jetObservables_nanoskim_185.root",
        "jetObservables_nanoskim_186.root",
        "jetObservables_nanoskim_187.root",
        "jetObservables_nanoskim_188.root",
        "jetObservables_nanoskim_189.root",
        "jetObservables_nanoskim_190.root",
        "jetObservables_nanoskim_191.root",
        "jetObservables_nanoskim_192.root",
        "jetObservables_nanoskim_193.root",
        "jetObservables_nanoskim_194.root",
        "jetObservables_nanoskim_195.root",
        "jetObservables_nanoskim_196.root",
        "jetObservables_nanoskim_197.root",
        "jetObservables_nanoskim_198.root",
        "jetObservables_nanoskim_199.root",
        "jetObservables_nanoskim_200.root",
        "jetObservables_nanoskim_201.root",
        "jetObservables_nanoskim_202.root",
        "jetObservables_nanoskim_203.root",
        "jetObservables_nanoskim_204.root",
        "jetObservables_nanoskim_205.root",
        "jetObservables_nanoskim_206.root",
        "jetObservables_nanoskim_207.root",
        "jetObservables_nanoskim_208.root",
        "jetObservables_nanoskim_209.root",
        "jetObservables_nanoskim_210.root",
        "jetObservables_nanoskim_211.root",
        "jetObservables_nanoskim_212.root",
        "jetObservables_nanoskim_213.root",
        "jetObservables_nanoskim_214.root",
        "jetObservables_nanoskim_215.root",
        "jetObservables_nanoskim_216.root",
        "jetObservables_nanoskim_217.root",
        "jetObservables_nanoskim_218.root",
        "jetObservables_nanoskim_219.root",
        "jetObservables_nanoskim_220.root",
        "jetObservables_nanoskim_221.root",
        "jetObservables_nanoskim_222.root",
        "jetObservables_nanoskim_223.root",
        "jetObservables_nanoskim_224.root",
        "jetObservables_nanoskim_225.root",
        "jetObservables_nanoskim_226.root",
        "jetObservables_nanoskim_227.root",
        "jetObservables_nanoskim_228.root",
        "jetObservables_nanoskim_229.root",
        "jetObservables_nanoskim_230.root",
        "jetObservables_nanoskim_231.root",
        "jetObservables_nanoskim_232.root",
        "jetObservables_nanoskim_233.root",
        "jetObservables_nanoskim_234.root",
        "jetObservables_nanoskim_235.root",
        "jetObservables_nanoskim_236.root",
        "jetObservables_nanoskim_237.root",
        "jetObservables_nanoskim_238.root",
        "jetObservables_nanoskim_239.root",
        "jetObservables_nanoskim_240.root",
        "jetObservables_nanoskim_241.root",
        "jetObservables_nanoskim_242.root",
        "jetObservables_nanoskim_243.root",
        "jetObservables_nanoskim_244.root",
        "jetObservables_nanoskim_245.root",
        "jetObservables_nanoskim_246.root",
        "jetObservables_nanoskim_247.root",
        "jetObservables_nanoskim_248.root",
        "jetObservables_nanoskim_249.root",
        "jetObservables_nanoskim_250.root",
        "jetObservables_nanoskim_251.root",
        "jetObservables_nanoskim_252.root",
        "jetObservables_nanoskim_253.root",
        "jetObservables_nanoskim_254.root",
        "jetObservables_nanoskim_255.root",
        "jetObservables_nanoskim_256.root",
        "jetObservables_nanoskim_257.root",
        "jetObservables_nanoskim_258.root",
        "jetObservables_nanoskim_259.root",
        "jetObservables_nanoskim_260.root",
        "jetObservables_nanoskim_261.root",
        "jetObservables_nanoskim_262.root",
        "jetObservables_nanoskim_263.root",
        "jetObservables_nanoskim_264.root",
        "jetObservables_nanoskim_265.root",
        "jetObservables_nanoskim_266.root",
        "jetObservables_nanoskim_267.root",
        "jetObservables_nanoskim_268.root",
        "jetObservables_nanoskim_269.root",
        "jetObservables_nanoskim_270.root",
        "jetObservables_nanoskim_271.root",
        "jetObservables_nanoskim_272.root",
        "jetObservables_nanoskim_273.root",
        "jetObservables_nanoskim_274.root",
        "jetObservables_nanoskim_275.root",
        "jetObservables_nanoskim_276.root",
        "jetObservables_nanoskim_277.root",
        "jetObservables_nanoskim_278.root",
        "jetObservables_nanoskim_279.root",
        "jetObservables_nanoskim_280.root",
        "jetObservables_nanoskim_281.root",
        "jetObservables_nanoskim_282.root",
        "jetObservables_nanoskim_283.root",
        "jetObservables_nanoskim_284.root",
        "jetObservables_nanoskim_285.root",
        "jetObservables_nanoskim_286.root",
        "jetObservables_nanoskim_287.root",
        "jetObservables_nanoskim_288.root",
        "jetObservables_nanoskim_289.root",
        "jetObservables_nanoskim_290.root",
        "jetObservables_nanoskim_291.root",
        "jetObservables_nanoskim_292.root",
        "jetObservables_nanoskim_293.root",
        "jetObservables_nanoskim_294.root",
        "jetObservables_nanoskim_295.root",
        "jetObservables_nanoskim_296.root",
        "jetObservables_nanoskim_297.root",
        "jetObservables_nanoskim_298.root",
        "jetObservables_nanoskim_299.root",
        "jetObservables_nanoskim_300.root",
        "jetObservables_nanoskim_301.root",
        "jetObservables_nanoskim_302.root",
        "jetObservables_nanoskim_303.root",
        "jetObservables_nanoskim_304.root",
        "jetObservables_nanoskim_305.root",
        "jetObservables_nanoskim_306.root",
        "jetObservables_nanoskim_307.root",
        "jetObservables_nanoskim_308.root",
        "jetObservables_nanoskim_309.root",
        "jetObservables_nanoskim_310.root",
        "jetObservables_nanoskim_311.root",
        "jetObservables_nanoskim_312.root",
        "jetObservables_nanoskim_313.root",
        "jetObservables_nanoskim_314.root",
        "jetObservables_nanoskim_315.root",
        "jetObservables_nanoskim_316.root",
        "jetObservables_nanoskim_317.root",
        "jetObservables_nanoskim_318.root",
        "jetObservables_nanoskim_319.root",
        "jetObservables_nanoskim_320.root",
        "jetObservables_nanoskim_321.root",
        "jetObservables_nanoskim_322.root",
        "jetObservables_nanoskim_323.root",
        "jetObservables_nanoskim_324.root",
        "jetObservables_nanoskim_325.root",
        "jetObservables_nanoskim_326.root",
        "jetObservables_nanoskim_327.root",
        "jetObservables_nanoskim_328.root",
        "jetObservables_nanoskim_329.root",
        "jetObservables_nanoskim_330.root",
        "jetObservables_nanoskim_331.root",
        "jetObservables_nanoskim_332.root",
        "jetObservables_nanoskim_333.root",
        "jetObservables_nanoskim_334.root",
        "jetObservables_nanoskim_335.root",
        "jetObservables_nanoskim_336.root",
        "jetObservables_nanoskim_337.root",
        "jetObservables_nanoskim_338.root",
        "jetObservables_nanoskim_339.root",
        "jetObservables_nanoskim_340.root",
        "jetObservables_nanoskim_341.root",
        "jetObservables_nanoskim_342.root",
        "jetObservables_nanoskim_343.root",
        "jetObservables_nanoskim_344.root",
        "jetObservables_nanoskim_345.root",
        "jetObservables_nanoskim_346.root",
        "jetObservables_nanoskim_347.root",
        "jetObservables_nanoskim_348.root",
        "jetObservables_nanoskim_349.root",
        "jetObservables_nanoskim_350.root",
        "jetObservables_nanoskim_351.root",
        "jetObservables_nanoskim_352.root",
        "jetObservables_nanoskim_353.root",
        "jetObservables_nanoskim_354.root",
        "jetObservables_nanoskim_355.root",
        "jetObservables_nanoskim_356.root",
        "jetObservables_nanoskim_357.root",
        "jetObservables_nanoskim_358.root",
        "jetObservables_nanoskim_359.root",
        "jetObservables_nanoskim_360.root",
        "jetObservables_nanoskim_361.root",
        "jetObservables_nanoskim_362.root",
        "jetObservables_nanoskim_363.root",
        "jetObservables_nanoskim_364.root",
        "jetObservables_nanoskim_365.root",
        "jetObservables_nanoskim_366.root",
        "jetObservables_nanoskim_367.root",
        "jetObservables_nanoskim_368.root",
        "jetObservables_nanoskim_369.root",
        "jetObservables_nanoskim_370.root",
        "jetObservables_nanoskim_371.root",
        "jetObservables_nanoskim_372.root",
        "jetObservables_nanoskim_373.root",
        "jetObservables_nanoskim_374.root",
        "jetObservables_nanoskim_375.root",
        "jetObservables_nanoskim_376.root",
        "jetObservables_nanoskim_377.root",
        "jetObservables_nanoskim_378.root",
        "jetObservables_nanoskim_379.root",
        "jetObservables_nanoskim_380.root",
        "jetObservables_nanoskim_381.root",
        "jetObservables_nanoskim_382.root",
        "jetObservables_nanoskim_383.root",
        "jetObservables_nanoskim_384.root",
        "jetObservables_nanoskim_385.root",
        "jetObservables_nanoskim_386.root",
        "jetObservables_nanoskim_387.root",
        "jetObservables_nanoskim_388.root",
        "jetObservables_nanoskim_389.root",
        "jetObservables_nanoskim_390.root",
        "jetObservables_nanoskim_391.root",
        "jetObservables_nanoskim_392.root",
        "jetObservables_nanoskim_393.root",
        "jetObservables_nanoskim_394.root",
        "jetObservables_nanoskim_395.root",
        "jetObservables_nanoskim_396.root",
        "jetObservables_nanoskim_397.root",
        "jetObservables_nanoskim_398.root",
        "jetObservables_nanoskim_399.root",
        "jetObservables_nanoskim_400.root",
    ),
)
