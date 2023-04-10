# Multiple deployment config

import itertools


class Config:
    # Dataset declarations
    simulations = {
        'MIMICDeathAKISimultaneousDeploy': {
            'dataset': '../Datasets/MIMIC/MIMIC_Death_AKI.pickle',  # Path to pickle file
            'display_name': 'MIMIC IV: AKI prediction + Mortality prediction',
            'outcomes': ['aki', 'death'],  # Features which correspond to outcomes
            'time_col': 'icu_admin_time',
            'redundant_cols': ['aki_to_death'],
        },
        'SINAIDeathAKISimultaneousDeploy': {
            'dataset': '../Datasets/SINAI/SINAI_DeathAKI.pickle',
            'display_name': 'MSHS: AKI prediction + Mortality prediction',
            'outcomes': ['AKI', 'DEATH'],  # Features which correspond to outcomes
            'time_col': 'ICU_STAY_START',
            'redundant_cols': ['DEATH_AKI_TD'],
        }
    }

    # General
    debug = False

    # Common to all simulations
    effect_sizes = [0, .05, .1, .2, .5, .75, 1]  # This fraction of patients have a manifest effect
    plot_effect_sizes = [0, .1, .2, .5, .75, 1]
    pdist_random_iterations = 64    # Number of random iterations for hamming pdist calculation
                                    # Adjust this by core count
    random_state = 42

    dir_results = '/sc/arion/projects/EHR_ML/akhil/ModelEatModelGPU/MultipleDeployment/Results/'
    dir_plots = '/sc/arion/projects/EHR_ML/akhil/ModelEatModelGPU/MultipleDeployment/Plots/'

    # dir_results = 'Results'
    # dir_plots = 'Plots'

    # Modelling specific
    run_params_default = {
        'model_name': 'XGB',
        'downsample_dataset': False,  # Downsample dataset to have equal number of patients for each (ALL) outcome(s)
        'threshold_calibration': '90SENS',  # Possible options: 'YOUDEN', '90SENS', '90SPEC'
        'first_oc_train_interval': .5,  # Training testing temporal relationship
        'false_positives_render_invalid': True,  # False positives affect downstream validity
        'n_dummies': 1,
        'hyperparameter_optimization': False,
    }

    # Relevant sims only
    relevant = True

    # Plotting
    dpi = 200
    sd_error_multiplier = 1.96  # Standard error multiplier for plotting

    # Fontsizes for axes
    fs_suptitle = 38
    fs_title = 26
    fs_axis = 22
    fs_legend = 17
    fs_subheading = 24


def get_run_params():
    # CRITICAL: These names MUST match the names in the config file
    run_params_pos = {
        'model_name': ['XGB', 'LASSO'],
        'downsample_dataset': [False],  # TODO Remove this entirely if the results make sense
        'threshold_calibration': ['YOUDEN', '90SENS', '90SPEC'],
        'first_oc_train_interval': [.25, .50],
        'false_positives_render_invalid': [True, False],
        'n_dummies': [0, 1],
        'hyperparameter_optimization': [True, False],
    }

    all_run_params = []
    for run_params in (run_params_pos,):
        for values in itertools.product(*map(run_params.get, run_params.keys())):
            all_run_params.append(dict(zip(run_params.keys(), values)))

    # Relevant run params for the manuscript
    run_params_relevant = [
        {
            'model_name': 'XGB', 'downsample_dataset': False, 'threshold_calibration': '90SENS',
            'first_oc_train_interval': .50, 'false_positives_render_invalid': True, 'n_dummies': 1,
            'hyperparameter_optimization': True,
        },
        {
            'model_name': 'XGB', 'downsample_dataset': False, 'threshold_calibration': '90SENS',
            'first_oc_train_interval': .50, 'false_positives_render_invalid': True, 'n_dummies': 1,
            'hyperparameter_optimization': False,
        },
        {
            'model_name': 'XGB', 'downsample_dataset': False, 'threshold_calibration': '90SENS',
            'first_oc_train_interval': .50, 'false_positives_render_invalid': True, 'n_dummies': 0,
            'hyperparameter_optimization': True,
        },
        {
            'model_name': 'XGB', 'downsample_dataset': False, 'threshold_calibration': '90SENS',
            'first_oc_train_interval': .25, 'false_positives_render_invalid': True, 'n_dummies': 1,
            'hyperparameter_optimization': True,
        },
        {
            'model_name': 'XGB', 'downsample_dataset': False, 'threshold_calibration': '90SENS',
            'first_oc_train_interval': .50, 'false_positives_render_invalid': False, 'n_dummies': 1,
            'hyperparameter_optimization': True,
        },
        {
            'model_name': 'LASSO', 'downsample_dataset': False, 'threshold_calibration': '90SENS',
            'first_oc_train_interval': .50, 'false_positives_render_invalid': True, 'n_dummies': 1,
            'hyperparameter_optimization': True,
        },
    ]

    if Config.relevant:
        return run_params_relevant

    return all_run_params


class PRNGSeeds:
    seeds = [
        186607, 669794, 183632, 638880, 714303, 167126, 169208, 590742,
        420450, 580660, 728497,  20405, 585727, 462151,  53076, 529595,
        770584, 455448, 278424, 114584, 471103, 324564, 689151, 746763,
        88926, 634986, 783798, 572285, 723612,  78870, 575222, 434955,
        340249, 581404, 611864, 248010, 406928, 390936, 610558, 308514,
        677486, 641079, 103191, 584210, 253089, 249878, 950433,  48774,
        80504, 505843, 513241, 326231, 384604, 924981,  93195,  94040,
        416646, 951251,  36199, 824163, 831723, 120876, 413239, 611009,
        647404, 278943, 490160,  80040, 784564, 912496, 969614, 567602,
        789471, 775962, 191761, 587849, 543576, 329840, 179351, 980626,
        400554, 530891, 268386, 556623, 251070, 758741, 748995, 568801,
        141462, 157465, 212984, 780645, 368168, 889546, 811714, 465052,
        678584, 839779, 317731, 972137, 345802, 711703, 159803, 632513,
        423967, 258846, 917725, 659498, 793233, 481714, 843596, 841892,
        543113, 979398, 815977, 912500, 520714, 715724, 164746, 464763,
        566412, 347744, 898140, 388559, 710923, 908747, 887167, 530594,
        7425, 156131, 331110, 777106, 245094, 884666, 820297, 581506,
        344841, 319883, 205980, 141999, 804381, 440913, 456159, 820107,
        976869, 773004, 191806, 559803, 654858, 725093, 253127, 100543,
        470346, 854669, 723465, 113148, 670902, 123093, 161475, 975769,
        883890, 834841, 260272, 862336, 768983, 722057, 560201, 780140,
        954585, 235939, 414511, 720114, 951951, 333125, 363596, 112159,
        949165, 111353,  59939, 716957, 312637, 550761, 786223, 679344,
        982157, 586279, 832387, 394637, 891436, 731264, 762571, 453755,
        718743, 772480,  81573, 694251, 195314, 886320, 407116, 395355,
        855445, 809072, 561014, 950223, 822555,  34515, 436395,  83977,
        873877,  40299, 963553, 375428, 214246, 496951,  68971, 503823,
        48576, 536158, 931549, 182702, 484433, 830894, 592198, 128929,
        827273, 791511,  36452, 664002, 925566, 341251, 200567,  17878,
        961955, 803300, 481517, 780793, 620438, 390638, 977286,  49826,
        356712,  90052, 369889, 610160,  33984, 547927, 819259, 979802,
        220592, 890685, 913655, 935897,   5015, 861245, 765265, 317523,
        232063, 384414, 553822, 153010, 812891,   4623, 837524,  70709,
        329751, 724858, 683002, 727588, 570420,  56841, 954966,  95912,
        322145, 980709,  99820, 160084, 480134, 285819, 153304, 793405,
        60309, 366345, 967833, 522203, 776013, 911617, 446925, 757916,
        287032, 773976,  22629, 542477, 240326, 368532,  11766, 607224,
        59772, 101588, 284167, 437909, 994045,  42013, 111963, 744527,
        790914, 860873, 900954, 641361, 942834, 382483,  47699, 316808,
        133896, 871086, 950937, 986998, 444960, 177319, 727820, 629203,
        974852, 542111, 515358, 997085, 941474, 878839, 622980, 229188,
        489366, 183625, 307722, 702419, 249527,  79022, 757800,  74646,
        389798, 147364, 395299, 788547,  54977, 365734, 775275,  42255,
        317000, 245272, 763774, 225169, 498564, 311202, 389754, 130111,
        160830, 641678, 201076, 387131,    722, 197259, 597243, 328308,
        407863, 945032, 458233, 724265, 751118, 655466, 838307, 186466,
        372354, 558757,  12891, 723032, 382940, 342322, 790600, 426955,
        188110, 266264, 764161, 823235, 955734, 892136, 323743, 113063,
        359835, 835046, 860432, 913043, 672044, 972196, 787329, 232815,
        139323, 284204, 631152, 965722, 351135, 911443, 462288, 542955,
        964649, 474631,  87678, 231629, 826740, 873841,  66573, 494213,
        351299, 265338, 575009, 928257, 845256,  77063, 709047, 458409,
        1030, 687650, 382731,  19023, 973411, 924036, 755214, 688900,
        665101, 255307, 903837, 525522, 147956, 819928, 250768, 288099,
        55692, 232921, 849886, 527832, 682969, 772225, 541947, 321562,
        821843, 287671,  76497,  29829, 196135, 372225, 897427, 618619,
        860058, 677521,  41581,  32227, 134877, 586003, 960596,  78880,
        344780, 599809, 113228,  48750, 849478, 617802, 214433, 788142,
        688781, 978903, 164410,  15956, 515606, 858221, 302170, 566525,
        853837, 818801, 496029, 240407, 200459, 518782, 247836, 289486,
        165717, 624940, 908287, 115135,  38424, 456890, 119475, 621708,
        677616, 710120, 972231, 866364, 789445, 325760, 561924, 187374,
        145801, 116373, 350970, 197114]
