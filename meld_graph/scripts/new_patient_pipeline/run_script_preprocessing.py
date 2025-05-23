"""
Pipeline to prepare data from new patients : 
1) combat harmonise (make sure you have computed the combat harmonisation parameters for your site prior)
2) inter & intra normalisation
3) Save data in the "combat" hdf5 matrix
"""

## To run : python run_script_preprocessing.py -harmo_code <harmo_code> -ids <text_file_with_subject_ids> 

import os
import sys
import glob
import json
import argparse
import pandas as pd
import numpy as np
import tempfile
import shutil
from os.path import join as opj
from meld_graph.meld_cohort import MeldCohort
from meld_graph.data_preprocessing import Preprocess, Feature
from meld_graph.tools_pipeline import get_m, create_demographic_file, create_dataset_file
from meld_graph.paths import (
                            BASE_PATH, 
                            MELD_PARAMS_PATH, 
                            MELD_DATA_PATH,
                            DEMOGRAPHIC_FEATURES_FILE,
                            CLIPPING_PARAMS_FILE,
                            NORM_CONTROLS_PARAMS_FILE, 
                            FS_SUBJECTS_PATH,
                            )   

import warnings
warnings.filterwarnings("ignore")

def which_combat_file(harmo_code):
    # file_site=os.path.join(BASE_PATH, f'MELD_{harmo_code}', f'{harmo_code}_combat_parameters.hdf5')
    file_site=os.path.join(BASE_PATH, f'{harmo_code}_combat_parameters.hdf5')
    if os.path.isfile(file_site):
        print(get_m(f'Use combat parameters from site', None, 'INFO'))
        return file_site
    else:
        print(get_m(f'Could not find combat parameters for {harmo_code}', None, 'WARNING'))
        return 'None'

def check_demographic_file(demographic_file, subject_ids):
    #check demographic file has the right columns
    # try:
    df = pd.read_csv(demographic_file, sep='\t')
    #     if not any(ext in ';'.join(df.keys()) for ext in ['participant_id', 'Sex', 'Age at preoperative', 'Group', 'Harmo code', 'Scanner']):
    #         sys.exit(get_m(f'Error with column names', None, 'ERROR'))
    # except Exception as e:
    #     sys.exit(get_m(f'Error with the demographic file provided for the harmonisation\n{e}', None, 'ERROR'))
    #check demographic file has the right subjects
    
    if set(subject_ids).issubset(set(np.array(df['participant_id']))):
        return demographic_file
    else:
        sys.exit(get_m(f'Missing subject in the demographic file', None, 'ERROR'))
    #check variance in age, otherwise combat fails
    df = pd.read_csv(demographic_file)
    ages = df['Age at preoperative (in years)']
    if len(np.unique(ages))<=1:
        sys.exit(get_m(f'There is no variance in the ages provided. Harmonisation will fail', None, 'ERROR'))
    
def run_data_processing_new_subjects(subject_ids, harmo_code, compute_harmonisation = False, harmonisation_only = False, demographic_file=None,  output_dir=BASE_PATH, withoutflair=False):

    # Set features and smoothed values
    if withoutflair:
        features = {
		".on_lh.thickness.mgh": 3,
		".on_lh.w-g.pct.mgh" : 3,
		".on_lh.pial.K_filtered.sm20.mgh": None,
		'.on_lh.sulc.mgh' : 3,
		'.on_lh.curv.mgh' : 3,
			}
    else:
        features = {
		".on_lh.thickness.mgh": 3,
		".on_lh.w-g.pct.mgh" : 3,
		".on_lh.pial.K_filtered.sm20.mgh": None,
		'.on_lh.sulc.mgh' : 3,
		'.on_lh.curv.mgh' : 3,
		'.on_lh.gm_FLAIR_0.25.mgh' : 3,
		'.on_lh.gm_FLAIR_0.5.mgh' : 3,
		'.on_lh.gm_FLAIR_0.75.mgh' :3,
		".on_lh.gm_FLAIR_0.mgh": 3,
		'.on_lh.wm_FLAIR_0.5.mgh' : 3,
		'.on_lh.wm_FLAIR_1.mgh' : 3,
    			}
    feat = Feature()
    features_smooth = [feat.smooth_feat(feature, features[feature]) for feature in features]
    features_combat = [feat.combat_feat(feature) for feature in features_smooth]
    
    ### INITIALISE ### 
        
    #create dataset
    tmp = tempfile.NamedTemporaryFile(mode="w")
    create_dataset_file(subject_ids, tmp.name)  

    ### SMOOTHING ###
    c_raw = MeldCohort(hdf5_file_root="{site_code}_featurematrix.hdf5", dataset=tmp.name, data_dir=BASE_PATH)
    smoothing = Preprocess(c_raw, 
                           site_codes=subject_ids,
                           write_output_file="{site_code}_featurematrix_smoothed.hdf5", 
                           data_dir=output_dir)
    
    #file to store subject with outliers vertices
    outliers_file=opj(output_dir, 'list_subject_extreme_vertices.csv')
    for feature in np.sort(list(set(features))):
        print(get_m(f'Smoothing feature {feature}', None, 'STEP'))
        smoothing.smooth_data(feature, features[feature], clipping_params=CLIPPING_PARAMS_FILE, outliers_file=outliers_file)

    ### REGRESS THICKNESS ###
    if (".on_lh.thickness" in "".join(features_smooth)) and (".on_lh.curv" in "".join(features_smooth)):
        print(get_m(f'Regress thickness with curvature', subject_ids, 'STEP'))
        #create cohort for the new subject
        c_smooth = MeldCohort(hdf5_file_root='{site_code}_featurematrix_smoothed.hdf5', dataset=tmp.name)
        #create object combat
        regress =Preprocess(c_smooth,
                            site_codes=subject_ids,
                            write_output_file='{site_code}_featurematrix_smoothed.hdf5',
                            data_dir=output_dir)
        #features names
        feature = [feat for feat in features_smooth if ".on_lh.thickness" in feat][0]
        curv_feature = [feat for feat in features_smooth if ".on_lh.curv" in feat][0]

        regress.curvature_regress(feature, curv_feature=curv_feature)

        #add features to list
        feat_regress = feat.regress_feat(feature)
        print(f'Add feature {feat_regress} in features')
        features_smooth = features_smooth + [feat_regress]
        features_combat = [feat.combat_feat(feature) for feature in features_smooth]

    print(compute_harmonisation)
    
    if compute_harmonisation:
        
        ### COMPUTE COMBAT PARAMS ###
        #-----------------------------------------------------------------------------------------------
        print(get_m(f'Compute combat harmonisation parameters for new site', None, 'STEP'))
        
        #check enough subjects for harmonisation
        if len(np.unique(subject_ids))<20:
            print(get_m(f'We recommend to use at least 20 subjects for an accurate harmonisation of the data. Here you are using only {len(np.unique(subject_ids))}', None, 'WARNING'))
  
        #create cohort for the new subject
        c_smooth= MeldCohort(hdf5_file_root='{site_code}_featurematrix_smoothed.hdf5', 
                        dataset=tmp.name)
        #create object combat
        combat =Preprocess(c_smooth,
                            site_codes=subject_ids,
                            # write_output_file="MELD_{site_code}/{site_code}_combat_parameters.hdf5",
                            write_output_file="{harmo_code}_combat_parameters.hdf5",
                            data_dir=output_dir)
        #features names
        for feature in features_smooth:
            print(get_m(f'Compute combat parameters feature {feature}', None, 'STEP'))
            combat.get_combat_new_site_parameters(feature, demographic_file, harmo_code)

    if not harmonisation_only:
        
        if harmo_code != 'noHarmo':
            ### COMBAT DATA ###
            #-----------------------------------------------------------------------------------------------
            print(get_m(f'Combat harmonise subjects', subject_ids, 'STEP'))
            # combat_params_file = os.path.join(BASE_PATH, f'MELD_{harmo_code}', f'{harmo_code}_combat_parameters.hdf5')
            combat_params_file = os.path.join(BASE_PATH, f'{harmo_code}_combat_parameters.hdf5')
            #create cohort for the new subject
            c_smooth = MeldCohort(hdf5_file_root='{site_code}_featurematrix_smoothed.hdf5', dataset=tmp.name)
            #create object combat
            combat =Preprocess(c_smooth,
                               site_codes=subject_ids,
                               write_output_file='{site_code}_featurematrix_combat.hdf5',
                               data_dir=output_dir)
            #features names
            for feature in features_smooth:
                print(get_m(f'Combat feature {feature}', None, 'STEP'))
                combat.combat_new_subject(feature, combat_params_file)
        else:
            #transfer smoothed features as combat features
            print(get_m(f'Transfer features - no harmonisation', subject_ids, 'STEP'))
            #create cohort for the new subject
            c_smooth = MeldCohort(hdf5_file_root='{site_code}_featurematrix_smoothed.hdf5', dataset=tmp.name)
            #create object no combat
            nocombat =Preprocess(c_smooth,
                                 site_codes=subject_ids,
                                 write_output_file='{site_code}_featurematrix_combat.hdf5',
                                 data_dir=output_dir)
            #features names
            for feature in features_smooth:
                print(get_m(f'Transfer feature {feature}', None, 'STEP'))
                nocombat.transfer_features_no_combat(feature)

        ###  INTRA, INTER & ASYMETRY ###
        #-----------------------------------------------------------------------------------------------
        print(get_m(f'Intra-inter normalisation & asymmetry subjects', subject_ids, 'STEP'))
        #create cohort to normalise
        c_combat = MeldCohort(hdf5_file_root='{site_code}_featurematrix_combat.hdf5', dataset=tmp.name)
        # provide mean and std parameter for normalisation by controls
        param_norms_file = os.path.join(MELD_PARAMS_PATH, NORM_CONTROLS_PARAMS_FILE.format('nocombat'))
        # create object normalisation
        norm = Preprocess(c_combat,
                          site_codes=subject_ids,
                          write_output_file='{site_code}_featurematrix_combat.hdf5',
                          data_dir=output_dir)
        # call functions to normalise data
        for feature in features_combat:
            print(get_m(f'Normalise feature {feature}', None, 'STEP'))
            norm.intra_inter_subject(feature, params_norm = param_norms_file)
            norm.asymmetry_subject(feature, params_norm = param_norms_file )

        ### PLOT FEATURES FOR QC
        #-----------------------------------------------------------------
        # features_to_plot = [ ".inter_z.asym.intra_z" + feature for feature in features_combat] 
        # c_norm = MeldCohort(hdf5_file_root="{site_code}_featurematrix_combat.hdf5", dataset=tmp.name, data_dir=BASE_PATH)
        # plot = Preprocess(c_norm, 
        #                 site_codes=subject_ids,
        #                 write_output_file=None, 
        #                 data_dir=output_dir)
                
        # print(get_m(f'Plot features to QC', None, 'STEP'))
        # plot.plot_subject_features(features_to_plot)
        
        tmp.close()

def run_script_preprocessing(list_ids=None, sub_id=None, harmo_code='noHarmo', output_dir=BASE_PATH, demographic_file=None, harmonisation_only=False, withoutflair=False, verbose=False):
    harmo_code = str(harmo_code)
    subject_id= sub_id
    subject_ids=None
    if list_ids != None:
        list_ids=opj(MELD_DATA_PATH, list_ids)
        try:
            sub_list_df=pd.read_csv(list_ids)
            subject_ids=np.array(sub_list_df.ID.values)
        except:
            print(get_m(f'⚠️ Could not open CSV with pandas, trying loadtxt: {e}', None, 'WARNING'))
            try:
                subject_ids = np.loadtxt(list_ids, dtype='str', ndmin=1)
            except Exception as e2:
                sys.exit(get_m(f'❌ Could not load list_ids with any method: {e2}', None, 'ERROR'))                     
    elif sub_id != None:
        subject_id=sub_id
        subject_ids=np.array([sub_id])
    else:
        print(get_m(f'No ids were provided', None, 'ERROR'))
        print(get_m(f'Please specify both subject(s) and site_code ...', None, 'ERROR'))
        sys.exit(-1) 

    if harmo_code != 'noHarmo':
        #check that combat parameters exist for this site or compute it
        combat_params_file = which_combat_file(harmo_code)
        if combat_params_file=='None':
            print(get_m(f'New harmonisation code. Compute the harmonisation parameters for {harmo_code} with subjects {subject_ids}', None, 'INFO'))
            #check that demographic file exist and is adequate
            demographic_file = os.path.join(MELD_DATA_PATH, DEMOGRAPHIC_FEATURES_FILE)
            
            if os.path.isfile(demographic_file):
                print(get_m(f'Use demographic file {demographic_file}', None, 'INFO'))
                demographic_file = check_demographic_file(demographic_file, subject_ids)
                compute_harmonisation = True
            else:
                sys.exit(get_m(f'Could not find demographic file provided {demographic_file}. Provide the demographic file with the --demos flag to run the harmonisation', None, 'ERROR'))
        else:
            compute_harmonisation = False
             
    else:
        print(get_m(f'No harmonisation done on the features', None, 'INFO'))
        compute_harmonisation = False
        combat_params_file = None

    #compute the combat parameters for a new site
    run_data_processing_new_subjects(subject_ids, 
                                        harmo_code=harmo_code,
                                        compute_harmonisation = compute_harmonisation,
                                        demographic_file=demographic_file,
                                        harmonisation_only = harmonisation_only,
                                        output_dir=output_dir, 
                                        withoutflair=withoutflair)
        

if __name__ == '__main__':

    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='data-processing on new subject')
    #TODO think about how to best pass a list
    parser.add_argument("-id","--id",
                        help="Subject ID.",
                        default=None,
                        required=False,
                        )
    parser.add_argument("-ids","--list_ids",
                        default=None,
                        help="File containing list of ids. Can be txt or csv with 'participant_id' column",
                        required=False,
                        )
    parser.add_argument("-harmo_code","--harmo_code",
                        default="noHarmo",
                        help="Harmonisation code",
                        required=False,
                        )
    parser.add_argument('-demos', '--demographic_file', 
                        type=str, 
                        help='provide the demographic files for the harmonisation',
                        required=False,
                        default=None,
                        )
    parser.add_argument('--harmo_only', 
                        action="store_true", 
                        help='only compute the harmonisation combat parameters, no further process',
                        required=False,
                        default=False,
                        )
    parser.add_argument("--withoutflair",
                        action="store_true",
                        default=False,
                        help="do not use flair information",
                        )
    parser.add_argument("--debug_mode", 
                        help="mode to debug error", 
                        required=False,
                        default=False,
                        action="store_true",
                        )

    
    args = parser.parse_args()
    print(args)
    
    ### Create demographic file for prediction if not provided
    demographic_file_tmp = DEMOGRAPHIC_FEATURES_FILE
    harmo_code = str(args.harmo_code)
    subject_id=None
    subject_ids=None
    if args.list_ids != None:
        list_ids=os.path.join(MELD_DATA_PATH, args.list_ids)
        print(list_ids)
        try:
            sub_list_df=pd.read_csv(list_ids)
            subject_ids= np.array(sub_list_df.ID.values)
        except:
            print(get_m(f'⚠️ Could not open CSV with pandas, trying loadtxt: {e}', None, 'WARNING'))
            try:
                subject_ids = np.loadtxt(list_ids, dtype='str', ndmin=1)
            except Exception as e2:
                sys.exit(get_m(f'❌ Could not load list_ids with any method: {e2}', None, 'ERROR'))           
    elif args.id != None:
        subject_id=args.id
        subject_ids=np.array([args.id])
    else:
        print(get_m(f'No ids were provided', None, 'ERROR'))
        print(get_m(f'Please specify both subject(s) and site_code ...', None, 'ERROR'))
        sys.exit(-1) 
    
    if args.demographic_file is None:
        create_demographic_file(subject_ids, demographic_file_tmp, harmo_code=harmo_code)
    # else:
    #     shutil.copy(os.path.join(MELD_DATA_PATH), demographic_file_tmp)
    
    df = pd.read_csv(os.path.join(MELD_DATA_PATH, DEMOGRAPHIC_FEATURES_FILE), sep="\t")
    root_dir = 'data/input/ds004199'
    site_scanners = []

    for subj in df['participant_id']:
        subj_path = os.path.join(root_dir, subj)
        manufacturers = set()
        manufacturersModelNames = set()

        for subdir, _, files in os.walk(subj_path):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(subdir, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            manufacturer = data.get("Manufacturer")
                            manufacturersModelName = data.get("ManufacturersModelName")

                            if manufacturer and manufacturersModelName:
                                manufacturers.add(manufacturer)
                                manufacturersModelNames.add(manufacturersModelName)
                    except Exception as e:
                        print(f"Error in preprocessing {file_path}: {e}")

        scanner_id = "_".join(sorted(manufacturers)) + "_" + "_".join(sorted(manufacturersModelNames))
        site_scanners.append(scanner_id)

    df["Scanner"] = site_scanners
    scanners = df["Scanner"].unique()

    subject_list_path = os.path.join(MELD_DATA_PATH, args.list_ids)
    subject_list_df = pd.read_csv(subject_list_path)
    subject_list = set(subject_list_df["ID"].tolist())

    for scanner in scanners:
        df_scanner = df[df["Scanner"] == scanner].copy()
    
        subject_ids = df_scanner["participant_id"]
        subject_ids = [sid for sid in subject_ids if sid in subject_list]
        subject_ids_path = f"data/subject_ids_{scanner}.csv"

        pd.DataFrame(subject_ids, columns=["ID"]).to_csv(subject_ids_path, index=False)

        run_script_preprocessing(
                        harmo_code= args.harmo_code + '_' + scanner,
                        list_ids= f"subject_ids_{scanner}.csv", # args.list_ids,
                        sub_id=subject_id,
                        demographic_file=args.demographic_file,
                        harmonisation_only = args.harmo_only,
                        withoutflair=args.withoutflair,
                        verbose = args.debug_mode,
                        )