## This script runs the MELD surface-based FCD classifier on the patient using the output features from script 2.
## The predicted clusters are then saved as file " " in the /output/<pat_id>/xhemi/classifier folder
## The predicted clusters are then registered back to native space and saved as a .mgh file in the /output/<pat_id>/classifier folder
## The predicted clusters are then registered back to the nifti volume and saved as nifti in the input/<pat_id>/predictions folder
## Individual reports for each identified cluster are calculated and saved in the input/<pat_id>/predictions/reports folder
## These contain images of the clusters on the surface and on the volumetric MRI as well as saliency reports
## The saliency reports include the z-scored feature values and how "salient" they were to the classifier

## To run : python run_script_prediction.py -ids <text_file_with_ids> -harmo_code <harmo_code>


import json
import os
import subprocess
import sys

import numpy as np

if not hasattr(np, "float"):
    np.float = float

import argparse
import shutil
import tempfile
import warnings
from os.path import join as opj

import nibabel as nb
import pandas as pd

import scripts.env_setup
from meld_graph.evaluation import Evaluator
from meld_graph.experiment import Experiment
from meld_graph.meld_cohort import MeldCohort
from meld_graph.paths import (BASE_PATH,
                              DEFAULT_HDF5_FILE_ROOT,
                              DEMOGRAPHIC_FEATURES_FILE, EXPERIMENT_PATH,
                              FS_SUBJECTS_PATH, MELD_DATA_PATH, MODEL_PATH)
from meld_graph.tools_pipeline import (create_dataset_file,
                                       create_demographic_file, get_m)
from scripts.manage_results.move_predictions_to_mgh import \
    move_predictions_to_mgh
from scripts.manage_results.plot_prediction_report import \
    generate_prediction_report
from scripts.manage_results.register_back_to_xhemi import \
    register_subject_to_xhemi

warnings.filterwarnings("ignore")

def save_surface_mgh(arr_1d, out_path: str):
    arr = np.asarray(arr_1d).ravel().astype(np.float32, copy=False)
    data3d = arr[:, None, None]          # (N, 1, 1) — НЕ (N,1,1,1)!
    img = nb.freesurfer.mghformat.MGHImage(data3d, np.eye(4))
    nb.save(img, out_path)               # .mgh или .mgz — без разницы

def predict_subjects(subject_ids, output_dir, plot_images = False, saliency=False,
    experiment_path=EXPERIMENT_PATH, hdf5_file_root= DEFAULT_HDF5_FILE_ROOT, aug_mode='test', return_results=False):       
    ''' function to predict on new subject using trained MELD classifier'''

    hdf5_file_root = "{site_code}_{group}_featurematrix_combat.hdf5"
    # create dataset csv
    tmp = tempfile.NamedTemporaryFile(mode="w")
    create_dataset_file(subject_ids, tmp.name)

    # load models
    print(experiment_path)
    exp = Experiment.from_folder(experiment_path)
    
    #update experiment 
    exp.cohort = MeldCohort(hdf5_file_root=hdf5_file_root, dataset=tmp.name, data_dir=BASE_PATH)
    exp.data_parameters["hdf5_file_root"] = hdf5_file_root
    exp.data_parameters["dataset"] = tmp.name
    if aug_mode == 'test':
        exp.data_parameters["augment_data"] = {}
    exp.experiment_path = experiment_path
    
    # launch evaluation
    cohort = MeldCohort(
                hdf5_file_root=exp.data_parameters["hdf5_file_root"],
                dataset=exp.data_parameters["dataset"],
                data_dir=BASE_PATH
            )
    
    subject_id = subject_ids[0]
    subject_output_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)

    eva = Evaluator(
        experiment=exp,
        checkpoint_path=experiment_path,
        cohort=cohort,
        subject_ids=subject_ids,
        save_dir=subject_output_dir,#output_dir,
        aug_mode=aug_mode,
        mode=aug_mode,#"test", Mode should be the same as aug_mode
        model_name="best_model",
        threshold='slope_threshold',
        thresh_and_clust=True,
        saliency=saliency,
        make_images=plot_images,
        
    )

    #predict for the dataset
    eva.load_predict_data(
        save_prediction=True,
        roc_curves_thresholds=None,
        )

    # eva.load_predict_data(save_prediction=False)

    # # вручную сохраняем predictions для каждого пациента
    # for subject_id in subject_ids:
    #     pred = eva.data_dictionary[subject_id]["result"]
    #     eva.save_prediction_for_subject(subject_id, pred, dataset_str="prediction_clustered")


    results_dict = {}
    for subject_id in subject_ids:  
        features        = eva.data_dictionary[subject_id]["feature_maps"]
        result          = eva.data_dictionary[subject_id]["result"]
        labels          = eva.data_dictionary[subject_id]["full_labels"]
        input_labels    = eva.data_dictionary[subject_id]["input_labels"] # labels after cortex mask
        dist_map_gt     = eva.data_dictionary[subject_id]["geodesic_distance"]
        # xyzr_gt         = eva.data_dictionary[subject_id]["xyzr"]
        
        if return_results:
            # складываем всё в словарь Python
            results_dict[subject_id] = {
                "features": {k: v.detach().cpu().numpy() for k, v in features.items()},
                "result": result if isinstance(result, np.ndarray) else {
                    k: v.detach().cpu().numpy() for k, v in result.items()
                },
                # "labels": np.asarray(labels),
                # "input_labels": np.asarray(input_labels),
                # "dist_map_gt": np.asarray(dist_map_gt),
            }
        else:
            labels_np = np.asarray(labels).reshape(-1)  # (2*N,)

            assert labels_np.size % 2 == 0, f"Unexpected labels length: {labels_np.size}"
            n_hemi = labels_np.size // 2


            lh = labels_np[:n_hemi].astype(np.float32)
            rh = labels_np[n_hemi:].astype(np.float32)
            
            # estimates    = eva.data_dictionary[subject_id]["estimates"]
            # используем тот output_dir, который пришёл в функцию
            save_dir = f"./data/preprocessed/meld_files/{subject_id}/features"
            print(save_dir)
            os.makedirs(save_dir, exist_ok=True)

            feat_path = os.path.join(save_dir, "feature_maps.npz")
            np.savez_compressed(
                feat_path,
                **{stage: tensor.detach().cpu().numpy() for stage, tensor in features.items()}
            )
            print(f"Saved features to {feat_path}")

            dist_maps_path = os.path.join(save_dir, "distance_maps_gt.npz")
            np.savez(dist_maps_path, dist_map_gt)
            print(f"Saved distance maps GT to {dist_maps_path}")

            # xyzr_path = os.path.join(save_dir, "xyzr_gt.npz")
            # np.savez(xyzr_path, xyzr_gt)
            # print(f"Saved xyzr GT to {xyzr_path}")

            # сохраняем labels в .mgh или .mgz (можно .mgz — будет меньше весить)
            lab_dir  = os.path.join(f"./data/preprocessed/meld_files/{subject_id}", "labels")
            lh_path = os.path.join(lab_dir, "labels-lh.mgh")  # или "labels-lh.mgz"
            rh_path = os.path.join(lab_dir, "labels-rh.mgh")  # или "labels-rh.mgz"
            os.makedirs(lab_dir, exist_ok=True)
            save_surface_mgh(lh, lh_path)
            save_surface_mgh(rh, rh_path)
            print(f"Saved labels to {lh_path} and {rh_path}")

            input_labels_path = os.path.join(lab_dir, "labels_gt.npz")
            np.savez(input_labels_path, input_labels)
            print(f"Saved distance maps GT to {input_labels_path}")
            meta = {
                "n_vertices": int(n_hemi),
                "lh_sum": float(lh.sum()),
                "rh_sum": float(rh.sum()),
                "lh_has_lesion": bool(lh.sum() > 0),
                "rh_has_lesion": bool(rh.sum() > 0),
            }
            
            np.savez(os.path.join(lab_dir, "labels-meta.npz"), **meta)
            # estimates_path = os.path.join(save_dir, "estimates.npz")
            # np.savez_compressed(estimates_path, 
            #                     estimates)

            res_path = os.path.join(save_dir, "result.npz")
            if isinstance(result, np.ndarray):
                np.savez_compressed(res_path, result=result)        
            elif isinstance(result, dict):
                np.savez_compressed(
                    res_path,
                    **{f"pred_{hemi}": arr.detach().cpu().numpy()
                    for hemi, arr in result.items()}
                )
            else:
                raise TypeError(f"Unexpected type result: {type(result)}")

            print(f"Saved prediction result to {res_path}")

    if return_results:
        return results_dict
    #threshold predictions
    eva.threshold_and_cluster()
    #write results in csv
    eva.stat_subjects()
    #plot images 
    # if plot_images: 
    #     eva.plot_subjects_prediction()
    #compute saliency:
    # if saliency:
    #     eva.calculate_saliency()
    # return None

def run_script_prediction(list_ids=None, sub_id=None, harmo_code='noHarmo', no_prediction_nifti=False, no_report=False, skip_prediction=False, split=False, verbose=False, aug_mode='test', return_results=False):
    harmo_code = str(harmo_code)
    subject_id=None
    subject_ids=None
    if list_ids != None:
        list_ids=opj(MELD_DATA_PATH, list_ids)
        try:
            sub_list_df=pd.read_csv(list_ids)
            subject_ids=np.array(sub_list_df.ID.values)
        except:
            subject_ids=np.array(np.loadtxt(list_ids, dtype='str', ndmin=1)) 
        # else:
        #     sys.exit(get_m(f'Could not open {subject_ids}', None, 'ERROR'))       
    elif sub_id != None:
        subject_id=sub_id
        subject_ids=np.array([sub_id])
    else:
        print(get_m(f'No ids were provided', None, 'ERROR'))
        print(get_m(f'Please specify both subject(s) and harmonisation code ...', None, 'ERROR'))
        sys.exit(-1) 
    
    # ВРЕМЕННО    
    # subject_ids = np.array([s for s in subject_ids if any(h in s for h in ["H26_", "H27_", "H28_", "H101_"])])
    # initialise variables
    model_name = MODEL_PATH
    experiment_path = os.path.join(EXPERIMENT_PATH, model_name)
    # subjects_dir = FS_SUBJECTS_PATH
    classifier_output_dir = opj(MELD_DATA_PATH,'output','classifier_outputs', model_name)
    # data_dir = opj(MELD_DATA_PATH,'input')
    # predictions_output_dir = opj(MELD_DATA_PATH,'output','predictions_reports')
    # prediction_file = opj(classifier_output_dir, 'results_best_model', 'predictions.hdf5')
    
    subject_ids_failed=[]

    #predict on new subjects
    results = {}
    if not skip_prediction:
        print(get_m(f'Run predictions', subject_ids, 'STEP 1'))
        for subject_id in subject_ids:
            if subject_id in [
                # Missing controls (no HDF5 file found)
                "MELD_H3_3T_C_0007",
                "MELD_H3_3T_C_0065",
                "MELD_H4_3T_C_0006",
                "MELD_H4_3T_C_0020",
                "MELD_H9_3T_C_0006",
                "MELD_H10_3T_C_0001",
                "MELD_H101_3T_C_00160",
                "MELD_H101_3T_C_C0007",

                # Missing patients (no HDF5 file found)
                "MELD_H5_3T_FCD_0027",
                "MELD_H6_3T_FCD_0007",
                "MELD_H6_3T_FCD_0016",
                "MELD_H16_3T_FCD_044",
                "MELD_H17_3T_FCD_0110",
                "MELD_H17_3T_FCD_0138",
                "MELD_H18_3T_FCD_0112",
                "MELD_H19_3T_FCD_003",
                "MELD_H19_3T_FCD_004",
                "MELD_H28_3T_FCD_0003",
                "MELD_H28_3T_FCD_0008",
                "MELD_H28_3T_FCD_0015",
                "MELD_H28_3T_FCD_0017",
                "MELD_H28_3T_FCD_0018",
                "MELD_H101_3T_FCD_00014",
                "MELD_H101_3T_FCD_00078",

                # No lesion mask present (HDF5 exists but lesion data missing)
                "MELD_H5_3T_FCD_0006",
                "MELD_H5_3T_FCD_0009",
                "MELD_H5_3T_FCD_0012",
                "MELD_H5_3T_FCD_0013",
                "MELD_H5_3T_FCD_0018",
                "MELD_H5_3T_FCD_0025",
                "MELD_H5_3T_FCD_0028",
                "MELD_H5_3T_FCD_0032",
                "MELD_H6_3T_FCD_0001",
                "MELD_H6_3T_FCD_0002",
                "MELD_H6_3T_FCD_0003",
                "MELD_H6_3T_FCD_0008",
                "MELD_H6_3T_FCD_0014",
                "MELD_H6_3T_FCD_0015",
                "MELD_H6_3T_FCD_0019",
                "MELD2_H7_3T_FCD_002",
                "MELD2_H7_3T_FCD_011",
                "MELD2_H7_3T_FCD_014",
                "MELD_H12_15T_FCD_0001",
                "MELD_H12_15T_FCD_0002",
                "MELD_H12_15T_FCD_0003",
                "MELD_H12_15T_FCD_0005",
                "MELD_H12_3T_FCD_0020",
                "MELD_H12_3T_FCD_0021",
                "MELD_H12_3T_FCD_0022",
                "MELD_H12_3T_FCD_0023",
                "MELD_H12_3T_FCD_0024",
                "MELD_H12_3T_FCD_0025",
                "MELD_H12_3T_FCD_0026",
                "MELD_H12_3T_FCD_0028",
                "MELD_H12_3T_FCD_0031",

                # Zero mask after converting to Nifti
                "MELD_H9_3T_FCD_0003",

                # Missed by me <- move from here later
                "MELD_H101_3T_FCD_00062",
            ]:
                continue
            

            preproc_root = os.path.join("./data", "preprocessed", "meld_files", subject_id)
            if os.path.isdir(preproc_root):
                print(f"[SKIP] {subject_id} уже обработан (папка {preproc_root} существует)")
                continue

            result = predict_subjects(subject_ids=np.array([subject_id]), 
                            output_dir=classifier_output_dir,  
                            plot_images=True, 
                            saliency=True,
                            experiment_path=experiment_path, 
                            hdf5_file_root= DEFAULT_HDF5_FILE_ROOT,
                            aug_mode=aug_mode,
                            return_results=return_results)
            if result is not None:
                results.update(result)   
    else:
        print(get_m(f'Skip predictions', subject_ids, 'STEP 1'))
    
    # if not no_prediction_nifti:        
    #     #Register predictions to native space
    #     for i, subject_id in enumerate(subject_ids):
    #         print(get_m(f'Move predictions into volume', subject_id, 'STEP 2'))
    #         if subject_id != "MELD_H101_3T_C_00005":
    #             continue

    #         prediction_file = opj(classifier_output_dir, subject_id,'results_best_model', 'predictions.hdf5')
    #         # print(subjects_dir)
    #         # sys.exit(0)
    #         result = move_predictions_to_mgh(subject_id=subject_id, 
    #                             subjects_dir=subjects_dir, 
    #                             prediction_file=prediction_file,
    #                             verbose=verbose)

    #         if result == False:
    #             print(get_m(f'One step of the pipeline has failed. Process has been aborted for this subject', subject_id, 'ERROR'))
    #             subject_ids_failed.append(subject_id)
    #             continue
            
    #         #Register prediction back to nifti volume
    #         print(get_m(f'Move prediction back to native space', subject_id, 'STEP 3'))
    #         result = register_subject_to_xhemi(subject_id=subject_id, 
    #                                     subjects_dir=subjects_dir, 
    #                                     output_dir=predictions_output_dir, 
    #                                     verbose=verbose)
    #         if result == False:
    #             print(get_m(f'One step of the pipeline has failed. Process has been aborted for this subject', subject_id, 'ERROR'))
    #             subject_ids_failed.append(subject_id)
    #             continue

    #         break
            
    # if not no_report:
    #     # Create individual reports of each identified cluster
    #     print(subject_ids[0])
    #     if subject_ids[0] == "MELD_H2_3T_FCD_0001":
    #         print(get_m(f'Create pdf report', subject_ids, 'STEP 4'))
    #         generate_prediction_report(
    #             subject_ids = subject_ids,
    #             data_dir = data_dir,
    #             prediction_path=classifier_output_dir,
    #             experiment_path=experiment_path, 
    #             output_dir = predictions_output_dir,
    #             harmo_code = harmo_code,
    #             hdf5_file_root = DEFAULT_HDF5_FILE_ROOT
    #         )
    #     sys.exit(0)
    
        
    if len(subject_ids_failed)>0:
        print(get_m(f'One step of the pipeline has failed and process has been aborted for subjects {subject_ids_failed}', None, 'ERROR'))
        return False
    
    if return_results:
        return results
    return None

if __name__ == '__main__':
    scripts.env_setup.setup()

    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-id","--id",
                        help="Subject ID.",
                        default=None,
                        required=False,
                        )
    parser.add_argument("-ids","--list_ids",
                        default=None,
                        help="File containing list of ids. Can be txt or csv with 'ID' column",
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
    parser.add_argument('--no_prediction_nifti',
                        action="store_true",
                        help='Only predict. Does not produce prediction on native T1, nor report',
                        )
    parser.add_argument('--no_report',
                        action="store_true",
                        help='Predict and map back into native T1. Does not produce report',)
    parser.add_argument('--skip_prediction',
                        action="store_true",
                        help='Skip prediction and go straight to registration and report.',)
    parser.add_argument('--split',
                        action="store_true",
                        help='Split subjects list in chunk to avoid data overload',
                        )
    parser.add_argument("--debug_mode", 
                        help="mode to debug error", 
                        required=False,
                        default=False,
                        action="store_true",
                        )
    parser.add_argument("--aug_mode",
                        help="Make augmentation for training data or not",
                        required=True,
                        default='test')
    parser.add_argument("--return_results",
                        help="For FastAPI: return results as dictionary",
                        default=False)
    args = parser.parse_args()
    print(args) 
    
    ### Create demographic file for prediction if not provided
    demographic_file_tmp = os.path.join(MELD_DATA_PATH, f"input/data4sharing/demographics_file_tmp.csv") #DEMOGRAPHIC_FEATURES_FILE
    if args.demographic_file is None:
        harmo_code = str(args.harmo_code)
        subject_id=None
        subject_ids=None
        if args.list_ids != None:
            list_ids=os.path.join(MELD_DATA_PATH, args.list_ids)
            try:
                sub_list_df=pd.read_csv(list_ids)
                subject_ids=np.array(sub_list_df.ID.values)
            except:
                subject_ids=np.array(np.loadtxt(list_ids, dtype='str', ndmin=1)) 
            # else:
            #     print('Else')
            #     sys.exit(get_m(f'Could not open {subject_ids}', None, 'ERROR'))             
        elif args.id != None:
            subject_id=args.id
            subject_ids=np.array([args.id])
        else:
            print(get_m(f'No ids were provided', None, 'ERROR'))
            print(get_m(f'Please specify both subject(s) and site_code ...', None, 'ERROR'))
            sys.exit(-1) 
        create_demographic_file(subject_ids, demographic_file_tmp, harmo_code=harmo_code)
    else:
        print(MELD_DATA_PATH)
        print(os.path.join(MELD_DATA_PATH,args.demographic_file))
        os.makedirs(os.path.dirname(demographic_file_tmp), exist_ok=True)
        # print(os.path.dirname(demographic_file_tmp))
        shutil.copy(os.path.join(MELD_DATA_PATH,args.demographic_file), demographic_file_tmp)
    
    # Run: ./meldgraph.sh run_script_prediction_meld.py --list_ids /home/s17gmikh/FCD-Detection/meld_graph/data/input/data4sharing/demographics_qc_allgroups_withH27H28H101.csv --demographic_file /home/s17gmikh/FCD-Detection/meld_graph/data/input/data4sharing/demographics_qc_allgroups_withH27H28H101.csv --aug_mode train
    results = run_script_prediction(
                        harmo_code = args.harmo_code,
                        list_ids=args.list_ids,
                        sub_id=args.id,
                        no_prediction_nifti = args.no_prediction_nifti,
                        no_report = args.no_report,
                        split = args.split,
                        skip_prediction=args.skip_prediction,
                        verbose = args.debug_mode,
                        aug_mode=args.aug_mode,
                        return_results=args.return_results
                        )
    
    if args.return_results and results is not None:
        import pickle

        # базовая директория для всех результатов
        base_dir = "/backend/data_results"
        os.makedirs(base_dir, exist_ok=True)

        # если results — список, берём первый словарь
        if isinstance(results, list):
            results = results[0]

        # берём subject_id (он у тебя ключ в словаре)

        # создаём папку для конкретного пациента
        subject_dir = os.path.join(base_dir, *results.keys())  # достаёт id без хардкода
        os.makedirs(subject_dir, exist_ok=True)

        out_path = os.path.join(subject_dir, "results.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(results, f)


        # выводим в stdout только путь
        print(out_path)
