
import os
import sys
import argparse
import subprocess
from subprocess import Popen
from meld_graph.tools_pipeline import  get_m
import glob
import nibabel as nib
from meld_graph.paths import MELD_DATA_PATH
import numpy as np
def project_lesion_to_surface(subject_id, subjects_dir):
    
    env = os.environ.copy()
    env["SUBJECTS_DIR"] = subjects_dir
    
    subject_anat_dir = os.path.join(MELD_DATA_PATH, "input", "ds004199", subject_id, "anat")
    roi_candidates = glob.glob(os.path.join(subject_anat_dir, "*roi*.nii*"))
    if not roi_candidates:
        print(get_m(f"No ROI file found for {subject_id}", subject_id, "ERROR"))
        return False
    roi_path = roi_candidates[0]

    flair_reg = os.path.join(subjects_dir, subject_id, "mri", "transforms", "FLAIRraw.auto.dat")
    t1_mgz = os.path.join(subjects_dir, subject_id, "mri", "T1.mgz")
    resampled_roi_path = os.path.join(subject_anat_dir, "roi_in_T1_space.nii.gz")

    # 1. Convert ROI: FLAIR -> T1
    cmd_vol2vol = (
        f"mri_vol2vol --mov {roi_path} "
        f"--targ {t1_mgz} "
        f"--reg {flair_reg} "
        f"--o {resampled_roi_path} "
        f"--interp nearest"
    )
    
    subprocess.run(cmd_vol2vol, shell=True, check=True, env=env)

    for hemi in ["lh", "rh"]:
        out_path = os.path.join(subjects_dir, subject_id, "surf_meld", f"{hemi}.lesion_linked.mgh")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        cmd_surf = (
            f"mri_vol2surf --mov {resampled_roi_path} "
            f"--regheader {subject_id} "
            f"--hemi {hemi} "
            f"--interp nearest "
            f"--o {out_path}"
        )

        subprocess.run(cmd_surf, shell=True, check=True, env=env)

    return True

def non_zero(path):
    if not os.path.isfile(path):
        return False
    data = nib.load(path).get_fdata()
    return np.count_nonzero(data) > 0

def lesion_labels(subject_id, subjects_dir, verbose=False):

    for hemi in ["lh", "rh"]:
        hemi_path = f"{subjects_dir}/{subject_id}/surf_meld/{hemi}.lesion_linked.mgh"
        if os.path.isfile(hemi_path):
            if not os.path.isfile(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/lh.on_lh.lesion.mgh"):
                command = f"SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subjects_dir}/{subject_id}/surf_meld/lh.lesion_linked.mgh --trg {subjects_dir}/{subject_id}/xhemi/surf_meld/lh.on_lh.lesion.mgh --streg {subjects_dir}/{subject_id}/surf/lh.sphere.reg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg"
                proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
                stdout, stderr= proc.communicate()
                if verbose:
                    print(stdout)
                if proc.returncode!=0:
                    print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
                    return False

        elif os.path.isfile(hemi_path):
            if not os.path.isfile(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/rh.on_lh.lesion.mgh"):
                command = f"SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subjects_dir}/{subject_id}/surf_meld/rh.lesion_linked.mgh --trg {subjects_dir}/{subject_id}/xhemi/surf_meld/rh.on_lh.lesion.mgh --streg {subjects_dir}/{subject_id}/xhemi/surf/lh.fsaverage_sym.sphere.reg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg"
                proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
                stdout, stderr= proc.communicate()
                if verbose:
                    print(stdout)
                if proc.returncode!=0:
                    print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
                    return False


if __name__ == "__main__":
    #parse commandline arguments pointing to subject_dir etc
    parser = argparse.ArgumentParser(description='create lesion labels')
    parser.add_argument('subject_id', type=str,
                        help='subject_id')
    parser.add_argument('subjects_dir', type=str,
                        help='freesurfer subject directory ')
    args = parser.parse_args()
    #save subjects dir and subject ids. import the text file containing subject ids
    subject_id=args.subject_id
    subjects_dir=args.subject_dir
    lesion_labels(subject_id, subjects_dir)