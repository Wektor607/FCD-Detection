import os
import csv
import glob
import pandas as pd

if __name__ == "__main__":
    # Path to the existing CSV file with results
    for name in ["MELD_splits.csv", "BONN_splits.csv"]:
        reports_csv = f"/home/s17gmikh/FCD-Detection/meld_graph/data/preprocessed/{name}"
        
        # Root directory where the HDF5/NIfTI files are stored
        comb_root   = "/home/s17gmikh/FCD-Detection/meld_graph/data/input/data4sharing/meld_combats"
        
        out_dir     = "/home/s17gmikh/FCD-Detection/meld_graph/data/preprocessed/Dataset_without_text.csv"  # H101_reports_full.csv"
        
        # Read the reports CSV
        reports_df = pd.read_csv(reports_csv, dtype=str)

        # Open the output CSV file in append mode
        with open(out_dir, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['DATA_PATH', 'ROI_PATH'])  # , 'harvard_oxford', 'aal'])

            for _, row in reports_df.iterrows():
                sid   = row['subject_id']
                # harv  = row['report_harvard_oxford']
                # aal   = row['report_aal']

                # 1) Search for the HDF5 file using the pattern "{sid}_*featurematrix*.hdf5"
                pattern_h5 = os.path.join(comb_root, f"{sid}_*featurematrix*.hdf5")
                h5_list = glob.glob(pattern_h5)
                if not h5_list:
                    print(f"❌ HDF5 not found for {sid}")
                    continue

                data_path = h5_list[0]
                if "control" in pattern_h5:
                    roi_path = ''
                else:
                    roi_path = h5_list[0]

                # 3) Write the row to the output CSV
                writer.writerow([data_path, roi_path])  # , harv, aal])
