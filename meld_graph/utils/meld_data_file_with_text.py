import os
import csv
import glob
import pandas as pd
import sys

if __name__ == "__main__":
    # Путь к вашему существующему CSV с результатами
    # CHANGE BACK TO FULL #######################################################
    reports_csv = "/home/s17gmikh/FCD-Detection/meld_graph/data/preprocessed/meld_files/all_augmented_reports.csv" #H101_reports.csv"
    # Корневая папка, где лежат ваши HDF5/NIfTI-файлы
    comb_root   = "/home/s17gmikh/FCD-Detection/meld_graph/data/input/data4sharing/meld_combats"
    # CHANGE BACK TO FULL #######################################################
    out_dir     = "/home/s17gmikh/FCD-Detection/meld_graph/data/preprocessed/MELD_BONN_dataset_augmented_final.csv" # H101_reports_full.csv"
    # Читаем отчёты
    reports_df = pd.read_csv(reports_csv, dtype=str)

    # Открываем новый CSV на запись
    with open(out_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['DATA_PATH', 'ROI_PATH', 'harvard_oxford', 'aal'])

        for _, row in reports_df.iterrows():
            sid   = row['subject_id']
            harv  = row['report_harvard_oxford']
            aal   = row['report_aal']

            # 1) Ищем HDF5-файл по шаблону "{sid}_*featurematrix*.hdf5"
            pattern_h5 = os.path.join(comb_root, f"{sid}_*featurematrix*.hdf5")
            h5_list = glob.glob(pattern_h5)
            if not h5_list:
                print(f"❌ HDF5 не найден для {sid}")
                # print(pattern_h5)
                # data_path = pattern_h5.split("*")[0] + 'control_featurematrix_combat.hdf5'
                
                # writer.writerow([data_path, '', harv, aal])
                continue
            
            data_path = h5_list[0]
            if "control" in pattern_h5:
                roi_path = ''
            else:
                roi_path = data_path

            # 3) Записываем строку в выходной CSV
            writer.writerow([data_path, roi_path, harv, aal])