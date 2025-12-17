import fnmatch
import os

import h5py


def recursive_copy(src_group, dst_group):
    for key in src_group.keys():
        item = src_group[key]
        if isinstance(item, h5py.Dataset):
            src_group.copy(key, dst_group)
        elif isinstance(item, h5py.Group):
            dst_subgroup = dst_group.create_group(key)
            recursive_copy(item, dst_subgroup)


if __name__ == '__main__':
    input_root = '/home/s17gmikh/FCD-Detection/meld_graph/data/input/data4sharing'
    output_dir = os.path.join(input_root, 'meld_combats')
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all subfolders
    for folder_name in os.listdir(input_root):
        # Process only MELD_H101 for now
        if folder_name != 'MELD_H101':
            continue

        folder_path = os.path.join(input_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Search for all combat feature matrix files in the folder
        for file_name in os.listdir(folder_path):
            if fnmatch.fnmatch(file_name, '*featurematrix_combat*.hdf5'):
                input_file = os.path.join(folder_path, file_name)
                print(f"Processing: {input_file}")

                with h5py.File(input_file, 'r') as f:
                    for subject_id in f.keys():
                        subject_group = f[subject_id]

                        for group in subject_group.keys():
                            scanner = subject_group[group]

                            for scan in scanner.keys():
                                persons = scanner[scan]

                                for id_pers in persons.keys():
                                    person_data = persons[id_pers]

                                    out_file = os.path.join(
                                        output_dir,
                                        f"{id_pers}_{scan}_featurematrix_combat.hdf5"
                                    )

                                    with h5py.File(out_file, 'w') as out_f:
                                        recursive_copy(person_data, out_f)

                                    print(f"Saved: {out_file}")
