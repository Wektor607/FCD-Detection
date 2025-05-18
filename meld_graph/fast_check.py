from atlasreader import create_output
from nilearn import datasets, image
import nibabel as nib
from nilearn import plotting
import numpy as np
# Загружаем Harvard-Oxford atlas
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm', symmetric_split=True)

# atlas['maps'] уже является путём к .nii.gz, это строка
atlas_img = atlas['maps']

# pred_img = nib.load("/home/s17gmikh/FCD-Detection/meld_graph/data/output/predictions_reports/sub-00146/sub-00146_prediction_to_mni.nii.gz")
# roi_img = nib.load("/home/s17gmikh/FCD-Detection/data/ds004199/sub-00146/anat/preprocessed/sub-00146_roi_to_mni.nii.gz")

pred_img = nib.load('/home/s17gmikh/FCD-Detection/meld_graph/dataset/sub-00001/pred_in_atlas.nii.gz')
roi_img  = nib.load('/home/s17gmikh/FCD-Detection/meld_graph/dataset/sub-00001/roi_in_atlas.nii.gz')

pred_data = pred_img.get_fdata()
roi_data = roi_img.get_fdata()

# 3. Приводим маску в формат атласа
roi_to_atlas  = image.resample_to_img(roi_img, atlas_img, interpolation="nearest")
print(roi_to_atlas.shape)
pred_to_atlas = image.resample_to_img(pred_img, atlas_img, interpolation="nearest")
print(pred_to_atlas.shape)
# 4. Генерируем отчёт (можно указать outdir для сохранения)
create_output(roi_to_atlas, cluster_extent=5, direction="both")

plotting.plot_stat_map(pred_to_atlas, title="Prediction Original", threshold=0.2)
plotting.plot_stat_map(roi_to_atlas, title="ROI Original", threshold=0.2)