import matplotlib.pyplot as plt
from nilearn import plotting

if __name__ == '__main__':
    wanted = {
        'sub-00136', 'sub-00138', 'sub-00139', 'sub-00140',
        'sub-00141', 'sub-00142', 'sub-00144', 'sub-00145',
        'sub-00146'
    }

    for subj in wanted:

        pred_nii = f'/home/s17gmikh/FCD-Detection/meld_graph/data/output/predictions_reports/{subj}/predictions/prediction_{subj}.nii.gz'
        meld_pred_nii = f'/home/s17gmikh/FCD-Detection/meld_graph/data/output/predictions_reports/{subj}/predictions/prediction.nii.gz'
        if subj == 'sub-00140':
            roi    = f'/home/s17gmikh/FCD-Detection/meld_graph/data/input/{subj}/anat/{subj}_acq-tse3dvfl_FLAIR_roi.nii.gz'
            bg_nii = f'/home/s17gmikh/FCD-Detection/meld_graph/data/input/{subj}/anat/{subj}_acq-tse3dvfl_FLAIR.nii.gz'
        else:
            roi    = f'/home/s17gmikh/FCD-Detection/meld_graph/data/input/ds004199/{subj}/anat/{subj}_acq-T2sel_FLAIR_roi.nii.gz'
            bg_nii = f'/home/s17gmikh/FCD-Detection/meld_graph/data/input/ds004199/{subj}/anat/{subj}_acq-T2sel_FLAIR.nii.gz'

        from matplotlib import gridspec

        # 1) Создаём фигуру и GridSpec-сетку
        fig = plt.figure(figsize=(14, 10))
        gs  = gridspec.GridSpec(
            2, 2,
            figure=fig,
            wspace=0.02,  # горизонтальные отступы
            hspace=0.02   # вертикальные отступы
        )

        # 2) Рисуем каждую ячейку
        ax0 = fig.add_subplot(gs[0, 0])
        plotting.plot_roi(
            roi, bg_img=bg_nii, axes=ax0,
            display_mode='ortho',
            title='ROI', cmap='autumn', annotate=False
        )

        ax1 = fig.add_subplot(gs[0, 1])
        plotting.plot_roi(
            pred_nii, bg_img=bg_nii, axes=ax1,
            display_mode='ortho',
            title='My prediction', cmap='autumn', annotate=False
        )

        ax2 = fig.add_subplot(gs[1, 0])
        plotting.plot_roi(
            roi, bg_img=bg_nii, axes=ax2,
            display_mode='ortho',
            title='ROI', cmap='autumn', annotate=False
        )

        ax3 = fig.add_subplot(gs[1, 1])
        plotting.plot_roi(
            meld_pred_nii, bg_img=bg_nii, axes=ax3,
            display_mode='ortho',
            title='MELD prediction', cmap='autumn', annotate=False
        )

        # 3) Пустая ячейка
        ax3 = fig.add_subplot(gs[1,1])
        ax3.axis('off')

        plt.savefig(f'./data/output/predictions_reports/comparisons/{subj}.png')