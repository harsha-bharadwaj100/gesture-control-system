import shutil
import os

os.makedirs('paper_elsarticle/figures', exist_ok=True)

media_dir = 'extracted_docx_images/word/media'
fig_mapping = {
    'image1.jpeg': 'paper_elsarticle/figures/fig1_cumulative_pipeline.jpg',
    'image2.jpeg': 'paper_elsarticle/figures/fig2_independent_pipeline.jpg',
    'image3.jpeg': 'paper_elsarticle/figures/fig3_correlation_heatmap.jpg',
    'image4.jpeg': 'paper_elsarticle/figures/fig4_feature_importance.jpg',
    'image5.png':  'paper_elsarticle/figures/fig5_baseline_vs_proposed.png',
    'image6.jpeg': 'paper_elsarticle/figures/fig6_pareto_frontier.jpg',
    'image7.png':  'paper_elsarticle/figures/fig7_classwise_f1.png',
    'image8.jpeg': 'paper_elsarticle/figures/fig8_confusion_matrix_grid.jpg'
}

for src_name, dst_path in fig_mapping.items():
    src_path = os.path.join(media_dir, src_name)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Copied {src_path} -> {dst_path}")
    else:
        print(f"ERROR: {src_path} not found!")

# Copy 9th figure (anonymized S01-S15 plot)
subj_plot_src = 'paper_images/Figure_Subject_Accuracy_SEM.png'
subj_plot_dst = 'paper_elsarticle/figures/fig9_subject_accuracy_sem.png'
if os.path.exists(subj_plot_src):
    shutil.copy(subj_plot_src, subj_plot_dst)
    print(f"Copied {subj_plot_src} -> {subj_plot_dst}")
