import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import nilearn
from nilearn import plotting, image
import nibabel as nib

# 设置路径
mask_path = "region_masks_91_109_91/frontal_lobe_mask.nii"
data_dir = os.path.join("datasets", "Image", "train")
label_path = "datasets/Table/train.xlsx"

# 加载标签数据
df = pd.read_excel(label_path)

# 分组数据
group1_ids = df[df['label'] == 0]['patient_id'].astype(str).tolist()  # 正常组
group2_ids = df[df['label'] == 1]['patient_id'].astype(str).tolist()  # 自闭症组

# 构建图像路径
group1_imgs = [os.path.join(data_dir, f"{id}.nii") for id in group1_ids]
group2_imgs = [os.path.join(data_dir, f"{id}.nii") for id in group2_ids]

# 加载ROI掩模
mask_img = nilearn.image.load_img(mask_path)

# 提取ROI信号
def extract_roi_signals(func_imgs, mask_img):
    from nilearn.input_data import NiftiMasker
    # 设置target_affine为None以使用原始分辨率
    masker = NiftiMasker(mask_img=mask_img, target_affine=None)
    return masker.fit_transform(func_imgs)

# 提取每组的ROI信号
print("提取第一组ROI信号...")
group1_signals = extract_roi_signals(group1_imgs, mask_img)
print("提取第二组ROI信号...")
group2_signals = extract_roi_signals(group2_imgs, mask_img)

# 计算信号强度的平均值
group1_mean = np.mean(group1_signals, axis=0)
group2_mean = np.mean(group2_signals, axis=0)

print(f"\n组1平均信号强度: {group1_mean.mean():.4f} ± {group1_mean.std():.4f}")
print(f"组2平均信号强度: {group2_mean.mean():.4f} ± {group2_mean.std():.4f}")

# 进行独立样本t检验
t_stat, p_values = ttest_ind(group1_signals, group2_signals, axis=0)

# 多重比较校正（例如FDR）
from statsmodels.stats.multitest import multipletests
corrected_p = multipletests(p_values, method='fdr_bh')[1]

# 筛选显著差异的脑区
significant_regions = corrected_p < 0.05
print(f"\n显著差异的体素数量: {np.sum(significant_regions)}")
print(f"显著性水平(p < 0.05)下的比例: {(np.sum(significant_regions) / len(significant_regions)) * 100:.2f}%")

# 创建结果目录
results_dir = "roi_analysis_results"
os.makedirs(results_dir, exist_ok=True)

# 保存统计结果
results_df = pd.DataFrame({
    'voxel_id': range(len(p_values)),
    't_statistic': t_stat,
    'p_value': p_values,
    'corrected_p_value': corrected_p,
    'is_significant': significant_regions
})
results_df.to_csv(os.path.join(results_dir, 'statistical_results.csv'), index=False)

# 创建显著性掩码
mask_data = mask_img.get_fdata()
significant_data = np.zeros_like(mask_data)
significant_data[mask_data > 0] = significant_regions.astype(float)

# 保存显著性掩码
significant_nii = nib.Nifti1Image(significant_data, mask_img.affine)
nib.save(significant_nii, os.path.join(results_dir, 'significant_regions.nii'))

# 可视化显著差异
if np.sum(significant_regions) > 0:
    plotting.plot_roi(significant_nii, title="显著差异的ROI区域")
    plt.savefig(os.path.join(results_dir, 'significant_regions.png'))
    plt.close()
else:
    print("\n没有发现显著差异的区域，跳过可视化步骤。")

print(f"\n分析结果已保存到: {results_dir}/")