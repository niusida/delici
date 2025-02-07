import os
import numpy as np
import torch
import nibabel as nib
from scipy import stats
from statsmodels.stats.multitest import multipletests
import logging
from tqdm import tqdm

class VoxelCorrection:
    """体素重要性校正"""
    
    def __init__(self, results_file):
        """初始化体素校正类"""
        self.results_file = results_file
        self.results = None
        self.voxel_data = {}
        self.load_results(results_file)
        self._convert_to_voxel_format()
    
    def load_results(self, results_path):
        """加载分析结果文件"""
        try:
            self.results = torch.load(results_path)
            logging.info("\n加载的结果文件内容:")
            logging.info(f"结果类型: {type(self.results)}")
            logging.info(f"顶层键: {list(self.results.keys())}")
            
            # 检查第一个fold的内容
            first_fold = next(iter(self.results))
            first_fold_data = self.results[first_fold]
            logging.info(f"\n第一个fold ({first_fold}) 的内容:")
            logging.info(f"数据类型: {type(first_fold_data)}")
            logging.info(f"包含的键: {list(first_fold_data.keys())}")
            
            # 如果存在channel_results，检查其结构
            if 'channel_results' in first_fold_data:
                channel_results = first_fold_data['channel_results']
                if channel_results:
                    first_channel = channel_results[0]
                    logging.info("\n第一个通道的数据结构:")
                    logging.info(f"键: {list(first_channel.keys())}")
                    logging.info(f"坐标数量: {len(first_channel.get('original_coordinates', []))}")
                    logging.info(f"激活值数量: {len(first_channel.get('activation_values', []))}")
        
        except Exception as e:
            logging.error(f"加载结果文件时出错: {str(e)}")
            raise
    
    def _convert_to_voxel_format(self):
        """将通道级别的结果转换为体素级别的分析格式"""
        self.voxel_data = {}
        
        for fold_name, fold_data in self.results.items():
            logging.info(f"\n分析 {fold_name} 的数据结构:")
            logging.info(f"fold_data类型: {type(fold_data)}")
            logging.info(f"包含的键: {list(fold_data.keys())}")
            
            fold_voxels = {}
            
            # 检查channel_results是否存在
            channel_results = fold_data.get('channel_results', [])
            if not channel_results:
                logging.warning(f"{fold_name} 没有channel_results")
                continue
            
            # 遍历每个通道的结果
            for channel_info in channel_results:
                channel_idx = channel_info['channel_idx']
                coordinates = channel_info['original_coordinates']
                activation_values = channel_info['activation_values']
                
                # 获取通道的重要性分数
                importance_scores = channel_info['importance_scores']
                disease_specificity = channel_info['disease_specificity']
                
                # 计算通道重要性（考虑疾病特异性）
                channel_importance = max(
                    importance_scores.values(),  # 类别重要性
                    disease_specificity['disease1_vs_normal'],  # 疾病1特异性
                    disease_specificity['disease2_vs_normal']   # 疾病2特异性
                )
                
                # 将每个体素的信息添加到fold_voxels中
                for coord, value in zip(coordinates, activation_values):
                    coord_tuple = tuple(coord)
                    importance = abs(value) * channel_importance  # 结合激活值和通道重要性
                    
                    if coord_tuple not in fold_voxels:
                        fold_voxels[coord_tuple] = {
                            'importance': importance,
                            'value': value,
                            'channels': [channel_idx]
                        }
                    else:
                        # 如果体素已存在，更新其重要性（取最大值）
                        fold_voxels[coord_tuple]['importance'] = max(
                            fold_voxels[coord_tuple]['importance'],
                            importance
                        )
                        fold_voxels[coord_tuple]['channels'].append(channel_idx)
            
            if fold_voxels:
                self.voxel_data[fold_name] = fold_voxels
                logging.info(f"{fold_name} 找到 {len(fold_voxels)} 个体素")
    
    def permutation_test(self, n_permutations=1000, alpha=0.05):
        """使用置换检验评估体素激活的显著性"""
        logging.info("开��置换检验...")
        
        corrected_results = {}
        for fold_name, fold_voxels in self.voxel_data.items():
            logging.info(f"\n处理 {fold_name}...")
            
            if not fold_voxels:
                logging.warning(f"{fold_name} 没有体素数据")
                continue
            
            # 收集所有体素的重要性分数
            voxel_coords = list(fold_voxels.keys())
            importance_scores = np.array([fold_voxels[coords]['importance'] 
                                        for coords in voxel_coords])
            
            # 对每个体素进行置换检验
            p_values = np.zeros(len(voxel_coords))
            
            for i in tqdm(range(len(voxel_coords)), desc=f"分析 {fold_name} 的体素"):
                orig_value = importance_scores[i]
                
                # 生成零分布
                null_distribution = []
                for _ in range(n_permutations):
                    # 随机打乱重要分数
                    perm_scores = np.random.permutation(importance_scores)
                    null_distribution.append(perm_scores[i])
                
                # 计算p值
                p_values[i] = np.mean(np.abs(null_distribution) >= np.abs(orig_value))
            
            # 比较校正
            rejected, p_corrected = multipletests(p_values, 
                                                alpha=alpha,
                                                method='fdr_bh')[:2]
            
            # 找出显著的体素
            sig_indices = np.where(rejected)[0]
            if len(sig_indices) > 0:
                significant_voxels = {}
                for idx in sig_indices:
                    coords = voxel_coords[idx]
                    voxel_info = fold_voxels[coords]
                    
                    significant_voxels[coords] = {
                        'importance': voxel_info['importance'],
                        'value': voxel_info['value'],
                        'channels': voxel_info['channels'],
                        'p_value': p_values[idx],
                        'p_corrected': p_corrected[idx]
                    }
                
                corrected_results[fold_name] = significant_voxels
                logging.info(f"{fold_name}: 找到 {len(sig_indices)} 个显著体素")
            else:
                logging.warning(f"{fold_name}: 没有找到显著的体素")
        
        return corrected_results
    
    def stability_analysis(self, importance_percentile=90):
        """分析重要体素在不同fold间的稳定性"""
        logging.info("开始稳定性分析...")
        
        # 收集所有fold的重要体素
        fold_voxels = {}
        for fold_name, voxels in self.voxel_data.items():
            if not voxels:
                logging.warning(f"{fold_name} 没有体素数据")
                continue
            
            # 使用更高的重要性阈值选择重要体素
            importance_scores = np.array([v['importance'] for v in voxels.values()])
            threshold = np.percentile(importance_scores, importance_percentile)
            
            # 收集重要体素的坐标
            important_voxels = set()
            for coords, voxel_info in voxels.items():
                if voxel_info['importance'] > threshold:
                    important_voxels.add(coords)
            
            if important_voxels:
                fold_voxels[fold_name] = important_voxels
                logging.info(f"{fold_name} 找到 {len(important_voxels)} 个重要体素")
        
        # 如果没有找到任何体素,返回空结果
        if not fold_voxels:
            logging.warning("没有找到任何重要体素")
            return {
                'overlap_matrix': np.array([]),
                'fold_names': [],
                'mean_overlap': 0.0,
                'common_voxels': {}
            }
        
        # 计算重叠系数
        n_folds = len(fold_voxels)
        overlap_matrix = np.zeros((n_folds, n_folds))
        fold_names = list(fold_voxels.keys())
        
        for i in range(n_folds):
            for j in range(i+1, n_folds):
                intersection = len(fold_voxels[fold_names[i]] & 
                                 fold_voxels[fold_names[j]])
                union = len(fold_voxels[fold_names[i]] | 
                           fold_voxels[fold_names[j]])
                if union > 0:
                    overlap = intersection / union
                else:
                    overlap = 0.0
                overlap_matrix[i,j] = overlap_matrix[j,i] = overlap
        
        # 找出在多个fold中都重要的体素
        if len(fold_voxels) > 1:
            # 使用第一个集合作为基准,然后与其他集合求交集
            common_voxels = fold_voxels[fold_names[0]]
            for name in fold_names[1:]:
                common_voxels = common_voxels & fold_voxels[name]
            
            # 收集这些共同体素的详细信息
            common_voxels_info = {}
            for coords in common_voxels:
                # 计算每个体素在不同fold中的平均重要性
                importances = []
                values = []
                all_channels = set()
                
                for fold_name in fold_names:
                    voxel_info = self.voxel_data[fold_name][coords]
                    importances.append(voxel_info['importance'])
                    values.append(voxel_info['value'])
                    all_channels.update(voxel_info['channels'])
                
                common_voxels_info[coords] = {
                    'mean_importance': np.mean(importances),
                    'std_importance': np.std(importances),
                    'mean_value': np.mean(values),
                    'channels': list(all_channels)
                }
            
            logging.info(f"在所有fold中都重要的体素数量: {len(common_voxels_info)}")
        else:
            common_voxels_info = {}
            logging.warning("没有足够的fold来计算共同体素")
        
        # 计算平均重叠度
        if n_folds > 1:
            mean_overlap = np.mean(overlap_matrix[np.triu_indices(n_folds, k=1)])
        else:
            mean_overlap = 0.0
        
        return {
            'overlap_matrix': overlap_matrix,
            'fold_names': fold_names,
            'mean_overlap': mean_overlap,
            'common_voxels': common_voxels_info
        }
    
    def bootstrap_analysis(self, n_bootstrap=1000, confidence_level=0.95, min_importance=0.5):
        """使用bootstrap方法评估体素的可靠性"""
        logging.info("开始Bootstrap分析...")
        
        bootstrap_results = {}
        for fold_name, voxels in self.voxel_data.items():
            logging.info(f"处理 {fold_name}...")
            
            if not voxels:
                logging.warning(f"{fold_name} 没有体素数据")
                continue
            
            # 准备数据
            coords_list = list(voxels.keys())
            importance_scores = np.array([voxels[coords]['importance'] 
                                        for coords in coords_list])
            
            # 只分析重要性超过阈值的体素
            significant_indices = np.where(importance_scores > min_importance)[0]
            if len(significant_indices) == 0:
                logging.warning(f"{fold_name}: 没有找到重要性超过{min_importance}的体素")
                continue
            
            # 对每个重要体素进行bootstrap分析
            bootstrap_values = np.zeros((len(significant_indices), n_bootstrap))
            
            for i in range(n_bootstrap):
                # 生成bootstrap样本
                indices = np.random.choice(len(significant_indices), 
                                         size=len(significant_indices), 
                                         replace=True)
                bootstrap_values[:, i] = importance_scores[significant_indices][indices]
            
            # 计算置信区间
            lower = np.percentile(bootstrap_values, 
                                ((1 - confidence_level) / 2) * 100, 
                                axis=1)
            upper = np.percentile(bootstrap_values, 
                                (1 - (1 - confidence_level) / 2) * 100, 
                                axis=1)
            
            # 找出稳定的体素（置信区间不包含0且下限大于min_importance）
            stable_indices = np.where(lower > min_importance)[0]
            
            # 收集稳定体素的信息
            stable_voxels = {}
            for idx in stable_indices:
                coords = coords_list[significant_indices[idx]]
                stable_voxels[coords] = {
                    'importance': voxels[coords]['importance'],
                    'value': voxels[coords]['value'],
                    'channels': voxels[coords]['channels'],
                    'ci_lower': lower[idx],
                    'ci_upper': upper[idx]
                }
            
            bootstrap_results[fold_name] = stable_voxels
            logging.info(f"{fold_name} 找到 {len(stable_voxels)} 个稳定体素")
        
        return bootstrap_results
    
    def correct_voxels(self, n_permutations=1000, significance_level=0.05):
        """执行体素校正分析"""
        logging.info("开始体素校正分析...")
        
        # 执行置换检验
        logging.info("\n执行置换检验...")
        permutation_results = self.permutation_test(n_permutations, significance_level)
        
        # 执行稳定性分析
        logging.info("\n执行稳定性分析...")
        stability_results = self.stability_analysis()
        
        # 执行bootstrap分析
        logging.info("\n执行bootstrap分析...")
        bootstrap_results = self.bootstrap_analysis()
        
        # 整合所有结果
        self.corrected_results = {
            'permutation_results': permutation_results,
            'stability_results': stability_results,
            'bootstrap_results': bootstrap_results
        }
        
        # 打印汇总信息
        logging.info("\n=== 分析结果汇总 ===")
        for fold_name in self.voxel_data.keys():
            logging.info(f"\n{fold_name}:")
            if fold_name in permutation_results:
                logging.info(f"- 显著体素数量: {len(permutation_results[fold_name])}")
            if stability_results['common_voxels']:
                logging.info(f"- 稳定体素数量: {len(stability_results['common_voxels'])}")
            if fold_name in bootstrap_results:
                logging.info(f"- Bootstrap稳定体素数量: {len(bootstrap_results[fold_name])}")
        
        return self.corrected_results
    
    def save_results(self, output_file):
        """保存校正后的结果"""
        if hasattr(self, 'corrected_results'):
            torch.save(self.corrected_results, output_file)
            logging.info(f"结果已保存到: {output_file}")
        else:
            logging.warning("没有可保存的校正结果")

def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # 设置参数
    results_file = os.path.join('analysis_results', 'analysis_results.pth')
    output_dir = 'correction_results'
    n_permutations = 1000
    significance_level = 0.05
    
    # 检查文件是否存在
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"找不到结果文件: {results_file}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'corrected_results.pth')
    
    # 创建VoxelCorrection实例
    corrector = VoxelCorrection(results_file)
    
    # 加载并分析结果
    logging.info("\n=== 开始加载和分析结果 ===")
    corrector.load_results(results_file)
    
    # 执行校正
    logging.info("\n=== 开始执行校正 ===")
    corrector.correct_voxels(n_permutations, significance_level)
    
    # 保存结果
    corrector.save_results(output_file)
    logging.info(f"\n校正后的结果已保存到: {output_file}")

if __name__ == '__main__':
    main() 