import os
import csv
from collections import defaultdict

def load_region_statistics(save_dir, layer_name='layer3', top_k=10, min_percentage=5.0):
    """加载指定层级和通道数的区域统计数据"""
    region_counts = defaultdict(int)
    # 查找所有以layer3_channel开头的文件夹
    channel_dirs = [
        os.path.join(save_dir, d)
        for d in os.listdir(save_dir)
        if d.startswith(f'{layer_name}_channel')
    ][:top_k]
    
    for channel_dir in channel_dirs:
        csv_path = os.path.join(channel_dir, 'region_statistics.csv')
        if not os.path.exists(csv_path):
            print(f"警告: 找不到文件 {csv_path}")
            continue
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                region_name = row['脑区名称']
                percentage = float(row['占比(%)'])
                if percentage >= min_percentage:
                    region_counts[region_name] += 1
    
    return region_counts

def analyze_repetitions(region_counts, threshold=2):
    """分析反复出现的脑区"""
    repeated_regions = {
        region: count for region, count in region_counts.items() if count >= threshold
    }
    return repeated_regions

def save_repeated_regions(repeated_regions, save_path):
    """保存反复出现的脑区到CSV文件"""
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['脑区名称', '出现次数'])
        for region, count in sorted(repeated_regions.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([region, count])
    print(f"反复出现的脑区已保存到: {save_path}")

def main():
    # 设置保存目录
    save_dir = "E:/py test/austim/nifti_results"
    layer_name = 'layer3'
    top_k = 15  # 前10个重要通道
    threshold = 2  # 最少出现次数
    min_percentage = 10.0  # 最小占比
    
    # 加载区域统计
    region_counts = load_region_statistics(save_dir, layer_name, top_k, min_percentage)
    
    if not region_counts:
        print("没有找到任何区域统计数据。")
        return
    
    # 分析反复出现的脑区
    repeated_regions = analyze_repetitions(region_counts, threshold)
    
    if not repeated_regions:
        print(f"没有脑区在至少{threshold}个通道中出现。")
        return
    
    # 保存结果
    output_csv = os.path.join(save_dir, f'{layer_name}_repeated_regions.csv')
    save_repeated_regions(repeated_regions, output_csv)

if __name__ == "__main__":
    main()