import matplotlib.pyplot as plt
import json
import re
import os
import argparse
import pandas as pd
import seaborn as sns

def parse_log(log_file):
    train_data = []
    val_metrics = {}
    
    print(f"正在读取日志文件: {log_file}...")
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    # 1. 提取训练数据 (JSON行)
    for line in lines:
        line = line.strip()
        if line.startswith('{"iteration"'):
            try:
                entry = json.loads(line)
                train_data.append(entry)
            except json.JSONDecodeError:
                continue

    # 2. 提取验证数据 (Eval行)
    # 逻辑：寻找日志中最后一次出现的 Eval 统计信息
    # 优先找 "Averaged eval stats"，如果没跑完（报错了），就找最后一行 "Eval [...]"
    last_eval_line = None
    for line in lines:
        # 匹配类似于 "Eval [ 10/250] ... bit_acc_none: 0.98 ..." 的行
        if "Eval" in line and "bit_acc_none" in line:
            last_eval_line = line
        # 如果有最终平均值，那是最好的
        if "Averaged eval stats" in line: 
            last_eval_line = line
            
    if last_eval_line:
        print("找到验证集数据，正在解析...")
        # 使用正则提取所有 "key: value" 格式的数据
        # 匹配模式： 字母下划线组合: 数字.数字
        matches = re.findall(r'([a-zA-Z0-9_]+):\s*([0-9.]+)', last_eval_line)
        for key, value in matches:
            try:
                val_metrics[key] = float(value)
            except ValueError:
                pass
    else:
        print("警告：日志中未找到 Eval 相关数据（可能验证未启动或格式不匹配）。")
    
    return pd.DataFrame(train_data), val_metrics

def plot_metrics(df, val_metrics, output_dir):
    # 设置学术风格
    sns.set(style="whitegrid", context="talk")
    plt.rcParams['font.family'] = 'DejaVu Sans' # 防止字体报错
    
    # ==================== Part 1: 训练过程曲线 (Line Charts) ====================
    if not df.empty:
        # 1. Loss 曲线
        plt.figure(figsize=(10, 6))
        plt.plot(df['iteration'], df['loss'], label='Total Loss', color='#E24A33', linewidth=2)
        if 'loss_w' in df.columns:
            plt.plot(df['iteration'], df['loss_w'], label='Watermark Loss', color='#348ABD', linestyle='--', alpha=0.7)
        if 'loss_i' in df.columns:
            plt.plot(df['iteration'], df['loss_i'], label='Image Loss', color='#988ED5', linestyle=':', alpha=0.7)
        
        plt.title('Training Loss Convergence', fontsize=16, fontweight='bold')
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'train_loss_curve.png'), dpi=300)
        plt.close()

        # 2. Accuracy 曲线
        plt.figure(figsize=(10, 6))
        if 'bit_acc_avg' in df.columns:
            plt.plot(df['iteration'], df['bit_acc_avg'], label='Train Bit Acc', color='#2ecc71', linewidth=2.5)
        if 'word_acc_avg' in df.columns:
            plt.plot(df['iteration'], df['word_acc_avg'], label='Train Word Acc', color='#27ae60', linestyle='--', linewidth=2)
        
        plt.axhline(y=0.5, color='gray', linestyle=':', label='Random Guess')
        plt.title('Training Accuracy (Bit & Word)', fontsize=16, fontweight='bold')
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.ylim(0, 1.05)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'train_accuracy_curve.png'), dpi=300)
        plt.close()

    # ==================== Part 2: 验证集鲁棒性分析 (Bar Chart) ====================
    if val_metrics:
        # 筛选出所有攻击类型的准确率数据
        attacks = []
        bit_scores = []
        word_scores = []
        
        for key, value in val_metrics.items():
            if key.startswith('bit_acc_'):
                # 提取攻击名称 (如 'crop_01', 'jpeg_80')
                attack_name = key.replace('bit_acc_', '')
                attacks.append(attack_name)
                bit_scores.append(value)
                # 同时找对应的 word accuracy
                word_key = f'word_acc_{attack_name}'
                word_scores.append(val_metrics.get(word_key, 0.0))
        
        if attacks:
            # 构造用于 Seaborn 绘图的数据框
            val_df = pd.DataFrame({
                'Attack Type': attacks,
                'Bit Accuracy': bit_scores,
                'Word Accuracy': word_scores
            })
            
            # 转换为长格式以便分组绘图
            val_melted = val_df.melt(id_vars='Attack Type', var_name='Metric', value_name='Score')
            
            plt.figure(figsize=(14, 7))
            # 绘制分组柱状图
            ax = sns.barplot(data=val_melted, x='Attack Type', y='Score', hue='Metric', palette='viridis')
            
            # 在柱子上方标注数值
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)
            
            plt.title('Robustness Analysis under Various Attacks (Validation)', fontsize=18, fontweight='bold')
            plt.xlabel('Attack Types', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.ylim(0, 1.15) # 留出顶部空间写数字
            plt.xticks(rotation=45) # 旋转标签防止重叠
            plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='Perfect Score')
            plt.legend(loc='upper right')
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, 'val_robustness_analysis.png')
            plt.savefig(save_path, dpi=300)
            print(f"已保存验证集分析图: {save_path}")
            plt.close()
            
            # 额外：打印 PSNR 结果
            if 'psnr' in val_metrics:
                print(f"\n>>> 最终验证集平均 PSNR: {val_metrics['psnr']:.2f} dB")

def main():
    parser = argparse.ArgumentParser(description="Plot training and validation results")
    parser.add_argument('--log_file', type=str, default='train.log', help='Path to log file')
    parser.add_argument('--output_dir', type=str, default='plots', help='Output directory')
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"错误：文件 {args.log_file} 不存在！")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 解析
    df, val_metrics = parse_log(args.log_file)

    # 绘图
    if df.empty and not val_metrics:
        print("未提取到有效数据。请检查日志文件格式。")
    else:
        plot_metrics(df, val_metrics, args.output_dir)
        print(f"\n全部图表已生成至 '{args.output_dir}' 文件夹！")

if __name__ == "__main__":
    main()