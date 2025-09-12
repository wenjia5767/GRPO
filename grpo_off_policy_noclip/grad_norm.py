import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_grad_norm_comparison(file1, file2, label1='Experiment 1', label2='Experiment 2',
                               output_filename='grad_norm_comparison.png', y_scale='linear'):
    """
    从两个JSON文件中读取训练结果，并绘制梯度范数（grad_norm）
    与训练步数（training step）的对比图。

    参数:
        file1 (str): 第一个JSON文件的路径。
        file2 (str): 第二个JSON文件的路径。
        label1 (str): 第一个实验在图中的标签。
        label2 (str): 第二个实验在图中的标签。
        output_filename (str): 输出图片的文件名。
        y_scale (str): Y轴的缩放方式，可以是 'linear'（线性）或 'log'（对数）。
    """
    # 检查文件是否存在
    if not os.path.exists(file1):
        print(f"错误: 文件 '{file1}' 不存在。")
        return
    if not os.path.exists(file2):
        print(f"错误: 文件 '{file2}' 不存在。")
        return

    # 从文件中读取和解析JSON数据
    try:
        with open(file1, 'r', encoding='utf-8') as f: 
            data1 = json.load(f)
        with open(file2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
    except json.JSONDecodeError as e:
        print(f"解析JSON文件时出错: {e}")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # 从数据中提取 'step' 和 'grad_norm'
    steps1 = [item['step'] for item in data1]
    grad_norms1 = [item['grad_norm'] for item in data1]
    
    steps2 = [item['step'] for item in data2]
    grad_norms2 = [item['grad_norm'] for item in data2]

    # 对于对数坐标轴，需要处理非正值
    if y_scale == 'log':
        # 过滤掉非正值（<=0），因为log(0)或负数无意义
        valid_data1 = [(s, g) for s, g in zip(steps1, grad_norms1) if g > 0]
        valid_data2 = [(s, g) for s, g in zip(steps2, grad_norms2) if g > 0]
        
        if not valid_data1 or not valid_data2:
            print("警告: 无法绘制对数坐标图，因为数据中没有大于0的 grad_norm 值。")
            return
        
        steps1, grad_norms1 = zip(*valid_data1)
        steps2, grad_norms2 = zip(*valid_data2)


    # 创建图形
    plt.figure(figsize=(12, 7))

    # 绘制第一个文件的数据
    plt.plot(steps1, grad_norms1, marker='o', linestyle='-', label=label1)

    # 绘制第二个文件的数据
    plt.plot(steps2, grad_norms2, marker='x', linestyle='--', label=label2)

    # 【关键修改】设置Y轴的缩放类型，使用对数刻度
    plt.yscale(y_scale)

    # 添加标题和标签
    title = f'Gradient Norm vs. Training Step Comparison ({y_scale.capitalize()} Scale)'
    plt.title(title, fontsize=16)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Gradient Norm', fontsize=12)

    # 添加图例
    plt.legend(fontsize=10)

    # 添加网格
    plt.grid(True)
    
    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_filename)
    print(f"图片已保存为 {output_filename}")

# --- 主程序 ---
if __name__ == '__main__':
    # 请确保文件路径正确
    file_experiment_1 = '/home/zhangwj/GRPO/grpo_off_policy/train_log.json'
    file_experiment_2 = '/home/zhangwj/GRPO/grpo_off_policy_noclip/train_log.json'

    label1 = 'Off policy with clip'
    label2 = 'Off policy no clip'

    # 2. 绘制对数刻度图
    plot_grad_norm_comparison(file_experiment_1, file_experiment_2,
                               label1=label1, label2=label2,
                               output_filename='grad_norm_comparison_log.png', y_scale='log')