import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_log(file_path='/home/zhangwj/GRPO/grpo_off_policy_entropy_clipfraction/train_log.json'):
    """
    从训练日志文件中加载数据，并绘制 policy_entropy 和 clipped_fraction 的变化曲线。

    参数:
    file_path (str): 训练日志JSON文件的路径。
    """
    try:
        # 使用 with 语句确保文件正确关闭
        with open(file_path, 'r') as f:
            # 加载JSON数据
            data = json.load(f)

        # 将JSON数据转换为pandas DataFrame，这是一种强大的数据处理工具
        df = pd.DataFrame(data)

        # 检查所需的数据列是否存在
        required_columns = ['step', 'policy_entropy', 'clipped_fraction']
        if not all(col in df.columns for col in required_columns):
            print(f"错误：JSON文件中缺少必要的列。请确保包含 {required_columns}")
            return

        # --- 开始绘图 ---
        # 创建一个图形和一组子图
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # 设置图表标题和x轴标签
        plt.title('Training Process Analysis', fontsize=16)
        ax1.set_xlabel('Step', fontsize=12)

        # 绘制 policy_entropy 曲线
        color = 'tab:blue'
        ax1.set_ylabel('Policy Entropy', color=color, fontsize=12)
        ax1.plot(df['step'], df['policy_entropy'], color=color, label='Policy Entropy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # 创建一个共享x轴的第二个y轴 (ax2)，用于绘制 clipped_fraction
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Clipped Fraction', color=color, fontsize=12)
        ax2.plot(df['step'], df['clipped_fraction'], color=color, label='Clipped Fraction')
        ax2.tick_params(axis='y', labelcolor=color)

        # --- 优化图表外观 ---
        # 让布局更紧凑，防止标签重叠
        fig.tight_layout()
        
        # 显示图例
        # 获取两个轴的图例句柄和标签，并合并它们
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        # 保存图表到文件
        output_filename = 'training_metrics.png'
        plt.savefig(output_filename, dpi=300)
        print(f"图表已保存为 {output_filename}")
        
        # 显示图表
        plt.show()

    except FileNotFoundError:
        print(f"错误: 未找到文件 '{file_path}'。请确保文件路径正确。")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

# --- 运行主函数 ---
if __name__ == '__main__':
    plot_training_log()