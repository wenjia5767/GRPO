import pandas as pd
import json

def analyze_log_file(input_path='/home/zhangwj/GRPO/grpo_run_lr_5e-05/train_log.json', output_path='/home/zhangwj/GRPO/grpo_run_lr_5e-05/results.txt'):
    """
    读取 JSON 日志文件，找到最高的评估准确率和格式正确率，并将结果保存到文件中。

    Args:
        input_path (str): 输入的 JSON 文件路径。
        output_path (str): 输出结果的文件路径。
    """
    try:
        # 使用 pandas 直接从 JSON 文件读取数据到 DataFrame
        # 这对于处理结构化的日志数据非常高效
        df = pd.read_json(input_path)

        # 确保所需的列存在
        if 'eval_accuracy' not in df.columns or 'eval_format_ok_rate' not in df.columns:
            print(f"错误: 输入文件 '{input_path}' 中缺少 'eval_accuracy' 或 'eval_format_ok_rate' 列。")
            return

        # 使用 .max() 函数轻松找到每列的最大值
        max_accuracy = df['eval_accuracy'].max()
        max_format_ok_rate = df['eval_format_ok_rate'].max()

        # 找到最大值对应的具体是哪一步 (step)
        # .idxmax() 会返回最大值第一次出现的行的索引
        step_for_max_accuracy = df.loc[df['eval_accuracy'].idxmax()]['step']
        step_for_max_format_ok = df.loc[df['eval_format_ok_rate'].idxmax()]['step']


        # 将结果写入指定的输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("训练日志分析结果:\n")
            f.write("="*30 + "\n")
            f.write(f"最高 eval_accuracy: {max_accuracy:.6f} (在 step {step_for_max_accuracy})\n")
            f.write(f"最高 eval_format_ok_rate: {max_format_ok_rate:.6f} (在 step {step_for_max_format_ok})\n")

        print(f"分析完成！结果已成功保存到 '{output_path}' 文件中。")

    except FileNotFoundError:
        print(f"错误: 未找到输入文件 '{input_path}'。请确保文件存在于正确的路径。")
    except Exception as e:
        print(f"处理文件时发生未知错误: {e}")

if __name__ == "__main__":
    # 调用函数，使用默认的文件名
    analyze_log_file()