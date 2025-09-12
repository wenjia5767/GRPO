import pandas as pd
import json

def analyze_log_file(input_path='/home/zhangwj/GRPO/sft_gsm8k_lr5e-06/n=1024_train_log.jsonl', output_path='/home/zhangwj/GRPO/sft_gsm8k_lr5e-06/results_1024.txt'):
    """
    读取 JSON Lines 格式的日志文件，筛选出评估日志，
    找到最高的 accuracy 和 format_accuracy，并将结果保存到文件中。

    Args:
        input_path (str): 输入的 JSON 文件路径。
        output_path (str): 输出结果的文件路径。
    """
    try:
        # 使用 pandas 读取 JSON Lines 文件 (lines=True 是关键)
        # 这会把每一行的 JSON 对象当作一条记录
        df = pd.read_json(input_path, lines=True)

        # 筛选出 type 为 'eval' 的行，因为只有这些行包含准确率数据
        eval_df = df[df['type'] == 'eval'].copy() # 使用 .copy() 避免 SettingWithCopyWarning

        # 检查是否有 'eval' 类型的日志
        if eval_df.empty:
            print(f"文件中没有找到类型为 'eval' 的日志记录。")
            return

        # 确保所需的列存在
        if 'accuracy' not in eval_df.columns or 'format_accuracy' not in eval_df.columns:
            print(f"错误: 'eval' 类型的日志中缺少 'accuracy' 或 'format_accuracy' 列。")
            return

        # 找到 'accuracy' 和 'format_accuracy' 的最大值
        max_accuracy = eval_df['accuracy'].max()
        max_format_accuracy = eval_df['format_accuracy'].max()

        # 找到最大值对应的 global_step
        # .idxmax() 返回最大值第一次出现的行的索引
        step_for_max_accuracy = eval_df.loc[eval_df['accuracy'].idxmax()]['global_step']
        step_for_max_format_accuracy = eval_df.loc[eval_df['format_accuracy'].idxmax()]['global_step']

        # 将结果写入指定的输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("训练日志分析结果:\n")
            f.write("="*30 + "\n")
            f.write(f"最高 accuracy: {max_accuracy:.6f} (在 global_step {int(step_for_max_accuracy)})\n")
            f.write(f"最高 format_accuracy: {max_format_accuracy:.6f} (在 global_step {int(step_for_max_format_accuracy)})\n")

        print(f"分析完成！结果已成功保存到 '{output_path}' 文件中。")

    except FileNotFoundError:
        print(f"错误: 未找到输入文件 '{input_path}'。请确保文件存在于正确的路径。")
    except ValueError:
        print(f"错误: '{input_path}' 文件内容可能不是有效的 JSON Lines 格式。请检查文件。")
    except Exception as e:
        print(f"处理文件时发生未知错误: {e}")

if __name__ == "__main__":
    # 调用函数，使用默认的文件名
    analyze_log_file()