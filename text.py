import torch
import numpy as np
import multiprocessing
import time
import sys

def cpu_intensive_task(matrix_size):
    """
    一个在CPU上执行密集计算的函数。
    它会无限循环地进行大型矩阵的乘法运算。
    """
    process_id = multiprocessing.current_process().pid
    print(f"启动 CPU 密集型任务... [进程ID: {process_id}]")
    print(f"CPU 任务将使用 {matrix_size}x{matrix_size} 的 NumPy 矩阵。")
    
    # 确保随机种子在不同进程中不同
    np.random.seed(int(time.time()) + process_id)
    
    i = 0
    while True:
        try:
            # 1. 创建两个大型随机矩阵
            matrix_a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
            matrix_b = np.random.rand(matrix_size, matrix_size).astype(np.float32)
            
            # 2. 执行矩阵乘法，这是非常消耗CPU的计算
            result = np.dot(matrix_a, matrix_b)
            
            i += 1
            # 每 10 次迭代打印一次状态，避免刷屏太快
            if i % 10 == 0:
                print(f"CPU 任务 [PID: {process_id}]: 已完成 {i} 次矩阵乘法。")
                # 强制刷新输出缓冲区，确保信息能实时显示
                sys.stdout.flush()

        except Exception as e:
            print(f"CPU 任务出现错误: {e}")
            break

def gpu_intensive_task(matrix_size):
    """
    一个在GPU上执行密集计算的函数。
    它会无限循环地在GPU上进行大型张量的乘法运算。
    """
    process_id = multiprocessing.current_process().pid
    
    # 1. 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("错误：未检测到 CUDA GPU。GPU 任务无法启动。")
        return
        
    device = torch.device('cuda')
    print(f"启动 GPU 密集型任务... [进程ID: {process_id}]，设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU 任务将使用 {matrix_size}x{matrix_size} 的 PyTorch 张量。")
    
    i = 0
    while True:
        try:
            # 2. 在GPU上创建两个大型随机张量
            tensor_a = torch.randn(matrix_size, matrix_size, device=device)
            tensor_b = torch.randn(matrix_size, matrix_size, device=device)
            
            # 3. 在GPU上执行矩阵乘法
            result = torch.matmul(tensor_a, tensor_b)
            
            # 4. 同步CUDA设备，确保操作完成。这会阻塞CPU直到GPU完成计算。
            torch.cuda.synchronize()
            
            i += 1
            if i % 10 == 0:
                print(f"GPU 任务 [PID: {process_id}]: 已完成 {i} 次矩阵乘法。")
                sys.stdout.flush()
                
        except Exception as e:
            print(f"GPU 任务出现错误: {e}")
            break

if __name__ == '__main__':
    # --- 参数配置 ---
    # 你可以根据你的硬件性能调整这些值。值越大，负载越高。
    # 建议从 1024 开始尝试
    CPU_MATRIX_SIZE = 2048
    GPU_MATRIX_SIZE = 4096 
    
    print("脚本启动... 按下 Ctrl+C 可以随时停止。")
    
    # 创建两个独立的进程
    p_cpu = multiprocessing.Process(target=cpu_intensive_task, args=(CPU_MATRIX_SIZE,))
    p_gpu = multiprocessing.Process(target=gpu_intensive_task, args=(GPU_MATRIX_SIZE,))
    
    try:
        # 启动进程
        p_cpu.start()
        p_gpu.start()
        
        # 等待进程结束（在这个脚本里因为是无限循环，所以会一直等待）
        p_cpu.join()
        p_gpu.join()
        
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，正在终止所有进程...")
        # 终止进程
        p_cpu.terminate()
        p_gpu.terminate()
        print("所有进程已终止。")