#!/usr/bin/env python3
"""
GPU线程监控脚本
用于监控NVIDIA TITAN V GPU的线程使用情况
"""

import subprocess
import time
import sys
import os
import re

def run_command(cmd):
    """运行命令并返回输出"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "命令超时"
    except Exception as e:
        return f"错误: {e}"

def get_gpu_info():
    """获取GPU基本信息"""
    print("=" * 80)
    print("GPU基本信息")
    print("=" * 80)
    
    # 1. 查看GPU型号和数量
    cmd = "nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv"
    print("1. GPU列表:")
    print(run_command(cmd))
    print()
    
    # 2. 查看CUDA核心数（TITAN V特定）
    print("2. NVIDIA TITAN V规格:")
    print("   - CUDA核心: 5120")
    print("   - SM数量: 80 (每个SM 64核心)")
    print("   - 内存带宽: 652.8 GB/s")
    print("   - 显存: 12GB HBM2")
    print()

def monitor_gpu_utilization(interval=2, duration=30):
    """监控GPU利用率"""
    print("=" * 80)
    print(f"GPU利用率监控 (间隔{interval}秒，持续{duration}秒)")
    print("=" * 80)
    
    start_time = time.time()
    end_time = start_time + duration
    
    while time.time() < end_time:
        current_time = time.strftime("%H:%M:%S")
        print(f"\n时间: {current_time}")
        print("-" * 40)
        
        # 查看GPU利用率
        cmd = "nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv"
        output = run_command(cmd)
        print("GPU状态:")
        print(output)
        
        # 查看进程
        cmd = "nvidia-smi pmon -c 1"
        output = run_command(cmd)
        print("GPU进程监控:")
        print(output)
        
        time.sleep(interval)

def check_python_threads():
    """检查Python进程线程"""
    print("=" * 80)
    print("Python进程线程检查")
    print("=" * 80)
    
    # 1. 查找所有Python进程
    cmd = "ps aux | grep python | grep -v grep | grep -v monitor"
    processes = run_command(cmd)
    
    if processes:
        print("1. 运行的Python进程:")
        print(processes)
        print()
        
        # 提取PID
        pids = []
        for line in processes.split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) > 1:
                    pids.append(parts[1])
        
        # 2. 检查每个进程的线程
        for pid in pids[:5]:  # 只检查前5个进程
            print(f"2. 进程 {pid} 的线程信息:")
            
            # 查看线程数
            cmd = f"cat /proc/{pid}/status | grep Threads"
            threads = run_command(cmd)
            print(f"   {threads}")
            
            # 查看线程详情
            cmd = f"ps -T -p {pid}"
            thread_details = run_command(cmd)
            if thread_details and "错误" not in thread_details:
                lines = thread_details.split('\n')
                print(f"   共有 {len(lines)-1} 个线程")
                
                # 统计不同类型的线程
                worker_count = 0
                for line in lines[1:]:  # 跳过标题行
                    if "python" in line:
                        worker_count += 1
                
                print(f"   Python线程数: {worker_count}")
                print(f"   建议: num_workers应设置为 {max(1, worker_count-2)}")
            print()
    else:
        print("未找到运行的Python进程")

def check_num_workers_effectiveness():
    """检查num_workers是否生效"""
    print("=" * 80)
    print("num_workers有效性检查")
    print("=" * 80)
    
    # 1. 查找可能的数据加载进程
    cmd = "ps aux | grep -E '(DataLoader|worker|torch)' | grep -v grep | grep -v monitor"
    processes = run_command(cmd)
    
    if processes:
        print("1. 找到的数据加载相关进程:")
        print(processes)
        print()
        
        # 2. 检查进程树
        print("2. 进程树结构 (示例):")
        print("""
   python (主进程, PID: XXXX)
   ├── python (DataLoader worker 0)
   ├── python (DataLoader worker 1)
   └── python (DataLoader worker 2)
        """)
        
        # 3. 检查系统负载
        print("3. 系统CPU使用情况:")
        cmd = "mpstat -P ALL 1 1 | tail -10"
        cpu_usage = run_command(cmd)
        print(cpu_usage)
        
        # 分析CPU使用
        lines = cpu_usage.split('\n')
        cpu_count = 0
        for line in lines:
            if "Average" in line and "all" not in line and "CPU" not in line:
                parts = line.split()
                if len(parts) > 2:
                    usage = float(parts[2])
                    if usage > 10.0:  # 使用率超过10%
                        cpu_count += 1
        
        print(f"   活跃CPU核心数: {cpu_count}")
        print(f"   建议num_workers设置: {max(1, cpu_count // 5)} (5个GPU)")
    else:
        print("未找到数据加载进程，请确保推理任务正在运行")

def get_optimization_recommendations():
    """获取优化建议"""
    print("=" * 80)
    print("TITAN V优化建议")
    print("=" * 80)
    
    print("1. num_workers设置指南:")
    print("   - 轻负载 (小批量数据): 2-3")
    print("   - 中等负载: 3-4")
    print("   - 重负载 (大数据集): 4-6")
    print("   - 公式: min(CPU核心数/GPU数, 6)")
    print()
    
    print("2. 批量大小建议:")
    print("   - TITAN V有12GB显存，建议:")
    print("   - 训练: batch_size = 10000 (当前设置)")
    print("   - 推理: batch_size = 5000-10000")
    print()
    
    print("3. 监控指标:")
    print("   - GPU利用率: 目标 > 80%")
    print("   - 显存使用: 目标 8-10GB")
    print("   - 温度: 目标 < 85°C")
    print()
    
    print("4. 性能调优命令:")
    print("   # 实时监控")
    print("   watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv'")
    print()
    print("   # 查看进程详情")
    print("   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv")

def main():
    """主函数"""
    print("NVIDIA TITAN V GPU线程监控工具")
    print("=" * 80)
    
    # 检查nvidia-smi是否可用
    test = run_command("nvidia-smi --version")
    if "NVIDIA" not in test:
        print("错误: nvidia-smi不可用，请检查NVIDIA驱动")
        return
    
    # 运行各个检查
    get_gpu_info()
    
    # 询问是否进行实时监控
    response = input("\n是否进行实时GPU监控？(y/n, 默认n): ").strip().lower()
    if response == 'y':
        try:
            duration = int(input("监控时长(秒, 默认30): ") or "30")
            monitor_gpu_utilization(interval=2, duration=duration)
        except ValueError:
            monitor_gpu_utilization()
    
    check_python_threads()
    check_num_workers_effectiveness()
    get_optimization_recommendations()
    
    print("\n" + "=" * 80)
    print("监控完成！")
    print("=" * 80)
    
    # 提供快速命令参考
    print("\n快速命令参考:")
    print("1. 查看GPU状态: nvidia-smi")
    print("2. 查看进程线程: ps -T -p <PID>")
    print("3. 查看进程树: pstree -p <PID>")
    print("4. 实时监控: watch -n 1 nvidia-smi")
    print("5. 查看CPU使用: htop")

if __name__ == "__main__":
    main()
