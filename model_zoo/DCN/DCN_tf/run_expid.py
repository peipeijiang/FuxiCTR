# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""Universal run_expid.py

这是一个“模型目录级”的通用入口脚本，用于训练/推理（含 sweep 推理）。

核心设计点
1) 工作目录与导入：脚本会先切换到脚本所在目录，并把项目根目录加入 sys.path。
     因此它默认按“复制到模型目录”方式使用，让 `import src as model_zoo` 生效。
2) Dashboard/workflow 兼容：
     - 当环境变量 `FUXICTR_WORKFLOW_MODE=dashboard` 且提供 `FUXICTR_TASK_ID` 时，会初始化
         workflow logger，用于把进度推送到 Dashboard（WebSocket）。
     - 若未设置上述环境变量，则 workflow 相关逻辑会自动跳过，不影响命令行训练/推理。
3) 分布式：支持 torch.distributed 的训练/推理（注意推理建议 num_workers=0，避免 DDP + 多进程卡死）。

使用方式
        推荐：复制本文件到你的模型目录中，然后在模型目录执行。
        - 单任务模型：model_zoo/YourModel/run_expid.py
        - 多任务模型：model_zoo/multitask/YourModel/run_expid.py

        例：python run_expid.py --expid your_experiment --mode train --gpu 0
                python run_expid.py --expid your_experiment --mode inference --gpu 0
"""

import os
import sys

# ----------------------------------------------------------------------
# 路径与导入约定（非常关键）
# ----------------------------------------------------------------------
# 1) current_path: 当前脚本所在目录（脚本运行时工作目录会切换到这里）
# 2) root_path: 项目根目录（包含 fuxictr/ 的那一层目录）
# 3) sys.path:
#    - 加入 root_path：保证可以 import fuxictr
#    - 加入 model_zoo/common：保证可复用 common 目录下的公共模块
#
# 这样设计的原因：run_expid.py 通常会被“复制到模型目录”中使用。
# 模型目录下会有一个 src/ 包（导出模型类），脚本通过 `import src as model_zoo`
# 获取模型类，并通过 getattr(model_zoo, params['model']) 实例化。
# ----------------------------------------------------------------------

# Auto-detect project root (works at any directory depth)
current_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_path)
root_path = os.path.abspath(current_path)
for _ in range(5):  # Scan up to 5 levels
    if os.path.exists(os.path.join(root_path, "fuxictr")):
        break
    root_path = os.path.dirname(root_path)
sys.path.append(root_path)  # Add project root
sys.path.append(os.path.join(root_path, "model_zoo/common"))  # Add common directory

# Suppress warnings from deprecated packages
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging

# 说明：各模型目录里通常会放一个轻量的 fuxictr_version.py 用于打印版本/commit。
# 这里做成可选依赖，避免把 run_expid.py 放到不含该文件的目录时报错。
try:
    import fuxictr_version  # noqa: F401
except ModuleNotFoundError:
    fuxictr_version = None  # type: ignore
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything, init_distributed_env, distributed_barrier
from fuxictr.pytorch.dataloaders import RankDataLoader, DataFrameDataLoader, ParquetTransformBlockDataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset
from fuxictr.pytorch.inference import SweepInference, ParquetWriterWrapper, Inferenceutils

# 重要：本脚本预期放在“模型目录”下，该目录应包含 src/（导出模型类）。
# 如果你直接在 model_zoo/common 目录运行，会因为缺少 src/ 而导入失败。
try:
    import src as model_zoo
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Cannot import `src`. This script is intended to be copied into a model directory "
        "that contains a `src/` package (e.g., model_zoo/YourModel/src). "
        f"Current script dir: {current_path}"
    ) from e
import gc
import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.distributed as dist
import signal
import threading
import time
import re
import pyarrow.parquet as pq
from tqdm import tqdm


_stop_event = threading.Event()


def _install_signal_handlers():
    """Install SIGINT/SIGTERM handlers for graceful shutdown.

    推理（尤其是长时间的 Parquet 流式推理）可能会被用户中断。
    这里设置 stop flag，并尽可能清理 Dashboard 任务状态与 PyTorch 临时文件。
    """
    def _handle(sig, frame):
        logging.warning(f"Received signal {sig}, requesting stop...")
        _stop_event.set()
        # 被信号中断时尽量清理：
        # - Dashboard task state：避免前端残留“运行中”的任务
        # - PyTorch tmp：避免下次运行被旧文件干扰
        try:
            cleanup_task_state()
            cleanup_pytorch_tmp_files()
        except:
            pass
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handle)
        except Exception:
            pass


def cleanup_task_state(pid=None):
    """清理 dashboard/state/tasks 下的任务状态文件。

    Dashboard 模式下，前端会持续读取该状态文件展示进度。
    若进程退出/被中断，清理对应 pid 的 state 文件可避免前端残留“僵尸任务”。
    """
    if pid is None:
        pid = os.getpid()

    # task state 文件命名里一般会包含 pid（例如 *_12345.json）。
    # 这里按 pid 做匹配并删除，尽量做到“只清自己的”。
    task_state_dir = os.path.join(root_path, "dashboard", "state", "tasks")
    if not os.path.exists(task_state_dir):
        return

    try:
        for f in os.listdir(task_state_dir):
            if f.endswith(".json") and f"_{pid}.json" in f:
                fpath = os.path.join(task_state_dir, f)
                os.remove(fpath)
                logging.info(f"Cleaned up task state: {f}")
                break
    except Exception as e:
        logging.warning(f"Failed to cleanup task state: {e}")


def cleanup_pytorch_tmp_files():
    """清理 PyTorch 在 model_zoo 下产生的临时文件。

    部分 DDP/RPC 场景会生成 tmp*/_remote_module_non_scriptable.py；这里尽量清理。
    """
    try:
        # 只在 model_zoo/ 下做清理，避免误删项目外部的 tmp。
        # pattern 目标：tmp*/_remote_module_non_scriptable.py（DDP/RPC 某些路径会生成）
        pattern = os.path.join(root_path, "model_zoo", "**", "tmp*", "_remote_module_non_scriptable.py")
        matching_files = glob.glob(pattern, recursive=True)

        for tmp_file in matching_files:
            try:
                # 尝试删除文件；如果父目录空了，再删除父目录。
                tmp_dir = os.path.dirname(tmp_file)
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
                if os.path.exists(tmp_dir) and tmp_dir.startswith(os.path.join(root_path, "model_zoo")):
                    # Only remove if it's empty or only contains the remote_module file
                    try:
                        os.rmdir(tmp_dir)
                        logging.info(f"Cleaned up PyTorch tmp dir: {tmp_dir}")
                    except OSError:
                        # Directory not empty, just remove the file
                        logging.info(f"Cleaned up PyTorch tmp file: {tmp_file}")
            except Exception as e:
                logging.warning(f"Failed to cleanup {tmp_file}: {e}")
    except Exception as e:
        logging.warning(f"Failed to cleanup PyTorch tmp files: {e}")


# ========================================================================
# 辅助函数 - 提取的公共逻辑
# ========================================================================

def _get_processed_root(params):
    """获取 processed_root 参数，统一处理。

    一些数据流会把处理后的特征/索引数据放在 processed_root 下；
    若没有显式配置，则退化为 data_root。
    """
    return params.get('processed_root', params['data_root'])


def _get_data_dir(params, processed_root=None):
    """根据 dataset_id 拼接数据目录。

    data_dir = {processed_root}/{dataset_id}
    """
    if processed_root is None:
        processed_root = _get_processed_root(params)
    return os.path.join(processed_root, params['dataset_id'])


def _prepare_params_with_processed_root(params):
    """准备包含 processed_root 的参数副本。

    FeatureProcessor/数据构建有时依赖 processed_root；为了不污染原 params，
    这里复制一份并补齐该字段。
    """
    processed_root = _get_processed_root(params)
    params_with_processed = params.copy()
    params_with_processed['processed_root'] = processed_root
    return params_with_processed, processed_root


def _acquire_inference_lock(lock_file, max_retries=5):
    """获取推理锁文件，带重试机制

    Args:
        lock_file: 锁文件路径
        max_retries: 最大重试次数

    Returns:
        bool: 是否成功获取锁

    Raises:
        RuntimeError: 重试次数用尽后仍未获取锁
    """
    # 说明：锁文件用于避免“同一输出目录”被重复推理同时写入。
    # - 如果发现锁文件存在：
    #   - 若锁文件过旧（>300s）认为是异常退出残留，直接删除
    #   - 若锁文件较新：尝试从内容中解析 PID，并检查该 PID 是否仍存活
    #     - 存活则报错（认为已经有推理在运行）
    #     - 不存活则删除锁文件并重试
    for retry in range(max_retries):
        try:
            if os.path.exists(lock_file):
                lock_age = time.time() - os.path.getmtime(lock_file)
                if lock_age > 300:
                    logging.warning(f"Removing stale lock file (age: {lock_age:.0f}s)")
                    os.remove(lock_file)
                else:
                    # 检查锁文件中的进程是否还在运行
                    try:
                        with open(lock_file, 'r') as f:
                            content = f.read()
                            if 'PID:' in content:
                                match = re.search(r'PID:\s*(\d+)', content)
                                if match:
                                    pid = int(match.group(1))
                                    try:
                                        os.kill(pid, 0)
                                        raise RuntimeError(f"Inference already running (PID: {pid})")
                                    except OSError:
                                        logging.warning(f"Process {pid} not found, removing stale lock file")
                                        os.remove(lock_file)
                    except Exception:
                        os.remove(lock_file)
                    else:
                        raise RuntimeError(f"Inference already running (age: {lock_age:.0f}s)")

            # 写入新的锁文件
            with open(lock_file, 'w') as f:
                f.write(f"PID: {os.getpid()}, Time: {time.time()}")
            return True

        except RuntimeError:
            if retry == max_retries - 1:
                raise
            else:
                logging.warning(f"Retry {retry + 1}/{max_retries}: Lock acquisition failed")
                time.sleep(3)
        except Exception as e:
            if retry == max_retries - 1:
                raise RuntimeError(f"Failed to acquire lock after {max_retries} retries: {e}")
            else:
                logging.warning(f"Retry {retry + 1}/{max_retries}: {e}")
                time.sleep(3)

    return False


def _setup_inference_output(params, args, rank, data_dir):
    """设置推理输出目录和锁文件。

    注意：分布式推理时，多个 rank 会并行执行。锁文件必须只由 rank=0 创建，
    否则其余 rank 会误判“已有推理在运行”从而退出。

    Args:
        params: 参数字典
        args: 命令行参数
        rank: 当前进程 rank
        data_dir: 数据目录

    Returns:
        dict: 包含 output_dir 和 lock_file 的字典
    """
    # 推理输出会放在数据目录下，以 expid 作为隔离：
    # {data_dir}/{expid}_inference_result/
    # 里面会产生 part_*.parquet（以及分布式时的 part_*_rank*.parquet 临时文件）
    output_dir = os.path.join(data_dir, f"{args['expid']}_inference_result")
    lock_file = os.path.join(output_dir, ".inference_lock")

    # 只有 rank 0 负责创建目录与加锁；其它 rank 等待 barrier 后再继续。
    # 这样避免 DDP 推理时多个 rank 同时调用 _acquire_inference_lock 导致互相阻塞/误判。
    completed = set()
    if rank == 0:
        _acquire_inference_lock(lock_file)
        os.makedirs(output_dir, exist_ok=True)
        completed = get_completed_files(output_dir)
    if completed:
        logging.info(f"Resume mode: {len(completed)} files already completed: {sorted(completed)}")

    return {
        'output_dir': output_dir,
        'lock_file': lock_file,
        'completed': completed
    }


def run_train(model, feature_map, params, args, workflow_logger=None):
    """Training function.

    Args:
        model: Model instance for training
        feature_map: Feature map for data processing
        params: Parameters dictionary
        args: Arguments dictionary
        workflow_logger: Optional WorkflowLogger for Dashboard WebSocket broadcasting
    """
    # RankDataLoader 会根据 feature_map/params 加载 train/valid split。
    # 这里 stage='train' 会产出 train_gen 和 valid_gen。
    rank = params.get('distributed_rank', 0)
    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()

    # Dashboard 模式：把 workflow_logger 挂到模型上（模型内部可选用来上报 batch 级进度）。
    # 这不会影响普通训练：workflow_logger=None 时直接跳过。
    if workflow_logger:
        model._workflow_logger = workflow_logger

    # 训练主循环：内部会处理优化器、loss、early stop、checkpoint 等。
    model.fit(train_gen, validation_data=valid_gen, **params)

    if rank == 0:
        logging.info('****** Validation evaluation ******')
    # 训练后立刻对 valid 做一次评估，作为最终写入 CSV 的结果。
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()

    test_result = {}
    # 可选：如果配置了 test_data，则额外评估 test 集。
    if params.get("test_data"):
        if rank == 0:
            logging.info('******** Test evaluation ********')
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        test_result = model.evaluate(test_gen)

    if rank == 0:
        # 训练结束后写结果摘要：
        # - Dashboard 模式：追加到 config/ 目录下的 config.csv（便于前端读取展示）
        # - Workflow 模式：写到 workflow_models/results/ 下，避免污染 config.csv
        run_mode = params.get("run_mode", "dashboard")

        if run_mode == "workflow":
            # Workflow 模式：生成在 workflow_models/results/ 下
            model_name = params.get("model", "unknown")
            results_dir = "workflow_models/results"
            os.makedirs(results_dir, exist_ok=True)
            csv_path = os.path.join(results_dir, f"{model_name}_workflow_results.csv")
        else:
            # Dashboard 模式：追加到 config.csv
            config_dir = os.path.dirname(args.get('config', '.'))
            csv_path = os.path.join(config_dir, "config.csv")  # 固定为 config.csv

        # 写入 CSV（一行一个 expid 的结果摘要），便于 Dashboard 或 workflow 统一收集。
        with open(csv_path, 'a+') as fw:
            fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
                .format(datetime.now().strftime('%Y%m%d-%H%M%S'),
                        ' '.join(sys.argv), args['expid'], params['dataset_id'],
                        "N.A.", print_to_list(valid_result), print_to_list(test_result)))


def get_completed_files(output_dir):
    """获取已完成的输出分片（part_N.parquet）。

    推理支持“断点续跑”：如果 output_dir 下已经存在 part_0.parquet/part_1.parquet...
    则这些分片会被视为已完成，后续推理会跳过对应输入文件索引。
    """
    completed_files = set()
    if not os.path.exists(output_dir):
        return completed_files
    for f in glob.glob(os.path.join(output_dir, "part_*.parquet")):
        match = re.match(r'part_(\d+)\.parquet', os.path.basename(f))
        if match:
            completed_files.add(int(match.group(1)))
    return completed_files


def merge_distributed_results(output_dir, world_size):
    """合并分布式推理的临时输出。

    早期设计可能会输出 part_{fid}_rank{r}.parquet，最后再做全局 merge。
    当前推理流程也会尽量在每个 rank 完成后“就地 finalize”（rename），但如果
    仍残留临时文件，这里会兜底 merge。
    """
    part_files = glob.glob(os.path.join(output_dir, "part_*_rank*.parquet"))
    if not part_files:
        return

    part_groups = {}
    for f in part_files:
        basename = os.path.basename(f)
        match = re.match(r'part_(\d+)_rank\d+\.parquet', basename)
        if match:
            part_num = int(match.group(1))
            if part_num not in part_groups:
                part_groups[part_num] = []
            part_groups[part_num].append(f)

    for part_num, files in sorted(part_groups.items()):
        dfs = []
        for f in sorted(files):
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
                os.remove(f)
            except Exception as e:
                logging.warning(f"Failed to read or remove {f}: {e}")

        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_file = os.path.join(output_dir, f"part_{part_num}.parquet")
            merged_df.to_parquet(merged_file, index=False)
            logging.info(f"Merged {len(files)} rank files into {merged_file}")


def run_inference(model, feature_map, params, args, workflow_logger=None):
    """Universal inference function supporting single-task, multi-task, and sweep modes.

    Args:
        model: Model instance for inference
        feature_map: Feature map for data processing
        params: Parameters dictionary
        args: Arguments dictionary
        workflow_logger: Optional WorkflowLogger for Dashboard WebSocket broadcasting
    """
    # 推理通常耗时较长：安装信号处理，支持 Ctrl+C/kill 时优雅停止。
    _install_signal_handlers()
    distributed = params.get('distributed', False)
    rank = params.get('distributed_rank', 0)
    world_size = params.get('distributed_world_size', 1)

    # 加载训练好的 checkpoint。
    # model.checkpoint 通常由 params 里的 model_root/expid 等规则拼出来。
    model.load_weights(model.checkpoint)

    # FeatureProcessor 用于把原始字段映射为模型所需的 tensor（含 id->token 等）。
    # 这里读取训练阶段落盘的 feature_processor.pkl。
    params_with_processed, processed_root = _prepare_params_with_processed_root(params)
    data_dir = _get_data_dir(params, processed_root)
    feature_encoder = FeatureProcessor(**params_with_processed).load_pickle(
        params.get('pickle_file', os.path.join(data_dir, "feature_processor.pkl"))
    )
    # 业务侧常见：手机号等字段在 parquet 里可能被解析为 int，需要强制为 str。
    feature_encoder.dtype_dict.update({'phone': str, 'phone_md5': str})

    # ------------------------------------------------------------------
    # 输入文件解析
    # ------------------------------------------------------------------
    # infer_data 可以是：
    # - 一个目录：目录下找 *.parquet 或 *.csv（按文件名数字排序）
    # - 一个文件：单文件推理
    # files 的顺序决定了后续 part_{i}.parquet 的编号。
    infer_data = params['infer_data']
    if os.path.isdir(infer_data):
        parquet_files = glob.glob(os.path.join(infer_data, "*.parquet"))
        csv_files = glob.glob(os.path.join(infer_data, "*.csv"))

        def extract_number(filename):
            basename = os.path.basename(filename)
            match = re.search(r'(\d+)', basename)
            return int(match.group(1)) if match else 0

        if parquet_files:
            files = sorted(parquet_files, key=extract_number)
            data_format = 'parquet'
        elif csv_files:
            files = sorted(csv_files, key=extract_number)
            data_format = 'csv'
        else:
            files = []
            data_format = 'parquet'
            if rank == 0:
                logging.warning(f"No parquet or csv files found in {infer_data}")
    else:
        files = [infer_data]
        data_format = 'parquet' if infer_data.endswith('.parquet') else 'csv'
        if rank == 0:
            logging.info(f"Single file inference: {infer_data}")

    # 输出目录（实际会在 _setup_inference_output 中创建并加锁）
    output_dir = os.path.join(data_dir, f"{args['expid']}_inference_result")

    # ------------------------------------------------------------------
    # Sweep 推理（多场景/多域）
    # ------------------------------------------------------------------
    # SweepInference 会根据 params 自动判断是否启用 sweep：
    # - 例如配置 sweep_col、domains_per_pass 等
    # - run_batch 内部负责把同一批数据按 domain 拆分并调用模型
    sweep_inference = SweepInference(model, feature_map, params)
    sweep_inference.set_id_to_token(feature_encoder)

    if rank == 0:
        mode_str = "SWEEP" if sweep_inference.sweep_enabled else "NORMAL"
        logging.info(f"Inference Mode: {mode_str}")
        if sweep_inference.sweep_enabled:
            logging.info(f"Sweep column: {sweep_inference.sweep_col}, Domains per pass: {sweep_inference.domains_per_pass}")

    # 输出目录与锁文件：分布式场景下只允许 rank0 加锁。
    # 注意：rank0 加锁后，其它 rank 会在 barrier 处同步，避免误判。
    output_ctx = _setup_inference_output(params, args, rank, data_dir)
    output_dir = output_ctx['output_dir']
    lock_file = output_ctx['lock_file']

    if distributed and dist.is_initialized():
        distributed_barrier()

    if rank == 0:
        logging.info('******** Start Inference ********')
        logging.info(f"Results will be saved to: {output_dir}")

    warnings.simplefilter("ignore")
    logger = logging.getLogger()
    original_level = logger.level

    has_data = False

    # Log file distribution info
    if rank == 0:
        logging.info(f"Total files to process: {len(files)}")
        logging.info(f"World size: {world_size}, Rank: {rank}")

    # Resume 模式：rank0 扫描已完成的 part_*.parquet，并广播到所有 rank。
    # 这样每个 rank 在分配文件时，会跳过已完成部分。
    completed_files = get_completed_files(output_dir) if rank == 0 else set()

    # Broadcast completed files to all ranks
    if distributed and dist.is_initialized():
        completed_list = list(completed_files)
        shared = [completed_list]
        dist.broadcast_object_list(shared, src=0)
        completed_files = set(shared[0])

    # ------------------------------------------------------------------
    # 文件分配（按文件索引对 world_size 取模）
    # ------------------------------------------------------------------
    # 约定：第 i 个输入文件 -> part_i.parquet
    # 分布式时每个 rank 处理一部分 i：i % world_size == rank
    rank_files = []
    file_indices = []

    for i, f in enumerate(files):
        if i not in completed_files:
            if i % world_size == rank:
                rank_files.append(f)
                file_indices.append(i)

    if completed_files:
        logging.info(f"Rank {rank}: Skipping {len(completed_files)} completed files")
    logging.info(f"Rank {rank}: Processing {len(rank_files)} files (indices: {file_indices})")

    # DataLoader 内部会把 rank_files 重新编号为 block_idx（0..len(rank_files)-1）。
    # 我们需要把 block_idx 映射回“原始文件索引 i”，用于输出文件命名与进度显示。
    block_idx_to_file_idx = {block_idx: file_idx for block_idx, file_idx in enumerate(file_indices)}

    # Log configuration
    if rank == 0:
        num_workers_value = params.get('num_workers', 1)
        batch_size_value = params.get('batch_size', 10000)
        chunk_size_value = params.get('infer_chunk_size', 10000)

        is_ddp = distributed and dist.is_initialized()

        logging.info(f"=" * 60)
        logging.info(f"Rank {rank}: Inference Configuration:")
        logging.info(f"  - Distributed: {distributed}")
        logging.info(f"  - DDP initialized: {is_ddp}")
        logging.info(f"  - Num Workers: {num_workers_value}")
        if is_ddp and num_workers_value > 0:
            logging.warning(f"  WARNING: DDP + num_workers={num_workers_value} may cause hangs")
        logging.info(f"  - Batch Size: {batch_size_value}")
        logging.info(f"  - Chunk Size: {chunk_size_value}")
        logging.info(f"  - Files to process: {len(rank_files)}")
        logging.info(f"=" * 60)

    # 经验：DDP + num_workers>0 在某些环境容易 hang；默认这里会把 num_workers 设为 0。
    logging.info(f"Rank {rank}: Using num_workers={params.get('num_workers', 1)} for inference")

    if rank_files:
        # 扫描 parquet 元信息以获得每个文件的总行数，用于进度条。
        # 注意：这里只对本 rank 的 rank_files 扫描，避免 rank0 扫全量造成开销。
        file_total_rows = {}
        for block_idx, file_path in enumerate(rank_files):
            try:
                parquet_file = pq.ParquetFile(file_path)
                file_total_rows[block_idx] = parquet_file.metadata.num_rows
            except Exception as e:
                logging.warning(f"Failed to read metadata for {file_path}: {e}")
                file_total_rows[block_idx] = 0

        if rank == 0:
            logging.info(f"Rank {rank}: Scanned {len(file_total_rows)} files, total rows: {sum(file_total_rows.values())}")

        # ------------------------------------------------------------------
        # DataLoader 构建
        # ------------------------------------------------------------------
        # 这里复用 RankDataLoader 的测试阶段路径，但替换 data_loader 为
        # ParquetTransformBlockDataLoader，实现“块级 streaming”。
        #
        # 关键字段：
        # - feature_encoder：负责把每个 batch 的原始列转换为 tensor
        # - id_cols：推理结果中需要写回的 id 列（业务自定义）
        # - chunk_size：每次从 parquet 读取/转换的块大小，影响吞吐与内存
        inference_params = params.copy()
        inference_params['test_data'] = rank_files
        inference_params['data_loader'] = ParquetTransformBlockDataLoader
        inference_params['feature_encoder'] = feature_encoder
        inference_params['id_cols'] = ['phone', 'phone_md5']
        inference_params['shuffle'] = False
        inference_params['batch_size'] = params.get('batch_size', 10000)
        inference_params['num_workers'] = params.get('num_workers', 0)
        inference_params['multiprocessing_context'] = params.get('multiprocessing_context', 'fork')
        inference_params['chunk_size'] = params.get('infer_chunk_size', 10000)

        # test_gen 是一个迭代器：每次 yield 一个 batch_data(dict)，
        # 其中可能包含内部 meta 字段（例如 _file_idx）用于表示该 batch 来自哪个文件。
        test_gen = RankDataLoader(feature_map, stage='test', **inference_params).make_iterator()

        model._verbose = 0

        # Track progress
        file_processed_rows = {}
        file_progress_logged = {}
        file_progress_line_len = 0
        file_progress_current = None

        # 写出器：内部做 buffer 聚合，避免每个 batch 都落盘造成大量小文件/IO。
        # 分布式时临时文件会带 rank 后缀，后续会尝试 rename 成最终文件。
        writer_wrapper = ParquetWriterWrapper(
            output_dir,
            buffer_limit=params.get('sweep_write_buffer_rows', 50000),
            filename_fmt="part_{fid}_rank" + str(rank) + ".parquet"
        )

        try:
            # 只有在 TTY 环境下才启用 tqdm 动态进度条；否则只打印日志。
            is_tty = sys.stdout.isatty()

            # 进度条策略：
            # - total_pbar：本 rank 需要处理的文件数（不是行数）
            # - file_pbar：当前文件完成百分比
            total_pbar = None
            file_pbar = None
            batch_pbar = None

            if is_tty:
                # 如果有 workflow_logger（Dashboard 模式、且 rank0），则把 tqdm 进度
                # 通过 WebSocket 适配器推给前端。
                if workflow_logger and rank == 0:
                    from fuxictr.pytorch.utils import create_progress_adapter

                    # Create a custom iterable that tracks file completion
                    total_pbar = create_progress_adapter(
                        iterable=None,  # We'll manually update n
                        logger=workflow_logger,
                        step_name="inference_total",
                        rank=rank,
                        world_size=world_size,
                        total=len(file_total_rows),
                        desc=f"Rank {rank}: Total",
                        position=0,
                        leave=True,
                        dynamic_ncols=True,
                        disable=False
                    )
                else:
                    total_pbar = tqdm(
                        total=len(file_total_rows),
                        desc=f"Rank {rank}: Total",
                        position=0,
                        leave=True,
                        dynamic_ncols=True,
                        disable=False
                    )

                # Current file progress bar (position 1)
                if workflow_logger and rank == 0:
                    from fuxictr.pytorch.utils import create_progress_adapter
                    file_pbar = create_progress_adapter(
                        iterable=None,  # We'll manually update n
                        logger=workflow_logger,
                        step_name="inference_file",
                        rank=rank,
                        world_size=world_size,
                        total=100,
                        desc=f"Rank {rank}: Current file",
                        position=1,
                        leave=False,
                        dynamic_ncols=True,
                        disable=False,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}% [{elapsed}<{remaining}]"
                    )
                else:
                    file_pbar = tqdm(
                        total=100,
                        desc=f"Rank {rank}: Current file",
                        position=1,
                        leave=False,
                        dynamic_ncols=True,
                        disable=False,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}% [{elapsed}<{remaining}]"
                    )

            # Track current file for progress update
            current_file_id = None
            current_file_total = 0
            current_file_processed = 0

            # Track last logged progress for dashboard
            last_logged_files_done = -1
            batch_idx = 0

            # 推理主循环：逐 batch 读取 -> sweep 推理 -> 写出 parquet。
            for batch_data in test_gen:
                if _stop_event.is_set():
                    logging.warning(f"Rank {rank}: Stop requested, exiting inference loop")
                    break

                # _file_idx 是 DataLoader 生成的 meta 字段：表示本 batch 每条样本来自哪个 block 文件。
                # 注意：这里会 pop 掉 meta 字段，避免进入模型。
                file_idx_arr = batch_data.pop("_file_idx")
                ids_batch = {c: batch_data.pop(c) for c in list(batch_data.keys()) if c in ['phone', 'phone_md5']}

                # 把 block_idx 映射回原始文件索引（对应 part_{i}.parquet 的 i）
                original_file_indices = []
                for b_idx in file_idx_arr:
                    if isinstance(b_idx, torch.Tensor):
                        b_idx = b_idx.item()
                    original_file_indices.append(block_idx_to_file_idx.get(b_idx, b_idx))

                original_file_indices = np.array(original_file_indices)
                unique_files = np.unique(original_file_indices)

                # 预先整理 id 缓存（写结果时需要把 id 列与预测对齐）。
                id_cache = Inferenceutils._prepare_id_cache(ids_batch, unique_files, original_file_indices)

                # 通过 SweepInference 运行 batch：内部会调用模型 forward，并把输出写入 writer_wrapper。
                batch_success = sweep_inference.run_batch(batch_data, unique_files, id_cache, writer_wrapper)

                if batch_success:
                    has_data = True

                # 更新进度：file_processed_rows 记录每个 block_idx 已处理的“样本数”。
                # 注意：这里按 batch 中出现的 block_idx 累加 1，依赖 DataLoader 约定：
                # file_idx_arr 的单位是“行/样本”，因此每条样本计 1。
                files_completed_this_batch = []
                for b_idx in file_idx_arr:
                    if isinstance(b_idx, torch.Tensor):
                        b_idx = b_idx.item()
                    file_processed_rows[b_idx] = file_processed_rows.get(b_idx, 0) + 1

                    # Check if file is completed
                    total_rows = file_total_rows.get(b_idx, 0)
                    if total_rows > 0 and file_processed_rows[b_idx] >= total_rows:
                        if b_idx not in files_completed_this_batch:
                            files_completed_this_batch.append(b_idx)

                # Count total files completed
                files_done_count = sum(1 for idx, rows in file_processed_rows.items()
                                      if file_total_rows.get(idx, 0) > 0 and rows >= file_total_rows[idx])

                # 更新进度条与（可选）WebSocket 推送
                if is_tty and total_pbar:
                    total_pbar.n = files_done_count
                    total_pbar.refresh()

                    # Broadcast progress to WebSocket
                    if workflow_logger and rank == 0:
                        try:
                            pct = int(files_done_count * 100 / len(file_total_rows)) if len(file_total_rows) > 0 else 0
                            workflow_logger.progress(
                                step="inference",
                                current=files_done_count,
                                total=len(file_total_rows),
                                message=f"{pct}% files completed"
                            )
                        except Exception:
                            pass

                    # Update current file progress
                    if len(unique_files) > 0:
                        # Get the file being processed (first one in unique_files)
                        current_fid = unique_files[0]
                        if current_fid != current_file_id:
                            current_file_id = current_fid
                            current_file_total = file_total_rows.get(current_fid, 0)
                            current_file_processed = 0

                        current_file_processed = file_processed_rows.get(current_fid, 0)
                        if current_file_total > 0:
                            pct = min(100, int(current_file_processed * 100 / current_file_total))
                            file_pbar.n = pct
                            original_fid = block_idx_to_file_idx.get(current_fid, current_fid)
                            file_pbar.set_description(f"Rank {rank}: part_{original_fid}")
                            file_pbar.refresh()

                # 日志节流：每完成一个文件，或每 50 个 batch 打印一次。
                batch_idx += 1
                if rank == 0 and (files_done_count > last_logged_files_done or batch_idx % 50 == 0):
                    last_logged_files_done = files_done_count
                    pct = int(files_done_count * 100 / len(file_total_rows)) if len(file_total_rows) > 0 else 0
                    logging.info(f"Progress: {files_done_count}/{len(file_total_rows)} files completed ({pct}%)")

                del batch_data

            # 关闭进度条（避免终端残留）
            if is_tty:
                if total_pbar:
                    total_pbar.close()
                if file_pbar:
                    file_pbar.close()

        finally:
            writer_wrapper.close()
            logger.setLevel(original_level)

    # ------------------------------------------------------------------
    # finalize：把 part_*_rank{rank}.parquet 改名为 part_*.parquet
    # ------------------------------------------------------------------
    # 这样做的好处：
    # - 每个 rank 处理完就尽快产出最终文件，便于下游消费
    # - 避免最后统一 merge 带来的峰值内存/IO
    finalize_count = 0
    temp_files = glob.glob(os.path.join(output_dir, f"part_*_rank{rank}.parquet"))
    for temp_file in temp_files:
        try:
            basename = os.path.basename(temp_file)
            # 从 part_0_rank0.parquet 提取 part_0
            match = re.match(r'part_(\d+)_rank\d+\.parquet', basename)
            if match:
                part_num = match.group(1)
                final_file = os.path.join(output_dir, f"part_{part_num}.parquet")
                # 检查是否已存在（可能被其他rank重命名了）
                if not os.path.exists(final_file):
                    os.rename(temp_file, final_file)
                    finalize_count += 1
                    if rank == 0:  # 只打印rank 0的日志
                        logging.info(f"Finalized part_{part_num}.parquet")
                else:
                    # 文件已存在，删除临时文件
                    os.remove(temp_file)
        except Exception as e:
            logging.warning(f"Rank {rank}: Failed to finalize {temp_file}: {e}")

    if finalize_count > 0 and rank == 0:
        logging.info(f"Rank {rank}: Finalized {finalize_count} files")

    # 分布式同步：等待所有 rank 完成 finalize（或提前停止）
    if distributed and dist.is_initialized() and not _stop_event.is_set():
        distributed_barrier()

    if rank == 0:
        if has_data:
            logging.info(f"Inference completed. Data saved in: {output_dir}")
            remaining_temp_files = glob.glob(os.path.join(output_dir, "part_*_rank*.parquet"))
            if remaining_temp_files:
                logging.warning(f"Found {len(remaining_temp_files)} unfinalized temp files, merging...")
                try:
                    merge_distributed_results(output_dir, world_size)
                except Exception as e:
                    logging.warning(f"Failed to merge remaining results: {e}")
            else:
                logging.info("All files finalized successfully during inference")
        else:
            logging.warning("No data found in infer_data!")

        # 清理锁文件（只在 rank0）。
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
                logging.info(f"Lock file removed: {lock_file}")
        except Exception as e:
            logging.warning(f"Failed to remove lock file: {e}")

        # 清理 dashboard task state（rank0）
        try:
            cleanup_task_state()
        except Exception as e:
            logging.warning(f"Failed to cleanup task state: {e}")

        # 清理 PyTorch 临时文件（rank0）
        try:
            cleanup_pytorch_tmp_files()
        except Exception as e:
            logging.warning(f"Failed to cleanup PyTorch tmp files: {e}")


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # 命令行参数
    # ------------------------------------------------------------------
    # --config: 配置目录（通常是 ./config/）
    # --expid : 实验 id（对应配置中的某个 section）
    # --mode  : train / inference
    # --gpu   : 单机时指定 GPU id；分布式时会被 local_rank 覆盖
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'], help='The running mode.')
    parser.add_argument('--distributed', action='store_true', help='Enable torch.distributed for multi-GPU training')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='Torch distributed backend to use')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed by torchrun')
    args = vars(parser.parse_args())

    # ------------------------------------------------------------------
    # 读取配置 & 初始化分布式环境
    # ------------------------------------------------------------------
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    distributed, rank, world_size, local_rank = init_distributed_env(args)
    params['gpu'] = local_rank if distributed else args['gpu']
    params['distributed'] = distributed
    params['distributed_rank'] = rank
    params['distributed_world_size'] = world_size
    params['local_rank'] = local_rank
    # logger：只让 rank0 负责 set_logger（写日志文件等），其余 rank 用 basicConfig。
    if rank == 0:
        set_logger(params)
    else:
        logging.basicConfig(level=logging.INFO if params.get('verbose', 0) > 0 else logging.WARNING)
    logging.info("Rank {} initialized (world size {}).".format(rank, world_size))
    logging.info("Params: " + print_to_json(params))
    # 保证可复现（注意：DDP 仍可能因算子/通信导致非严格确定性）。
    seed_everything(seed=params['seed'])

    # ------------------------------------------------------------------
    # 构建/加载 feature_map（描述特征类型、embedding 等）
    # ------------------------------------------------------------------
    # 对 parquet + 训练模式：rank0 会先 build_dataset（会产出 train/valid/test split），
    # 然后广播给其它 rank。
    processed_root = _get_processed_root(params)
    data_dir = _get_data_dir(params, processed_root)
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "parquet" and args['mode'] == 'train':
        data_splits = (None, None, None)
        if rank == 0:
            # 使用统一的辅助函数准备参数
            params_with_processed, _ = _prepare_params_with_processed_root(params)
            feature_encoder = FeatureProcessor(**params_with_processed)
            data_splits = build_dataset(feature_encoder, **params_with_processed)
        if distributed and dist.is_initialized():
            shared = [data_splits]
            dist.broadcast_object_list(shared, src=0)
            data_splits = shared[0]
        params["train_data"], params["valid_data"], params["test_data"] = data_splits
        if distributed and dist.is_initialized():
            distributed_barrier()
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    if rank == 0:
        logging.info("Feature specs: " + print_to_json(feature_map.features))

    # ------------------------------------------------------------------
    # 加载模型
    # ------------------------------------------------------------------
    # 约定：params['model'] 是模型类名字符串；模型目录下的 src/__init__.py
    # 需要导出同名类，才能通过 getattr(model_zoo, ...) 找到。
    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters()

    # Dashboard 模式才初始化 workflow_logger：
    # - `FUXICTR_WORKFLOW_MODE=dashboard`：表示由 Dashboard 驱动
    # - `FUXICTR_TASK_ID`：前端创建的任务 id，用于建立 WebSocket 映射
    # 未设置时不会影响 CLI。
    workflow_logger = None
    if os.environ.get('FUXICTR_WORKFLOW_MODE') == 'dashboard':
        try:
            from fuxictr.workflow.utils.logger import get_workflow_logger
            task_id = os.environ.get('FUXICTR_TASK_ID')
            if task_id:
                workflow_logger = get_workflow_logger(int(task_id))
                if rank == 0:
                    logging.info(f"Workflow logger initialized for task {task_id}")
        except Exception as e:
            logging.warning(f"Failed to initialize workflow logger: {e}")

    # ------------------------------------------------------------------
    # 运行模式：训练 or 推理
    # ------------------------------------------------------------------
    if args['mode'] == 'train':
        run_train(model, feature_map, params, args, workflow_logger=workflow_logger)
    elif args['mode'] == 'inference':
        run_inference(model, feature_map, params, args, workflow_logger=workflow_logger)

    # 分布式收尾：barrier + destroy_process_group
    if distributed and dist.is_initialized():
        distributed_barrier()
        dist.destroy_process_group()

    # 正常结束时也做一次清理（仅 rank0）
    if rank == 0:
        try:
            cleanup_task_state()
            cleanup_pytorch_tmp_files()
        except Exception as e:
            logging.warning(f"Failed to cleanup on completion: {e}")
