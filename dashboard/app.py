import streamlit as st
import logging
import os
import subprocess
import sys
import time
import signal
import pandas as pd
import yaml
import shutil
import json
import base64
import copy
import psutil
import re
import socket
from datetime import datetime
from code_editor import code_editor
from collections import OrderedDict

def get_gpu_options():
    options = [-1]
    labels = ["CPU (-1)"]
    
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                options.append(i)
                labels.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception:
        pass
    
    return options, labels


def determine_mkl_threading_layer(devices):
    """Use GNU runtime whenever a GPU device participates."""
    return "GNU" if any(d is not None and d >= 0 for d in devices) else "INTEL"

def normalize_data_path(root, path):
    """Return absolute/normalized data path.

    - If `path` 已经是绝对路径，直接归一化返回。
    - 如果 `path` 已经以 `root` 开头（例如用户手动填了 `${root}/xxx`），避免重复拼接。
    - 否则在有 root 时进行拼接。
    """
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    root_norm = os.path.normpath(root) if root else ""
    path_norm = os.path.normpath(path)
    if root_norm and (path_norm.startswith(root_norm) or path.startswith(root_norm)):
        return path_norm
    if root_norm:
        return os.path.normpath(os.path.join(root_norm, path))
    return path_norm

# Set page config
st.set_page_config(
    page_title="XFDL 实验平台",
    page_icon="🍭",
    layout="wide",
    initial_sidebar_state="expanded"
)

USER_OPTIONS = [
    "yeshao",
    "chenzeng2", "cywang50", "gjwang5", "gxwang9",
    "hkhu3", "junzhang56", "mxsong", "qiancao6",
    "taozhang48", "wenzhang33", "yangzhou23", "ymbo2"
]

# Initialize Session State
if "run_pid" not in st.session_state:
    st.session_state.run_pid = None
if "run_logfile" not in st.session_state:
    st.session_state.run_logfile = None
if "running_model" not in st.session_state:
    st.session_state.running_model = None
if "show_tutorial" not in st.session_state:
    st.session_state.show_tutorial = False

# Custom CSS for better UI
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1E3A8A; /* Dark Blue */
        font-weight: 700;
    }
    h2 {
        color: #1F2937;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
        margin-top: 1rem;
    }
    h3 {
        color: #4B5563;
        font-size: 1.1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    .stSelectbox label {
        font-weight: 600;
        color: #374151;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F3F4F6;
        border-right: 1px solid #E5E7EB;
    }
    /* Card-like containers */
    .css-1r6slb0 {
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Exquisite Button Styling */
    .stButton > button {
        border-radius: 8px;
        height: auto;
        padding: 0.5em 1em;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    /* Primary Button (Start Training) - Gradient */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        box-shadow: 0 4px 6px -1px rgba(124, 58, 237, 0.3);
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px -1px rgba(124, 58, 237, 0.4);
    }
    
    /* Secondary Button & Download Button - Light Blue Style */
    .stButton > button[kind="secondary"], .stDownloadButton > button[kind="secondary"] {
        background-color: #EFF6FF;
        color: #2563EB;
        border: 1px solid #BFDBFE;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        width: 100%;
    }
    .stButton > button[kind="secondary"]:hover, .stDownloadButton > button[kind="secondary"]:hover {
        background-color: #DBEAFE;
        color: #1D4ED8;
        border-color: #93C5FD;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Custom Download Button */
    .custom-download-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.5em 1em;
        background-color: #EFF6FF;
        color: #2563EB;
        border-radius: 8px;
        text-decoration: none;
        font-size: 1rem;
        border: 1px solid #BFDBFE;
        font-weight: 600;
        transition: all 0.2s ease;
        margin-left: 0px;
        vertical-align: middle;
        line-height: 1.6;
    }
    .custom-download-btn:hover {
        background-color: #DBEAFE;
        color: #1D4ED8;
        border-color: #93C5FD;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-decoration: none;
    }
    
    /* Scrollable Code & JSON Container */
    .stCode pre, .stJson {
        max-height: 400px !important;
        overflow-y: auto !important;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_ZOO_DIR = os.path.join(ROOT_DIR, "model_zoo")
DATA_DIR = os.path.join(ROOT_DIR, "data")
LOG_DIR = os.path.join(ROOT_DIR, "dashboard", "logs")
TASK_STATE_DIR = os.path.join(ROOT_DIR, "dashboard", "state", "tasks")
HISTORY_DIR = os.path.join(ROOT_DIR, "dashboard", "state", "history")
USER_CONFIG_DIR = os.path.join(ROOT_DIR, "dashboard", "user_configs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TASK_STATE_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(USER_CONFIG_DIR, exist_ok=True)

# --- Task Management Helpers ---
def cleanup_stale_tasks():
    """Remove task files for processes that are no longer running."""
    if not os.path.exists(TASK_STATE_DIR):
        return
    for f in os.listdir(TASK_STATE_DIR):
        if not f.endswith(".json"): continue
        fpath = os.path.join(TASK_STATE_DIR, f)
        try:
            with open(fpath, 'r') as file:
                data = json.load(file)
            pid = data.get('pid')
            try:
                os.kill(pid, 0) # Check if process exists
            except OSError:
                os.remove(fpath) # Process dead
        except Exception:
            try:
                os.remove(fpath) # Corrupt file
            except:
                pass

def get_active_tasks():
    """Get list of all active tasks."""
    cleanup_stale_tasks()
    tasks = []
    if os.path.exists(TASK_STATE_DIR):
        for f in os.listdir(TASK_STATE_DIR):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(TASK_STATE_DIR, f), 'r') as file:
                        tasks.append(json.load(file))
                except:
                    pass
    return tasks

def save_task_state(username, pid, model, logfile):
    """Register a new task."""
    data = {
        "username": username,
        "pid": pid,
        "model": model,
        "logfile": logfile,
        "start_time": time.time()
    }
    # Filename includes username and pid to be unique
    fpath = os.path.join(TASK_STATE_DIR, f"{username}_{pid}.json")
    with open(fpath, 'w') as f:
        json.dump(data, f)

def remove_task_state(pid):
    """Unregister a task."""
    if not os.path.exists(TASK_STATE_DIR):
        return
    for f in os.listdir(TASK_STATE_DIR):
        if f"_{pid}.json" in f: # Match suffix to be safe
            try:
                os.remove(os.path.join(TASK_STATE_DIR, f))
            except:
                pass


def _sample_cpu_percent(proc, cache, warm_interval=0.1):
    """Return cpu_percent for a process, warming up if the first measurement was zero."""
    pid = proc.pid
    value = proc.cpu_percent(interval=0.0)
    if value == 0.0:
        now = time.time()
        last_probe = cache.get(pid, 0)
        if now - last_probe > warm_interval:
            value = proc.cpu_percent(interval=warm_interval)
            cache[pid] = now
    return value


def _aggregate_process_usage(pid):
    """Collect CPU% and memory for a pid including all child processes."""
    if "_cpu_probe_cache" not in st.session_state:
        st.session_state._cpu_probe_cache = {}
    cache = st.session_state._cpu_probe_cache
    try:
        root_proc = psutil.Process(pid)
    except psutil.Error:
        return None, None, None, None

    processes = [root_proc]
    try:
        processes.extend(root_proc.children(recursive=True))
    except psutil.Error:
        pass

    total_cpu = 0.0
    total_mem = 0
    pid_set = set()
    for proc in processes:
        try:
            total_cpu += _sample_cpu_percent(proc, cache)
            total_mem += proc.memory_info().rss
            pid_set.add(proc.pid)
        except psutil.Error:
            continue
    gpu_util, gpu_mem = _aggregate_gpu_usage(pid_set)
    return total_cpu, total_mem, gpu_util, gpu_mem


def _aggregate_gpu_usage(pids):
    """Return list of (util%, bytes) for each GPU touched by given pids.
    Returns (gpu_utils, gpu_mems) where each is a list of values per GPU."""
    try:
        import pynvml
    except ImportError:
        # pynvml not installed, return None
        return None, None
    except Exception:
        return None, None

    try:
        if not st.session_state.get("_nvml_initialized", False):
            pynvml.nvmlInit()
            st.session_state._nvml_initialized = True
    except pynvml.NVMLError:
        st.session_state._nvml_initialized = False
        return None, None
    except Exception:
        return None, None

    try:
        device_count = pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError:
        return None, None

    gpu_utils = []
    gpu_mems = []
    
    # Initialize lists for each GPU
    for idx in range(device_count):
        gpu_utils.append(0.0)
        gpu_mems.append(0)

    matched_any = False

    # First try to match specific processes
    if pids:
        proc_fn_names = [
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ]

        for idx in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            except pynvml.NVMLError:
                continue
            matched = False
            device_mem = 0
            proc_entries = []
            for fn_name in proc_fn_names:
                getter = getattr(pynvml, fn_name, None)
                if getter is None:
                    continue
                try:
                    proc_entries.extend(getter(handle))
                except pynvml.NVMLError:
                    continue
            for proc in proc_entries:
                proc_pid = getattr(proc, "pid", None)
                if proc_pid in pids:
                    matched = True
                    used = getattr(proc, "usedGpuMemory", 0) or 0
                    invalid = getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", None)
                    if invalid is not None and used == invalid:
                        used = 0
                    device_mem += max(0, used)
            if matched:
                matched_any = True
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                except pynvml.NVMLError:
                    util = 0.0
                gpu_utils[idx] = util
                gpu_mems[idx] = device_mem
    
    # If no specific process matched, return total GPU utilization for each device
    # This is useful for distributed training where child processes might not be detected
    if not matched_any:
        for idx in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_utils[idx] = util
                gpu_mems[idx] = mem_info.used
                matched_any = True
            except pynvml.NVMLError:
                continue
    
    if not matched_any:
        return None, None
    
    # Round to 1 decimal place for display
    gpu_utils = [round(util, 1) for util in gpu_utils]
    return gpu_utils, gpu_mems


def _guess_public_host():
    """Best-effort to pick a non-loopback IPv4 for external access."""
    candidates = []
    try:
        for if_name, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    ip = addr.address
                    if ip.startswith("127."):
                        continue
                    candidates.append(ip)
    except Exception:
        pass

    # Try socket trick to detect egress IP
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if not ip.startswith("127."):
                candidates.insert(0, ip)
    except Exception:
        pass

    for ip in candidates:
        if ip:
            return ip
    return None


def detect_tensorboard_process():
    """Detect a running TensorBoard process and return (url, pid, port).

    Strategy:
    1) Scan processes whose name/cmdline contains "tensorboard".
    2) Prefer an actual listening port from proc.connections(kind="inet").
    3) Fallback to parsing cmdline flags: --port / --host.
    """
    try:
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            info = proc.info or {}
            name = (info.get("name") or "").lower()
            cmdline = info.get("cmdline") or []
            joined_cmd = " ".join(cmdline).lower()

            if "tensorboard" not in name and "tensorboard" not in joined_cmd:
                continue

            host = None
            port = None

            try:
                for conn in proc.connections(kind="inet"):
                    if conn.status == psutil.CONN_LISTEN:
                        port = conn.laddr.port
                        host = conn.laddr.ip if conn.laddr and conn.laddr.ip not in ("0.0.0.0", None, "") else None
                        break
            except psutil.Error:
                pass

            if port is None:
                m = re.search(r"--port(?:=|\s)(\d+)", joined_cmd)
                if m:
                    port = m.group(1)
            if host is None:
                m = re.search(r"--host(?:=|\s)([^\s]+)", joined_cmd)
                if m:
                    host = m.group(1)
            # If still unknown/loopback, try to guess a non-loopback IP for remote access.
            if host in (None, "", "0.0.0.0", "localhost"):
                guessed = _guess_public_host()
                host = guessed or "localhost"
            try:
                port_int = int(port) if port is not None else None
            except Exception:
                port_int = None

            if port_int:
                url = f"http://{host}:{port_int}"
                return url, info.get("pid"), port_int
        return None, None, None
    except Exception:
        return None, None, None

# --- Run History Helpers ---
def _history_path(username):
    return os.path.join(HISTORY_DIR, f"{username}.json")

def load_history(username):
    path = _history_path(username)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_history(username, records):
    path = _history_path(username)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def append_history_record(username, record):
    history = load_history(username)
    history.insert(0, record)
    # Keep only the most recent 100 entries to avoid unbounded growth
    history = history[:100]
    save_history(username, history)

def _tail_text_file(path, max_bytes=20000, max_lines=200):
    """Read the tail of a text file for quick preview."""
    if not path or (not os.path.exists(path)):
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(size - max_bytes, 0))
            data = f.read().decode(errors="ignore").splitlines()
            if len(data) > max_lines:
                data = data[-max_lines:]
            return "\n".join(data)
    except Exception:
        return ""

def _read_text_file(path, max_bytes=None):
    """Read a text file with optional size cap; returns (text, truncated_flag)."""
    if not path or (not os.path.exists(path)):
        return "", False
    try:
        size = os.path.getsize(path)
        truncated = False
        if max_bytes is not None and size > max_bytes:
            truncated = True
            with open(path, "rb") as f:
                f.seek(max(size - max_bytes, 0))
                text = f.read().decode(errors="ignore")
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        return text, truncated
    except Exception:
        return "", False

def extract_latest_metrics(logfile):
    """Extract the latest [Metrics] line from a log file for compact display."""
    tail = _tail_text_file(logfile, max_bytes=20000, max_lines=400)
    if not tail:
        return "-"
    for line in reversed(tail.splitlines()):
        if "[Metrics]" in line:
            import re
            metric_str = line.split("[Metrics]", 1)[-1]
            # 只匹配包含字母的指标名，避免时间戳等纯数字键
            pairs = re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', metric_str)
            if pairs:
                return " / ".join(f"{k}={v}" for k, v in pairs[:3])
            return metric_str.strip()
    return "-"

def get_metrics_from_csv(expid, model_path):
    """从实验的CSV文件中读取最新的验证和测试指标"""
    csv_path = os.path.join(model_path, f"{expid}.csv")
    if not os.path.exists(csv_path):
        return "-", "-"
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                return "-", "-"
            # 取最后一行
            last_line = lines[-1].strip()
            # 解析格式：时间戳,[command] ...,[exp_id] ...,[dataset_id] ...,[train] ...,[val] ...,[test] ...
            parts = last_line.split(',')
            val_metrics = "-"
            test_metrics = "-"
            for part in parts:
                if part.startswith('[val] '):
                    val_metrics = part[6:]  # 去掉 '[val] '
                elif part.startswith('[test] '):
                    test_metrics = part[7:]  # 去掉 '[test] '
            return val_metrics, test_metrics
    except Exception:
        return "-", "-"

def update_history_record(username, pid, status, exit_code=None, message=None):
    history = load_history(username)
    updated = False
    current_time = time.time()
    for item in history:
        if item.get('pid') == pid and item.get('status') == 'running':
            item['end_time'] = current_time
            item['duration'] = current_time - item.get('start_time', current_time)
            item['status'] = status
            item['exit_code'] = exit_code
            if message:
                item['message'] = message
            if exit_code is not None:
                item['success'] = (exit_code == 0)
            elif status == 'stopped':
                item['success'] = False
            elif status == 'success':
                item['success'] = True
            elif status == 'failed':
                item['success'] = False
            
            # 如果任务完成（成功/失败/停止），尝试从CSV文件读取指标并存储到记录中
            if status in ['success', 'failed', 'stopped']:
                model = item.get('model')
                expid = item.get('expid')
                if model and expid:
                    model_path = os.path.join(MODEL_ZOO_DIR, model)
                    val_metrics, test_metrics = get_metrics_from_csv(expid, model_path)
                    if val_metrics != "-":
                        item['val_metrics'] = val_metrics
                    if test_metrics != "-":
                        item['test_metrics'] = test_metrics
            
            updated = True
            break
    if updated:
        save_history(username, history)

def delete_history_record(username, pid):
    """删除指定PID的历史记录"""
    history = load_history(username)
    # 过滤掉指定PID的记录
    new_history = [item for item in history if item.get('pid') != pid]
    if len(new_history) != len(history):
        save_history(username, new_history)
        return True
    return False

def delete_all_history(username):
    """删除用户的所有历史记录"""
    save_history(username, [])
    return True

def format_duration(seconds):
    if seconds is None:
        return "-"
    seconds = int(seconds)
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}h {mins}m"
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"

def get_subdirectories(directory):
    if not os.path.exists(directory):
        return []
    return sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and not d.startswith("__") and not d.startswith(".")])

def get_models(root_dir):
    models = []
    if not os.path.exists(root_dir):
        return []
    
    # First level
    for d in os.listdir(root_dir):
        path = os.path.join(root_dir, d)
        if os.path.isdir(path) and not d.startswith(".") and not d.startswith("__"):
            # Check if it is a model directory (has run_expid.py)
            if os.path.exists(os.path.join(path, "run_expid.py")):
                models.append(d)
            # Check if it is a container like 'multitask'
            else:
                # Check subdirectories
                for sub_d in os.listdir(path):
                    sub_path = os.path.join(path, sub_d)
                    if os.path.isdir(sub_path) and not sub_d.startswith(".") and not sub_d.startswith("__"):
                        if os.path.exists(os.path.join(sub_path, "run_expid.py")):
                            models.append(f"{d}/{sub_d}")
    return sorted(models)

def load_file_content(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def save_file_content(file_path, content):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def _get_buffered_content(buffer_key, file_path):
    """Prefer in-memory buffer over disk to survive reruns (e.g., Cmd/Ctrl+Enter)."""
    if buffer_key and buffer_key in st.session_state:
        return st.session_state[buffer_key]
    return load_file_content(file_path)


def _set_buffered_content(buffer_key, content):
    if buffer_key:
        st.session_state[buffer_key] = content


def _clear_buffer(buffer_key):
    if buffer_key and buffer_key in st.session_state:
        del st.session_state[buffer_key]


def _normalize_for_yaml(value):
    """Convert OrderedDict and other containers to YAML-friendly structures."""
    if isinstance(value, OrderedDict):
        return {k: _normalize_for_yaml(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _normalize_for_yaml(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_for_yaml(v) for v in value]
    return value


def _yaml_dump(data):
    """Dump YAML with consistent formatting."""
    normalized = _normalize_for_yaml(data)
    return yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True)


def _is_simple_yaml_value(value):
    """Decide whether a value fits compact single-line widgets."""
    if value is None:
        return True
    if isinstance(value, (bool, int, float)):
        return True
    if isinstance(value, str):
        return "\n" not in value and len(value) <= 120
    return False


def _convert_text_to_value(text, fallback, label):
    """Parse YAML scalar/structure from text; preserve fallback on error."""
    if text is None:
        return fallback
    stripped = text.strip()
    if stripped == "":
        if isinstance(fallback, str):
            return ""
        return fallback
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        st.warning(f"{label} 解析失败，已保留原值。错误信息: {exc}")
        return fallback


def _parse_name_list(text, fallback):
    """Parse feature name list from text area; accept YAML list or comma/newline separated."""
    if text is None:
        return fallback
    stripped = text.strip()
    if stripped == "":
        return []
    try:
        loaded = yaml.safe_load(text)
        if isinstance(loaded, list):
            return loaded
    except Exception:
        pass
    parts = []
    for line in stripped.splitlines():
        for chunk in line.split(','):
            val = chunk.strip()
            if val:
                parts.append(val)
    return parts if parts else fallback


def _parse_scalar_list(text, fallback):
    """Parse scalar list from a compact input."""
    if text is None:
        return fallback
    stripped = text.strip()
    if stripped == "":
        return [] if isinstance(fallback, list) else fallback
    try:
        loaded = yaml.safe_load(text)
        if isinstance(loaded, list):
            return loaded
    except Exception:
        pass
    items = []
    for line in stripped.splitlines():
        for part in line.split(','):
            val = part.strip()
            if val:
                items.append(_convert_text_to_value(val, val, "list_item"))
    return items if items else fallback


def _detect_simple_dict_list(value):
    """Return ordered keys when value is a list of flat dicts, else None."""
    if not isinstance(value, (list, tuple)) or not value:
        return None
    keys = []
    for item in value:
        if not isinstance(item, dict):
            return None
        for k, v in item.items():
            if not _is_simple_yaml_value(v):
                return None
            if k not in keys:
                keys.append(k)
    if len(keys) > 6:
        return None
    return keys


def _unique_keep_order(seq):
    seen = set()
    out = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _render_simple_dict_list(label, items, widget_key, keys):
    """Compact grid editor for list-of-dicts with simple scalar fields."""
    st.markdown(f"**{label}**")
    updated = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            updated.append(item)
            continue
        st.caption(f"#{idx + 1}")
        cols = st.columns(len(keys))
        new_item = OrderedDict()
        for col_idx, k in enumerate(keys):
            with cols[col_idx]:
                base = item.get(k)
                val_text = "" if base is None else str(base)
                input_val = st.text_input(
                    k,
                    val_text,
                    key=f"{widget_key}_{idx}_{k}",
                    label_visibility="visible" if len(keys) <= 3 else "collapsed",
                    placeholder=k
                )
                new_item[k] = _convert_text_to_value(input_val, base, f"{label}.{k}")
        updated.append(new_item)
    return updated


def _render_yaml_field(label, value, widget_key, *, is_fullscreen=False):
    """Render a Streamlit widget for common YAML value types."""
    if isinstance(value, bool):
        return st.checkbox(label, value=value, key=widget_key)

    if isinstance(value, (list, tuple, dict, OrderedDict)):
        dict_list_keys = _detect_simple_dict_list(value)
        if dict_list_keys:
            return _render_simple_dict_list(label, value, widget_key, dict_list_keys)

        # Compact edit for simple scalar lists
        if isinstance(value, (list, tuple)) and all(not isinstance(v, (list, dict, tuple, set)) for v in value):
            compact = ", ".join(map(str, value))
            text = st.text_input(label, compact, key=widget_key, placeholder="逗号或换行分隔")
            return _parse_scalar_list(text, list(value))

        serialized = _yaml_dump(value).strip()
        line_count = max(3, len(serialized.splitlines())) if serialized else 3
        base_height = 32 * min(10, line_count)
        height = 380 if is_fullscreen else max(120, min(260, base_height))
        text = st.text_area(label, serialized, key=widget_key, height=height)
        return _convert_text_to_value(text, value, label)

    display_text = "" if value is None else str(value)
    text = st.text_input(label, display_text, key=widget_key)
    return _convert_text_to_value(text, value, label)


def _render_fallback_editor(content, *, editor_key, lang, lines):
    """Use code_editor as fallback when structured rendering fails."""
    editor_options = {
        "wrap": False,
        "showGutter": True,
        "autoScrollEditorIntoView": True,
        "minLines": lines,
        "maxLines": lines,
        "scrollBeyondLastLine": False
    }
    res = code_editor(
        content,
        lang=lang,
        key=editor_key,
        response_mode="debounce",
        options=editor_options,
        buttons=[{
            "name": "Copy",
            "feather": "Copy",
            "hasText": True,
            "alwaysOn": True,
            "commands": ["copyAll"],
            "style": {"top": "0.46rem", "right": "0.4rem"}
        }]
    )
    if isinstance(res, dict):
        if res.get("text") == "" and res.get("type", "") == "":
            return content
        return res.get("text", content)
    return content


def _render_label_col_section(entry, editor_key, selected_name):
    """可视化编辑 label_col，支持单目标(dict)和多目标(list)及新增"""

    # Session state key for pending deletions
    pending_del_key = f"{editor_key}_{selected_name}_pending_deletions"
    if pending_del_key not in st.session_state:
        st.session_state[pending_del_key] = []

    label_col = entry.get("label_col")

    # 策略：如果是 dict，视为 items=[dict]；如果是 list，视为 items=list
    # 保存时：如果原先是 dict 且 count=1，存回 dict；否则存 list
    is_single_structure = isinstance(label_col, dict)
    items = []
    if is_single_structure:
        items = [label_col]
    elif isinstance(label_col, list):
        items = list(label_col)

    # 扫描所有 keys
    all_keys = ["name", "dtype"]
    for item in items:
        if isinstance(item, dict):
            for k in item.keys():
                if k not in all_keys:
                    all_keys.append(k)

    st.markdown("##### label_col")

    new_items = []
    deleted_indices = st.session_state[pending_del_key].copy()  # 从session state初始化
    
    # 渲染列表
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            new_items.append(item)
            continue
            
        cols = st.columns(len(all_keys) + 1, vertical_alignment="bottom")
        updated_item = OrderedDict(item)
        
        for i, k in enumerate(all_keys):
            with cols[i]:
                val = item.get(k)
                val_text = "" if val is None else str(val)
                new_val = st.text_input(
                    k, 
                    val_text, 
                    key=f"{editor_key}_{selected_name}_label_{idx}_{k}",
                    label_visibility="visible" if idx == 0 else "collapsed",
                    placeholder=k
                )
                updated_item[k] = _convert_text_to_value(new_val, val, f"label_col.{k}")
        
        with cols[-1]:
            # Delete button with flat modern style
            st.markdown("""
                <style>
                div[data-testid="stColumn"]:has(div[class="del_marker"]) button {
                    border: none !important;
                    background: transparent !important;
                    color: #9ca3af !important;
                    font-weight: bold !important;
                }
                div[data-testid="stColumn"]:has(div[class="del_marker"]) button:hover {
                    background: #fee2e2 !important;
                    color: #ef4444 !important;
                    border: none !important;
                }
                </style>
                <div class='del_marker' style='display:none'></div>
                """, unsafe_allow_html=True)
            if st.button("✕", key=f"{editor_key}_{selected_name}_label_del_{idx}", help="删除此任务"):
                st.session_state[pending_del_key].append(idx)
                st.rerun()
        
        new_items.append(updated_item)
    
    # 过滤被删除项
    final_items = [item for i, item in enumerate(new_items) if i not in deleted_indices]

    # 添加按钮
    if st.button("➕ 添加任务", key=f"{editor_key}_{selected_name}_label_add", help="添加新任务/标签"):
        final_items.append(OrderedDict({"name": "new_label", "dtype": "float"}))
        st.rerun()
    
    # 更新 entry
    if not final_items:
         # 如果清空了，是否移除 key?
         entry["label_col"] = []
    elif is_single_structure and len(final_items) == 1:
         entry["label_col"] = final_items[0]
    else:
         entry["label_col"] = final_items

    # 清空 pending deletions
    st.session_state[pending_del_key] = []


def render_dataset_config_body(
    content,
    *,
    editor_key,
    lang,
    lines,
    is_fullscreen=False,
    selected_model=None,
    buffer_key=None
):
    """Render dataset_config.yaml as structured form while hiding feature_cols."""

    def _list_data_files(data_root_val):
        if not data_root_val:
            return []
        candidates = [data_root_val]
        if selected_model:
            candidates.append(os.path.join(MODEL_ZOO_DIR, selected_model, data_root_val))
        candidates.append(os.path.join(ROOT_DIR, data_root_val))

        base_dir = None
        for cand in candidates:
            abs_cand = os.path.abspath(cand)
            if os.path.isdir(abs_cand):
                base_dir = abs_cand
                break
        if not base_dir:
            return []
        files = []
        try:
            for f in os.listdir(base_dir):
                if f.startswith('.'):
                    continue
                full = os.path.join(base_dir, f)
                rel_path = os.path.normpath(os.path.join(data_root_val, f))
                # 支持目录（常见是 parquet 目录）与文件（常见是 .json 配置）
                if os.path.isdir(full) or os.path.isfile(full):
                    files.append(rel_path)
        except Exception:
            return []
        # keep order, unique
        seen = set()
        out = []
        for f in files:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return sorted(out)

    def _auto_update_feature_cols_from_parquet(train_path, data_root=None):
        if not train_path:
            st.warning("请先填写 train_data 路径后再更新特征。")
            return None

        candidates = []
        # Raw path
        candidates.append(train_path)
        # data_root relative
        if data_root:
            candidates.append(os.path.join(data_root, train_path))
        # model directory relative
        if selected_model:
            model_dir = os.path.join(MODEL_ZOO_DIR, selected_model)
            candidates.append(os.path.join(model_dir, train_path))
        # project root
        candidates.append(os.path.join(ROOT_DIR, train_path))

        target = None
        chosen_dir = None
        for cand in candidates:
            if not cand:
                continue
            abs_cand = os.path.abspath(cand)
            if os.path.isfile(abs_cand) and abs_cand.endswith('.parquet'):
                target = abs_cand
                break
            if os.path.isdir(abs_cand):
                chosen_dir = abs_cand
                parquet_files = sorted([f for f in os.listdir(abs_cand) if f.endswith('.parquet')])
                if parquet_files:
                    target = os.path.join(abs_cand, parquet_files[0])
                    break

        if target is None:
            if chosen_dir:
                st.error(f"在目录 {chosen_dir} 下未找到 parquet 文件。")
            else:
                st.error("train_data 路径无效，未找到 parquet 文件（支持目录或 .parquet 文件，相对路径自动尝试 data_root/模型目录/项目根）。")
            return None

        try:
            df = pd.read_parquet(target)
        except Exception as exc:
            st.error(f"读取 parquet 失败：{exc}")
            return None

        cols = list(df.columns)
        cat = [c for c in cols if c.endswith("_tag")]
        if "product" in cols:
            cat.append("product")
        num = [c for c in cols if c.endswith("_cnt")]
        seq = [c for c in cols if c.endswith("_textlist")]

        specials = ["appInstalls", "outerBizSorted", "outerModelCleanSorted"]
        for s in specials:
            if s in cols:
                seq.append(s)

        cat = _unique_keep_order(cat)
        num = _unique_keep_order(num)
        seq = _unique_keep_order(seq)

        new_feature_cols = []
        if cat:
            new_feature_cols.append(OrderedDict({"name": cat, "type": "categorical", "dtype": "int", "active": True}))
        if num:
            new_feature_cols.append(OrderedDict({
                "name": num, 
                "type": "numeric", 
                "dtype": "float", 
                "active": True, 
                "normalizer": "StandardScaler"
            }))
        if seq:
            new_feature_cols.append(OrderedDict({
                "name": seq, 
                "type": "sequence", 
                "dtype": "str", 
                "active": True, 
                "max_len": 15, 
                "encoder": "MaskedAveragePooling"
            }))

        if not new_feature_cols:
            st.warning("未识别到符合命名规则的特征列，未更新 feature_cols。")
            return None
        return new_feature_cols

    try:
        raw_data = yaml.safe_load(content) or {}
        if not isinstance(raw_data, dict):
            raise ValueError("dataset_config.yaml 顶层需为映射")
    except Exception as exc:
        st.error(f"无法解析 dataset_config.yaml：{exc}")
        st.info("已回退到原始编辑器，请修复后再试。")
        return _render_fallback_editor(
            content,
            editor_key=f"{editor_key}_raw",
            lang=lang,
            lines=lines
        )

    data = OrderedDict(raw_data)
    dataset_names = list(data.keys())
    if not dataset_names:
        st.info("当前没有可编辑的数据集配置，将保留原内容。")
        return content

    select_key = f"{editor_key}_select"
    current_sel = st.session_state.get(select_key)
    if current_sel and current_sel not in dataset_names:
        del st.session_state[select_key]
    copy_mode_key = f"{select_key}_copy_mode"
    copy_input_key = f"{select_key}_copy_input"
    copy_clear_key = f"{select_key}_copy_clear"
    if copy_mode_key not in st.session_state:
        st.session_state[copy_mode_key] = False
    if st.session_state.get(copy_clear_key):
        st.session_state.pop(copy_input_key, None)
        st.session_state[copy_clear_key] = False

    # Copy Handler Logic (Must run before selectbox is rendered)
    copy_action_key = f"{select_key}_copy_action"
    if st.session_state.get(copy_action_key):
        st.session_state[copy_action_key] = False  # Consume trigger
        src_id = st.session_state.get(select_key)
        tgt_id = st.session_state.get(copy_input_key, "").strip()

        if not src_id:
            st.warning("复制操作已跳过：未选中源配置。")
        elif not tgt_id:
            st.warning("复制操作已跳过：新名称不能为空。")
        elif tgt_id in data:
            st.warning(f"复制操作已跳过：名称 '{tgt_id}' 已存在。")
        else:
            data[tgt_id] = copy.deepcopy(data.get(src_id) or {})
            dataset_names.append(tgt_id)
            st.session_state[select_key] = tgt_id
            if buffer_key:
                _set_buffered_content(buffer_key, _yaml_dump(data))
            st.toast(f"已复制为 {tgt_id}（记得保存）", icon="✅")
            st.session_state[copy_mode_key] = False
            st.session_state[copy_clear_key] = True
            st.rerun()

    sel_cols = st.columns([0.94, 0.06], vertical_alignment="bottom")
    with sel_cols[0]:
        selected_name = st.selectbox(
            "选择数据集配置 (dataset_id)",
            dataset_names,
            key=select_key,
            help="选择需要编辑的数据集条目"
        )
    with sel_cols[1]:
        st.markdown("""
            <style>
            div[data-testid="stColumn"]:has(div[class="copy_marker"]) button {
                border: 1px solid transparent !important;
                background-color: transparent !important;
                color: #666 !important;
                padding: 0rem 0.5rem !important;
            }
            div[data-testid="stColumn"]:has(div[class="copy_marker"]) button:hover {
                background-color: #f3f4f6 !important;
                color: #000 !important;
                border: 1px solid #e5e7eb !important;
            }
            </style>
            <div class='copy_marker' style='display:none'></div>
            """, unsafe_allow_html=True)
        if st.button("✚", key=f"{select_key}_copy_btn", help="克隆当前数据集配置为新名称", type="secondary"):
            st.session_state[copy_mode_key] = True
            st.rerun()

    new_ds_name = st.session_state.get(copy_input_key, "")
    if st.session_state[copy_mode_key]:
        with st.container(border=True):
            new_ds_name = st.text_input("新 dataset_id", key=copy_input_key, placeholder="输入新名称")
            btn_cols = st.columns([1, 1])

            def _on_confirm_copy():
                st.session_state[copy_action_key] = True

            with btn_cols[0]:
                if st.button("确定复制", key=f"{select_key}_confirm_copy", on_click=_on_confirm_copy):
                    st.rerun()
            with btn_cols[1]:
                if st.button("取消", key=f"{select_key}_cancel_copy"):
                    st.session_state[copy_mode_key] = False
                    st.session_state[copy_clear_key] = True
                    st.rerun()

    if not selected_name:
        return content

    entry_raw = data.get(selected_name) or {}
    if not isinstance(entry_raw, dict):
        st.warning("选中的数据集配置格式异常，已回退到原始编辑器。")
        return _render_fallback_editor(
            content,
            editor_key=f"{editor_key}_raw",
            lang=lang,
            lines=lines
        )

    entry = OrderedDict(entry_raw)

    visible_fields = [k for k in entry.keys() if k not in ["feature_cols", "label_col"]]
    simple_fields = [f for f in visible_fields if _is_simple_yaml_value(entry.get(f))]
    complex_fields = [f for f in visible_fields if f not in simple_fields]

    for idx in range(0, len(simple_fields), 2):
        cols = st.columns(2)
        for offset in range(2):
            if idx + offset >= len(simple_fields):
                break
            field = simple_fields[idx + offset]
            widget_key = f"{editor_key}_{selected_name}_{field}"
            with cols[offset]:
                data_root_val = entry.get("data_root")
                file_options = _list_data_files(data_root_val)

                if field in ["train_data", "valid_data", "test_data", "infer_data"] and data_root_val:
                    current_val = entry.get(field) or ""
                    options = list(file_options)
                    if current_val and current_val not in options:
                        options.insert(0, current_val)
                    display_label = field
                    entry[field] = st.selectbox(
                        display_label,
                        options,
                        key=widget_key,
                        index=options.index(current_val) if current_val in options else 0 if options else None,
                        placeholder="选择文件"
                    ) if options else st.text_input(display_label, current_val, key=widget_key)
                    if field == "train_data":
                        if st.button("⚡ 更新特征", key=f"{widget_key}_update_feature_cols", help="读取 train_data 下首个 parquet 列并覆盖 feature_cols"):
                            generated = _auto_update_feature_cols_from_parquet(entry[field], data_root_val)
                            if generated is not None:
                                entry["feature_cols"] = generated
                                # 立即写入 buffer，切换模型返回时仍能看到更新
                                if buffer_key:
                                    updated_data = OrderedDict(data)
                                    updated_data[selected_name] = entry
                                    _set_buffered_content(buffer_key, _yaml_dump(updated_data))
                                st.success("已根据 parquet 列生成并覆盖 feature_cols（记得保存）")
                                st.rerun()
                elif field == "train_data":
                    td_val = "" if entry.get(field) is None else str(entry.get(field))
                    entry[field] = st.text_input(field, td_val, key=widget_key, placeholder="train_data parquet 路径")
                    if st.button("⚡ 更新特征", key=f"{widget_key}_update_feature_cols", help="读取 train_data 下首个 parquet 列并覆盖 feature_cols"):
                        generated = _auto_update_feature_cols_from_parquet(entry[field], entry.get("data_root"))
                        if generated is not None:
                            entry["feature_cols"] = generated
                            if buffer_key:
                                updated_data = OrderedDict(data)
                                updated_data[selected_name] = entry
                                _set_buffered_content(buffer_key, _yaml_dump(updated_data))
                            st.success("已根据 parquet 列生成并覆盖 feature_cols（记得保存）")
                            st.rerun()
                else:
                    entry[field] = _render_yaml_field(field, entry.get(field), widget_key, is_fullscreen=is_fullscreen)

    # Label Col Editor
    if "label_col" in entry:
        _render_label_col_section(entry, editor_key, selected_name)

    feature_cols = entry.get("feature_cols")
    if feature_cols is not None:
        group_count = len(feature_cols) if isinstance(feature_cols, (list, tuple)) else 1
        feature_count = 0
        if isinstance(feature_cols, (list, tuple)):
            for group in feature_cols:
                if not isinstance(group, dict):
                    continue
                names = group.get("name")
                cnt = len(names) if isinstance(names, list) else (1 if names is not None else 0)
                feature_count += cnt
        st.caption(f"feature_cols 已隐藏，将在保存时保留原始 {group_count} 组，共约 {feature_count} 项。")

        new_feature_cols = []
        with st.expander("🎛️ feature_cols 可视化编辑", expanded=False):
            type_label = {
                "categorical": "类别型",
                "numeric": "数值型",
                "sequence": "序列型"
            }

            if isinstance(feature_cols, (list, tuple)):
                last_type = None
                for idx_fc, group in enumerate(feature_cols):
                    if not isinstance(group, dict):
                        new_feature_cols.append(group)
                        continue

                    g_type_val = str(group.get("type", "")).lower()
                    if last_type is not None and g_type_val != last_type:
                        st.markdown('<hr style="border: 1px solid #e5e7eb;" />', unsafe_allow_html=True)
                    last_type = g_type_val

                    header = type_label.get(g_type_val, g_type_val or "特征组")
                    names_val = group.get("name")
                    options = names_val if isinstance(names_val, list) else ([] if names_val is None else [str(names_val)])
                    name_count = len(options)
                    active_badge = "#16a34a" if bool(group.get("active", True)) else "#9ca3af"
                    st.markdown(
                        """
                        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:4px 0 8px 0;">
                            <div style="padding:4px 8px;border-radius:999px;background:#eef2ff;color:#4338ca;font-weight:700;font-size:12px;">{header}</div>
                            <div style="padding:4px 8px;border-radius:999px;background:#f1f5f9;color:#475569;font-weight:600;font-size:12px;">dtype: {dtype}</div>
                            <div style="padding:4px 8px;border-radius:999px;background:#ecfeff;color:#0f766e;font-weight:600;font-size:12px;">name: {count}</div>
                            <div style="padding:4px 8px;border-radius:999px;background:{active_bg};color:white;font-weight:700;font-size:12px;">active</div>
                        </div>
                        """.format(
                            header=header or "特征组",
                            dtype=str(group.get("dtype", "")) or "—",
                            count=name_count,
                            active_bg=active_badge
                        ),
                        unsafe_allow_html=True
                    )

                    col_a, col_b, col_c = st.columns([1, 1, 1])
                    with col_a:
                        g_type = st.text_input(
                            "type",
                            value=str(group.get("type", "")),
                            key=f"{editor_key}_{selected_name}_type_{idx_fc}"
                        )
                    with col_b:
                        g_dtype = st.text_input(
                            "dtype",
                            value=str(group.get("dtype", "")),
                            key=f"{editor_key}_{selected_name}_dtype_{idx_fc}"
                        )
                    with col_c:
                        g_active = st.checkbox(
                            "active",
                            value=bool(group.get("active", True)),
                            key=f"{editor_key}_{selected_name}_active_{idx_fc}"
                        )

                    names_val = group.get("name")
                    options = names_val if isinstance(names_val, list) else ([] if names_val is None else [str(names_val)])
                    show_names = st.checkbox(
                        f"name 列表（{name_count} 项）",
                        value=False,
                        key=f"{editor_key}_{selected_name}_names_toggle_{idx_fc}"
                    )
                    if show_names:
                        parsed_names = st.multiselect(
                            "name 列表",
                            options,
                            default=options,
                            key=f"{editor_key}_{selected_name}_names_{idx_fc}",
                            placeholder="点击展开选择/删减"
                        )
                    else:
                        parsed_names = options

                    other_keys = [k for k in group.keys() if k not in ["name", "type", "dtype", "active"]]
                    extra_updates = {}
                    if other_keys:
                        for ok in other_keys:
                            val = group.get(ok)
                            val_text = "" if val is None else str(val)
                            val_text = st.text_input(
                                ok,
                                value=val_text,
                                key=f"{editor_key}_{selected_name}_extra_{ok}_{idx_fc}"
                            )
                            extra_updates[ok] = _convert_text_to_value(val_text, val, ok)

                    new_group = OrderedDict(group)
                    new_group["type"] = g_type
                    new_group["dtype"] = g_dtype
                    new_group["active"] = g_active
                    new_group["name"] = parsed_names
                    for ok, ov in extra_updates.items():
                        new_group[ok] = ov

                    new_feature_cols.append(new_group)

            entry["feature_cols"] = new_feature_cols if new_feature_cols else feature_cols

    for field in complex_fields:
        widget_key = f"{editor_key}_{selected_name}_{field}"
        entry[field] = _render_yaml_field(field, entry.get(field), widget_key, is_fullscreen=is_fullscreen)

    data[selected_name] = entry

    return _yaml_dump(data)


def _render_loss_section(label, value, widget_key):
    """可视化的 Loss 编辑器，兼容混合类型（String/Dict）和多任务"""

    # Session state keys for pending additions and deletions
    pending_add_key = f"{widget_key}_pending_add"
    pending_del_key = f"{widget_key}_pending_del"
    if pending_add_key not in st.session_state:
        st.session_state[pending_add_key] = []
    if pending_del_key not in st.session_state:
        st.session_state[pending_del_key] = []

    # 1. 统一正规化为 List 进行编辑
    # is_single_structure 标记最初是否是非列表形式（String 或 单Dict），
    # 用于在保存时如果只有一项，尽可能还原回简洁形式。
    is_single_structure = False
    items = []

    if isinstance(value, (dict, OrderedDict)) and "name" in value:
        is_single_structure = True
        items = [value]
    elif isinstance(value, str):
        is_single_structure = True
        items = [value]
    elif isinstance(value, list) and value:
        # 只要列表里是 String 或 Dict 就认为是合法 Loss 列表
        if all(isinstance(x, (str, dict, OrderedDict)) for x in value):
            items = value
        else:
            return _render_yaml_field(label, value, widget_key)
    elif value is None:
        items = []
        is_single_structure = True # 视为原本为空的单项
    else:
        # 其他情况回退
        return _render_yaml_field(label, value, widget_key)

    # 合并 pending additions 到 items 中
    pending_items = st.session_state[pending_add_key]
    if pending_items:
        items.extend(pending_items)
        st.session_state[pending_add_key] = []  # 清空 pending 项

    st.markdown(f"##### {label}")

    new_items = []
    
    # Header
    cols_header = st.columns([0.3, 0.6, 0.1])
    cols_header[0].caption("Loss Name")
    cols_header[1].caption("Params (YAML/JSON)")
    # cols_header[2].caption("Del")

    # 获取 pending deletions
    pending_deletions = st.session_state[pending_del_key]

    for idx, item in enumerate(items):
        # 跳过待删除项
        if idx in pending_deletions:
            continue

        cols = st.columns([0.3, 0.6, 0.1], vertical_alignment="center")
        
        # 解析当前项：支持 String 或 Dict 混合
        if isinstance(item, str):
            current_name = item
            current_params = None
        elif isinstance(item, (dict, OrderedDict)):
            current_name = item.get("name", "")
            current_params = item.get("params", None)
        else:
             current_name = ""
             current_params = None
        
        # Name field
        new_name = cols[0].text_input("Name", current_name, label_visibility="collapsed", key=f"{widget_key}_{idx}_name", placeholder="e.g. binary_crossentropy")
        
        # Params field
        params_str = _yaml_dump(current_params).strip() if current_params else ""
        new_params_str = cols[1].text_area("Params", params_str, height=68, label_visibility="collapsed", key=f"{widget_key}_{idx}_params", placeholder="参数 (可选)")
        
        # Delete button
        if cols[2].button("✕", key=f"{widget_key}_{idx}_del"):
            st.session_state[pending_del_key].append(idx)
            st.rerun()
            
        # Reconstruct item
        parsed_params = None
        if new_params_str.strip():
            try:
                 parsed_params = _convert_text_to_value(new_params_str, {}, "params")
            except:
                 parsed_params = current_params

        # 智能保存：如果没有参数，存为 String；否则存为 Dict
        if not parsed_params:
            if new_name: # 忽略空名
                new_item = new_name 
            else:
                continue # 忽略空行
        else:
            new_item = OrderedDict()
            new_item["name"] = new_name
            new_item["params"] = parsed_params
        
        new_items.append(new_item)

    # 底部添加按钮栏
    add_cols = st.columns([0.3, 0.35, 0.35])
    if add_cols[0].button("✚ BCE", key=f"{widget_key}_add_bce", help="添加 binary_crossentropy", use_container_width=True):
        st.session_state[pending_add_key].append("binary_crossentropy")
        st.rerun()
    if add_cols[1].button("✚ FocalLoss", key=f"{widget_key}_add_focal", help="添加 FocalLoss", use_container_width=True):
        st.session_state[pending_add_key].append(OrderedDict([("name", "FocalLoss"), ("params", OrderedDict([("gamma", 2.0), ("alpha", 0.25)]))]))
        st.rerun()
    
    # 还原结构逻辑
    if is_single_structure:
        # 如果原本是单项，且现在还是 1 项 -> 还原为单项（String 或 Dict）
        # 这样保持配置文件的简洁性
        if len(new_items) == 1:
            # 清空 pending deletions
            st.session_state[pending_del_key] = []
            return new_items[0]
        elif len(new_items) == 0:
            # 清空 pending deletions
            st.session_state[pending_del_key] = []
            return None

    # 清空 pending deletions
    st.session_state[pending_del_key] = []
    return new_items


def render_model_config_body(
    content,
    *,
    editor_key,
    lang,
    lines,
    is_fullscreen=False,
    selected_model=None,
    buffer_key=None,
    **_
):
    """Render model_config.yaml as structured form with section selector."""
    try:
        raw_data = yaml.safe_load(content) or {}
        if not isinstance(raw_data, dict):
            raise ValueError("model_config.yaml 顶层需为映射")
    except Exception as exc:
        st.error(f"无法解析 model_config.yaml：{exc}")
        st.info("已回退到原始编辑器，请修复后再试。")
        return _render_fallback_editor(
            content,
            editor_key=f"{editor_key}_raw",
            lang=lang,
            lines=lines
        )

    data = OrderedDict(raw_data)
    config_names = list(data.keys())
    if not config_names:
        st.info("当前没有可编辑的模型配置，将保留原内容。")
        return content

    select_key = f"{editor_key}_select"
    current_sel = st.session_state.get(select_key)
    if current_sel and current_sel not in config_names:
        del st.session_state[select_key]
    default_idx = 0
    for i, name in enumerate(config_names):
        if name.lower() != "base":
            default_idx = i
            break
    copy_mode_key = f"{select_key}_copy_mode"
    copy_input_key = f"{select_key}_copy_input"
    copy_clear_key = f"{select_key}_copy_clear"
    if copy_mode_key not in st.session_state:
        st.session_state[copy_mode_key] = False
    if st.session_state.get(copy_clear_key):
        st.session_state.pop(copy_input_key, None)
        st.session_state[copy_clear_key] = False

    # Copy Handler Logic (Must run before selectbox is rendered)
    copy_action_key = f"{select_key}_copy_action"
    if st.session_state.get(copy_action_key):
        st.session_state[copy_action_key] = False  # Consume trigger
        src_id = st.session_state.get(select_key)
        tgt_id = st.session_state.get(copy_input_key, "").strip()

        if not src_id:
            st.warning("复制操作已跳过：未选中源配置。")
        elif not tgt_id:
             st.warning("复制操作已跳过：新名称不能为空。")
        elif tgt_id in data:
             st.warning(f"复制操作已跳过：名称 '{tgt_id}' 已存在。")
        else:
            data[tgt_id] = copy.deepcopy(data.get(src_id) or {})
            config_names.append(tgt_id)
            st.session_state[select_key] = tgt_id
            if buffer_key:
                _set_buffered_content(buffer_key, _yaml_dump(data))
            st.toast(f"已复制为 {tgt_id}（记得保存）", icon="✅")
            st.session_state[copy_mode_key] = False
            st.session_state[copy_clear_key] = True
            st.rerun()

    sel_cols = st.columns([0.94, 0.06], vertical_alignment="bottom")
    with sel_cols[0]:
        selected_name = st.selectbox(
            "选择模型配置(experiment_id)",
            config_names,
            index=default_idx if config_names else None,
            key=select_key,
            help="选择需要编辑的配置节"
        )
    with sel_cols[1]:
        st.markdown("""
            <style>
            div[data-testid="stColumn"]:has(div[class="copy_marker"]) button {
                border: 1px solid transparent !important;
                background-color: transparent !important;
                color: #666 !important;
                padding: 0rem 0.5rem !important;
            }
            div[data-testid="stColumn"]:has(div[class="copy_marker"]) button:hover {
                background-color: #f3f4f6 !important;
                color: #000 !important;
                border: 1px solid #e5e7eb !important;
            }
            </style>
            <div class='copy_marker' style='display:none'></div>
            """, unsafe_allow_html=True)
        if st.button("✚", key=f"{select_key}_copy_btn", help="克隆当前模型配置为新名称", type="secondary"):
            st.session_state[copy_mode_key] = True
            st.rerun()

    new_model_name = st.session_state.get(copy_input_key, "")
    if st.session_state[copy_mode_key]:
        with st.container(border=True):
            new_model_name = st.text_input("新配置名", key=copy_input_key, placeholder="输入新名称")
            btn_cols = st.columns([1, 1])
            
            def _on_confirm_copy_model():
                st.session_state[copy_action_key] = True

            with btn_cols[0]:
                if st.button("确定复制", key=f"{select_key}_confirm_copy", on_click=_on_confirm_copy_model):
                    st.rerun()
            with btn_cols[1]:
                if st.button("取消", key=f"{select_key}_cancel_copy"):
                    st.session_state[copy_mode_key] = False
                    st.session_state[copy_clear_key] = True
                    st.rerun()

    if not selected_name:
        return content

    entry_raw = data.get(selected_name) or {}
    if not isinstance(entry_raw, dict):
        st.warning("选中的模型配置非映射结构，已回退到原始编辑器。")
        return _render_fallback_editor(
            content,
            editor_key=f"{editor_key}_raw",
            lang=lang,
            lines=lines
        )

    entry = OrderedDict(entry_raw)

    st.caption("支持直接修改各字段，复杂结构以 YAML 文本形式编辑。")

    fields = list(entry.keys())
    simple_fields = [f for f in fields if _is_simple_yaml_value(entry.get(f))]
    complex_fields = [f for f in fields if f not in simple_fields]

    for idx in range(0, len(simple_fields), 2):
        cols = st.columns(2)
        for offset in range(2):
            if idx + offset >= len(simple_fields):
                break
            field = simple_fields[idx + offset]
            widget_key = f"{editor_key}_{selected_name}_{field}"
            with cols[offset]:
                entry[field] = _render_yaml_field(field, entry.get(field), widget_key, is_fullscreen=is_fullscreen)

    for field in complex_fields:
        widget_key = f"{editor_key}_{selected_name}_{field}"
        if field == "loss":
            entry[field] = _render_loss_section(field, entry.get(field), widget_key)
        else:
            entry[field] = _render_yaml_field(field, entry.get(field), widget_key, is_fullscreen=is_fullscreen)

    data[selected_name] = entry

    return _yaml_dump(data)

def get_config_paths(model_name, username):
    """
    Return the effective paths for config files and scripts.
    Priority: User Config > Default Model Config
    """
    model_dir = os.path.join(MODEL_ZOO_DIR, model_name)
    default_config_dir = os.path.join(model_dir, "config")
    user_config_dir = os.path.join(USER_CONFIG_DIR, username, model_name)
    
    # Define files and their default locations
    file_specs = {
        "dataset_config.yaml": os.path.join(default_config_dir, "dataset_config.yaml"),
        "model_config.yaml": os.path.join(default_config_dir, "model_config.yaml"),
        "run_expid.py": os.path.join(model_dir, "run_expid.py")
    }
    
    paths = {}
    
    for filename, default_path in file_specs.items():
        user_path = os.path.join(user_config_dir, filename)
        
        if os.path.exists(user_path):
            paths[filename] = {"path": user_path, "type": "custom", "default_path": default_path}
        else:
            paths[filename] = {"path": default_path, "type": "default", "default_path": default_path}
            
    return paths, user_config_dir

def reset_user_config(username, model_name, filename):
    """Delete user custom config to revert to default."""
    user_path = os.path.join(USER_CONFIG_DIR, username, model_name, filename)
    if os.path.exists(user_path):
        os.remove(user_path)

def get_download_link(content, filename, label):
    """Generate a styled download link."""
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" class="custom-download-btn">⬇️ {label}</a>'

# --- Tutorial Page ---
def render_tutorial():
    st.title("📚 XFDL 平台使用指南")
    st.markdown("---")
    if st.button("🔙 返回主页"):
        st.session_state.show_tutorial = False
        st.rerun()
    st.markdown("")
    guide_path = os.path.join(ROOT_DIR, "dashboard", "CONFIG_GUIDE.md")
    try:
        with open(guide_path, "r", encoding="utf-8") as f:
            guide_md = f.read()
        st.markdown(guide_md)
    except Exception as e:
        st.error(f"无法读取配置指南：{e}")
    st.markdown("---")
    if st.button("返回主界面"):
        st.session_state.show_tutorial = False
        st.rerun()
    st.stop() # Stop execution here to show only tutorial

# Header
if st.session_state.show_tutorial:
    render_tutorial()

col_main, col_help = st.columns([6, 1])
with col_main:
    st.title("XFDL 实验平台")
with col_help:
    st.write("")
    if st.button("📘 使用教程"):
        st.session_state.show_tutorial = True
        st.rerun()

st.markdown("专业的 CTR 模型训练与推理平台")

st.markdown("---")

# Sidebar for Selection
with st.sidebar:
    st.header("🎛️ 项目设置")
    
    # User Identity for Task Management
    st.markdown("### 👤 用户身份")
    
    # Initialize previous user reference for change detection
    if "prev_user" not in st.session_state:
        st.session_state.prev_user = "admin"

    # Initialize selected_model in session_state
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    # Ensure prev_user is in options
    default_index = 0
    if st.session_state.prev_user in USER_OPTIONS:
        default_index = USER_OPTIONS.index(st.session_state.prev_user)

    # Use key to maintain state and avoid issues
    current_user = st.selectbox("用户名", USER_OPTIONS, index=default_index, key="user_selector", help="用于任务限制（每位用户最多 3 个任务）。")

    # Detect User Switch
    if current_user != st.session_state.prev_user:
        st.session_state.prev_user = current_user
        # Clear session state to prevent leaking previous user's task info
        st.session_state.run_pid = None
        st.session_state.run_logfile = None
        st.session_state.running_model = None
        st.session_state.selected_model = None  # Also clear selected model
        st.rerun()

    if not current_user:
        st.warning("请输入用户名。")

    st.markdown("### 📍 模型选择")
    models = get_models(MODEL_ZOO_DIR)
    # Get default index for model selector
    default_model_index = 0
    if st.session_state.selected_model and st.session_state.selected_model in models:
        default_model_index = models.index(st.session_state.selected_model)

    # Use key to maintain state and avoid double-click issues
    selected_model = st.selectbox("选择模型", models, label_visibility="collapsed", index=default_model_index, key="model_selector")
    # Update session_state when model changes
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model

    if selected_model:
        st.caption(f"路径：`model_zoo/{selected_model}`")

    st.markdown("### 💾 数据配置")
    
    apply_override = st.checkbox("✅ 启用数据集覆盖", value=False, help="覆盖模型的默认数据集配置。")
    
    if apply_override:
        datasets = get_subdirectories(DATA_DIR)
        
        # Calculate relative path dynamically based on model depth
        # Default depth is 1 (e.g. model_zoo/AutoInt) -> ../../data/
        # If depth is 2 (e.g. model_zoo/multitask/APG) -> ../../../data/
        model_depth = len(selected_model.split('/')) if selected_model else 1
        relative_data_path = "../" * (model_depth + 1) + "data/"

        def update_dataset_fields():
            if st.session_state.dataset_template:
                d = st.session_state.dataset_template
                st.session_state.ds_id_val = d
                st.session_state.ds_root_val = relative_data_path
                st.session_state.ds_train_val = d
                st.session_state.ds_valid_val = d
                st.session_state.ds_test_val = d
                st.session_state.ds_infer_val = d

        st.selectbox(
            "快速加载数据集模板 (可选)", 
            datasets, 
            index=None,
            key="dataset_template",
            on_change=update_dataset_fields,
            placeholder="选择以自动填充路径..."
        )
        
        with st.expander("⚙️ 详细设置", expanded=True):
            # Initialize session state if not present
            if "ds_id_val" not in st.session_state: st.session_state.ds_id_val = ""
            if "ds_root_val" not in st.session_state: st.session_state.ds_root_val = relative_data_path
            if "ds_train_val" not in st.session_state: st.session_state.ds_train_val = ""
            if "ds_valid_val" not in st.session_state: st.session_state.ds_valid_val = ""
            if "ds_test_val" not in st.session_state: st.session_state.ds_test_val = ""
            if "ds_infer_val" not in st.session_state: st.session_state.ds_infer_val = ""
            if "ds_split_val" not in st.session_state: st.session_state.ds_split_val = "random"

            st.text_input("Dataset ID", key="ds_id_val", help="数据集的唯一标识符 (可手动输入)")
            st.text_input("Data Root", key="ds_root_val", help="数据根目录路径 (支持绝对路径)")

            # Dynamic file selection based on Data Root
            scan_dir = st.session_state.ds_root_val
            available_files = []
            if scan_dir:
                # Resolve relative path for scanning
                if not os.path.isabs(scan_dir):
                    if selected_model:
                        model_dir = os.path.join(MODEL_ZOO_DIR, selected_model)
                        scan_dir = os.path.normpath(os.path.join(model_dir, scan_dir))
                    else:
                        scan_dir = os.path.abspath(os.path.join(ROOT_DIR, scan_dir))
                
                if os.path.exists(scan_dir) and os.path.isdir(scan_dir):
                    try:
                        available_files = sorted([f for f in os.listdir(scan_dir) if not f.startswith(".")])
                    except:
                        pass

            def render_file_selector(label, key, help_msg):
                if available_files:
                    current = st.session_state.get(key, "")
                    options = list(available_files)
                    # Ensure current value is in options
                    if current and current not in options:
                        options.insert(0, current)
                    elif not current:
                        options.insert(0, "") # Allow empty selection

                    default_idx = options.index(current) if current in options else 0
                    st.selectbox(label, options, key=key, index=default_idx, help=help_msg)
                else:
                    st.text_input(label, key=key, help=help_msg)

            render_file_selector("Train Data", "ds_train_val", "训练数据文件路径")
            render_file_selector("Valid Data", "ds_valid_val", "验证数据文件路径")
            render_file_selector("Test Data", "ds_test_val", "测试数据文件路径")

            render_file_selector("Infer Data", "ds_infer_val", "推理数据文件路径 (可选，留空则忽略)")
            st.selectbox("Split Type", ["random", "sequential"], key="ds_split_val", help="数据切分方式")

if selected_model:
    model_path = os.path.join(MODEL_ZOO_DIR, selected_model)
    
    # Get isolated config paths
    config_info, user_config_save_dir = get_config_paths(selected_model, current_user)
    
    dataset_config_path = config_info["dataset_config.yaml"]["path"]
    model_config_path = config_info["model_config.yaml"]["path"]
    run_expid_path = config_info["run_expid.py"]["path"]
    
    # Tabs with Icons
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🛠️ 配置管理", "▶️ 任务执行", "📊 模型权重", "📈 可视化", "🗂️ 历史记录"])

    with tab1:
        st.markdown("### 📝 配置编辑器")
        
        # Check if any custom config is active
        has_custom = any(config_info[k]["type"] == "custom" for k in config_info)
        if has_custom:
            st.info(f"💡 当前正在编辑 **{current_user}** 的自定义配置。")
        else:
            st.info("💡 当前显示的是系统默认配置。保存修改后将自动创建您的个人副本。")
        
        # Initialize fullscreen state
        if "fullscreen_section" not in st.session_state:
            st.session_state.fullscreen_section = None

        # Helper to render editor
        def render_editor_section(title, file_key, content, lang, lines, key_suffix, download_mime, is_custom, reset_func, body_renderer=None, body_kwargs=None, buffer_key=None):
            # Header with integrated buttons
            is_fullscreen = st.session_state.fullscreen_section == key_suffix
            # run_expid 仍保持旧图标，其余配置切换为原始 YAML 视图提示
            if key_suffix == "script":
                fs_icon = "🗗" if is_fullscreen else "⛶"
                fs_help = "退出全屏" if is_fullscreen else "全屏编辑"
            else:
                fs_icon = "✖" if is_fullscreen else "{ }"
                fs_help = "退出原始视图" if is_fullscreen else "切换原始 YAML 视图"
            
            # Custom CSS for modern, flat design buttons
            st.markdown("""
                <style>
                /* Header action buttons - modern flat design */
                div[data-testid="stHorizontalBlock"] > div style {
                    gap: 0.5rem !important;
                }

                /* Download button */
                div[data-testid="column"] > div > button[kind="primary"] {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                    border: none !important;
                    color: white !important;
                    font-weight: 600 !important;
                    border-radius: 8px !important;
                    padding: 0.5rem 1rem !important;
                    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
                    transition: all 0.2s ease !important;
                }
                div[data-testid="column"] > div > button[kind="primary"]:hover {
                    transform: translateY(-1px) !important;
                    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
                }

                /* Secondary action buttons */
                div[data-testid="column"] > div > button:not([kind="primary"]) {
                    background-color: #f8f9fa !important;
                    border: 1px solid #e9ecef !important;
                    color: #495057 !important;
                    font-weight: 600 !important;
                    border-radius: 8px !important;
                    padding: 0.5rem 1rem !important;
                    transition: all 0.2s ease !important;
                    min-width: 40px !important;
                    cursor: pointer !important;
                }
                div[data-testid="column"] > div > button:not([kind="primary"]):hover {
                    background-color: #e9ecef !important;
                    border-color: #dee2e6 !important;
                    transform: translateY(-1px) !important;
                }

                /* 所有按钮提高点击响应速度 */
                button {
                    touch-action: manipulation !important;
                }
                </style>
            """, unsafe_allow_html=True)

            # Dynamic column ratios based on fullscreen state
            if is_fullscreen:
                col_ratios = [1, 0.08, 0.08, 0.08]
            else:
                col_ratios = [1, 0.15, 0.15, 0.15]

            header_cols = st.columns(col_ratios, gap="small", vertical_alignment="center")

            with header_cols[0]:
                st.markdown(f"### **{title}**")

            with header_cols[1]:
                st.download_button(
                    label="↓",
                    data=content,
                    file_name=title,
                    mime=download_mime,
                    key=f"dl_{key_suffix}_{selected_model}",
                    help="导出文件",
                    use_container_width=True
                )

            with header_cols[2]:
                if is_custom:
                    if st.button("↻", key=f"reset_{key_suffix}_{selected_model}", help="重置为系统默认", use_container_width=True):
                        reset_func(current_user, selected_model, title)
                        _clear_buffer(buffer_key)
                        st.rerun()
                else:
                    st.write("")

            with header_cols[3]:
                if st.button(fs_icon, key=f"fs_btn_{key_suffix}", help=fs_help, use_container_width=True):
                    if is_fullscreen:
                        st.session_state.fullscreen_section = None
                    else:
                        st.session_state.fullscreen_section = key_suffix
                    st.rerun()
            
            if is_custom:
                st.caption("✅ 使用中：个人自定义配置")
            else:
                st.caption("🔒 使用中：系统默认配置")

            # Upload
            with st.expander("📂 上传 / 替换文件"):
                uploaded = st.file_uploader(f"上传 {title}", type=["yaml", "yml", "py"], key=f"upl_{key_suffix}_{selected_model}")
                if uploaded is not None:
                    new_content = uploaded.read().decode("utf-8")
                    save_path = os.path.join(user_config_save_dir, title)
                    save_file_content(save_path, new_content)
                    _set_buffered_content(buffer_key, new_content)
                    st.success("已保存！")
                    st.rerun()

            # Editor
            editor_key = f"editor_{key_suffix}_{selected_model}"

            if body_renderer is not None and not is_fullscreen:
                body_kwargs = body_kwargs or {}
                return body_renderer(
                    content,
                    editor_key=editor_key,
                    lang=lang,
                    lines=lines,
                    is_fullscreen=is_fullscreen,
                    selected_model=selected_model,
                    buffer_key=buffer_key,
                    **body_kwargs
                )

            # Default code editor path
            saved_state = st.session_state.get(editor_key)
            if saved_state and isinstance(saved_state, dict):
                initial_text = saved_state.get('text', content)
            else:
                initial_text = content

            editor_options = {
                "wrap": False,
                "showGutter": True,
                "autoScrollEditorIntoView": True,
                "minLines": lines,
                "maxLines": lines,
                "scrollBeyondLastLine": False
            }

            res = code_editor(
                initial_text,
                lang=lang,
                key=editor_key,
                response_mode="debounce",
                options=editor_options,
                buttons=[{
                    "name": "Copy",
                    "feather": "Copy",
                    "hasText": True,
                    "alwaysOn": True,
                    "commands": ["copyAll"],
                    "style": {"top": "0.46rem", "right": "0.4rem"}
                }]
            )
            if isinstance(res, dict):
                if res.get("text") == "" and res.get("type", "") == "":
                    return initial_text
                return res.get("text", initial_text)
            return initial_text

        # Use buffered contents so Cmd/Ctrl+Enter reruns won't drop unsaved edits
        buf_ds_key = f"buffer_{current_user}_{selected_model}_dataset_config.yaml"
        buf_md_key = f"buffer_{current_user}_{selected_model}_model_config.yaml"
        buf_script_key = f"buffer_{current_user}_{selected_model}_run_expid.py"

        ds_content = _get_buffered_content(buf_ds_key, dataset_config_path)
        md_content = _get_buffered_content(buf_md_key, model_config_path)
        script_content = _get_buffered_content(buf_script_key, run_expid_path)
        
        new_ds_content = ds_content
        new_md_content = md_content
        new_script_content = script_content

        # Layout Logic
        fs = st.session_state.fullscreen_section
        
        if fs == "dataset":
            new_ds_content = render_editor_section(
                "dataset_config.yaml", "dataset_config.yaml", ds_content, "yaml", 35, "dataset", "application/x-yaml", 
                config_info["dataset_config.yaml"]["type"] == "custom", reset_user_config,
                body_renderer=render_dataset_config_body,
                buffer_key=buf_ds_key
            )
        elif fs == "model":
            new_md_content = render_editor_section(
                "model_config.yaml", "model_config.yaml", md_content, "yaml", 35, "model", "application/x-yaml",
                config_info["model_config.yaml"]["type"] == "custom", reset_user_config,
                body_renderer=render_model_config_body,
                buffer_key=buf_md_key
            )
        elif fs == "script":
            new_script_content = render_editor_section(
                "run_expid.py", "run_expid.py", script_content, "python", 35, "script", "text/x-python",
                config_info["run_expid.py"]["type"] == "custom", reset_user_config,
                buffer_key=buf_script_key
            )
        else:
            # Normal View
            col1, col2 = st.columns(2)
            with col1:
                new_ds_content = render_editor_section(
                    "dataset_config.yaml", "dataset_config.yaml", ds_content, "yaml", 15, "dataset", "application/x-yaml",
                    config_info["dataset_config.yaml"]["type"] == "custom", reset_user_config,
                    body_renderer=render_dataset_config_body,
                    buffer_key=buf_ds_key
                )
            with col2:
                new_md_content = render_editor_section(
                    "model_config.yaml", "model_config.yaml", md_content, "yaml", 15, "model", "application/x-yaml",
                    config_info["model_config.yaml"]["type"] == "custom", reset_user_config,
                    body_renderer=render_model_config_body,
                    buffer_key=buf_md_key
                )
            
            st.markdown("---")
            new_script_content = render_editor_section(
                "run_expid.py", "run_expid.py", script_content, "python", 15, "script", "text/x-python",
                config_info["run_expid.py"]["type"] == "custom", reset_user_config,
                buffer_key=buf_script_key
            )

        # Persist working copies in session to survive reruns
        _set_buffered_content(buf_ds_key, new_ds_content)
        _set_buffered_content(buf_md_key, new_md_content)
        _set_buffered_content(buf_script_key, new_script_content)

        if st.button("💾 保存所有配置", type="primary"):
            # Save configs to user directory
            # Note: If in fullscreen, we only really edited one, but we save all 'new_' contents.
            # Since we initialized 'new_' with file content, unedited ones are safe.
            # BUT if we had unsaved edits in others before entering fullscreen, they might be lost if we don't capture them.
            # However, since we only render one in fullscreen, the others are not rendered, so 'new_' is just file content.
            # This means: "Enter Fullscreen" -> "Edit" -> "Save" => SAVES ONLY THE FULLSCREEN ONE (others revert to file).
            # This is acceptable behavior for "Focus Mode".
            
            save_file_content(os.path.join(user_config_save_dir, "dataset_config.yaml"), new_ds_content)
            save_file_content(os.path.join(user_config_save_dir, "model_config.yaml"), new_md_content)
            save_file_content(os.path.join(user_config_save_dir, "run_expid.py"), new_script_content)

            _clear_buffer(buf_ds_key)
            _clear_buffer(buf_md_key)
            _clear_buffer(buf_script_key)
            
            st.toast("配置已保存到您的个人空间！", icon="✅")
            time.sleep(1)
            st.rerun()

    with tab2:
        st.markdown("### 🚀 实验控制")
        
        # --- Task State Restoration & Limits Check ---
        active_tasks = get_active_tasks()
        global_task_count = len(active_tasks)
        user_tasks = [t for t in active_tasks if t['username'] == current_user]
        user_task_count = len(user_tasks)
        
        # Restore state if user has a running task but session is empty
        if st.session_state.run_pid is None and user_task_count > 0:
            # Restore the first found task for this user
            task = user_tasks[0]
            st.session_state.run_pid = task['pid']
            st.session_state.run_logfile = task['logfile']
            st.session_state.running_model = task['model']
            st.toast(f"已恢复任务会话 PID: {task['pid']}", icon="🔄")

        # --- Task Monitor Dashboard ---
        with st.expander("📡 服务器活动与任务监控", expanded=True):
            col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
            
            with col_m1:
                st.metric("全局负载", f"{global_task_count} / 10", help="服务器上的总活跃任务数")
            
            with col_m2:
                delta_color = "normal" if user_task_count == 0 else "off"
                st.metric("您的配额", f"{user_task_count} / 3", "活跃任务", delta_color=delta_color, help="您同时最多只能运行 3 个任务")
            
            with col_m3:
                if active_tasks:
                    task_data = []
                    for t in active_tasks:
                        duration = int(time.time() - t['start_time'])
                        mins, secs = divmod(duration, 60)
                        hours, mins = divmod(mins, 60)
                        dur_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m {secs}s"
                        
                        # Get Resource Usage (root + children)
                        cpu_usage = "N/A"
                        mem_usage = "N/A"
                        gpu_usage = "—"
                        cpu_total, mem_total, gpu_utils, gpu_mems = _aggregate_process_usage(t['pid'])
                        if cpu_total is not None:
                            cpu_usage = f"{cpu_total:.1f}%"
                        if mem_total is not None:
                            mem_usage = f"{mem_total / (1024 * 1024):.0f} MB"
                        
                        # 显示每个GPU的占用，用/隔开
                        if gpu_utils is not None and gpu_mems is not None:
                            gpu_parts = []
                            for i, (util, mem) in enumerate(zip(gpu_utils, gpu_mems)):
                                if util > 0 or mem > 0:  # 只显示有占用的GPU
                                    util_str = f"{util:.0f}%" if util > 0 else "0%"
                                    mem_str = f"{mem / (1024 * 1024):.0f}MB" if mem > 0 else "0MB"
                                    gpu_parts.append(f"GPU{i}:{util_str}/{mem_str}")
                            
                            if gpu_parts:
                                gpu_usage = " / ".join(gpu_parts)
                            else:
                                gpu_usage = "—"

                        task_data.append({
                            "用户": t['username'],
                            "模型": t['model'],
                            "PID": t['pid'],
                            "CPU": cpu_usage,
                            "内存": mem_usage,
                            "GPU占用": gpu_usage,
                            "运行时长": dur_str
                        })
                    st.dataframe(task_data, hide_index=True, use_container_width=True)
                else:
                    st.info("暂无活跃任务")
                            
        def start_process(command, log_filename, model_name, config_override_path=None, meta=None, env_overrides=None):
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(log_filename)
            if not ext:
                ext = ".log"
            user_log_dir = os.path.join(LOG_DIR, current_user)
            os.makedirs(user_log_dir, exist_ok=True)
            log_path = os.path.join(user_log_dir, f"{base}_{run_id}{ext}")
            f = open(log_path, "w")
            
            final_cmd = command
            
            # Logic for Config Injection
            if config_override_path:
                final_cmd = f"{command} --config {config_override_path}"
            elif any(config_info[k]["type"] == "custom" for k in ["dataset_config.yaml", "model_config.yaml"]):
                # Use user config directory
                final_cmd = f"{command} --config {user_config_save_dir}"
            
            # Logic for Script Injection (run_expid.py)
            # If user has custom script, we need to run that instead of the default one.
            # To ensure imports work, we copy it to the model directory with a unique name.
            script_info = config_info["run_expid.py"]
            if script_info["type"] == "custom":
                custom_script_name = f"run_expid_{current_user}.py"
                custom_script_path = os.path.join(model_path, custom_script_name)
                shutil.copy(script_info["path"], custom_script_path)
                # Replace run_expid.py in the command with the custom script name
                final_cmd = final_cmd.replace("run_expid.py", custom_script_name)
            
            env = os.environ.copy()
            if env_overrides:
                for key, value in env_overrides.items():
                    if value is None:
                        env.pop(key, None)
                    else:
                        env[key] = str(value)
            
            # Use start_new_session=True to create a process group, so we can kill the whole tree later
            p = subprocess.Popen(final_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=env)
            f.close()
            
            # Update Session State
            st.session_state.run_pid = p.pid
            st.session_state.run_logfile = log_path
            st.session_state.running_model = model_name
            
            # Register Task Globally
            save_task_state(current_user, p.pid, model_name, log_path)
            
            # Append history record
            meta = meta or {}
            append_history_record(current_user, {
                "pid": p.pid,
                "model": model_name,
                "expid": meta.get("expid"),
                "mode": meta.get("mode", "train"),
                "gpu": meta.get("gpu"),
                "num_workers": meta.get("num_workers"),
                "start_time": time.time(),
                "run_id": run_id,
                "end_time": None,
                "duration": None,
                "status": "running",
                "success": None,
                "logfile": log_path
            })
            
        # ==================== 进程管理辅助函数 ====================
        def _get_os_user():
            try:
                return psutil.Process().username()
            except:
                import getpass
                return getpass.getuser()

        def _verify_process_ownership(pid, username):
            """验证进程是否属于当前系统用户"""
            try:
                proc = psutil.Process(pid)
                # 使用 UID 验证更可靠
                if proc.uids().real == os.getuid():
                    return True
                return proc.username() == _get_os_user()
            except psutil.NoSuchProcess:
                # 进程不存在，视为归属验证通过（以便后续清理状态）
                return True
            except Exception as e:
                logging.warning(f"无法验证进程 {pid} 的用户归属: {e}")
                # 如果无法验证（如权限拒绝），但我们正在尝试停止它
                # 最好返回 False 防止误杀，但应提示用户
                return False

        def _is_process_group_safe(pgid, username):
            """检查进程组是否只包含当前系统用户的进程"""
            system_user = _get_os_user()
            try:
                result = subprocess.run(
                    ["pgrep", "-g", str(pgid)],
                    capture_output=True,
                    text=True,
                    stderr=subprocess.DEVNULL
                )
                if result.stdout:
                    pids_in_pg = result.stdout.strip().split()
                    for proc_pid in pids_in_pg:
                        try:
                            proc = psutil.Process(int(proc_pid))
                            if proc.username() != system_user:
                                logging.warning(f"进程组 {pgid} 中包含其他用户的进程 {proc_pid}")
                                return False
                        except Exception:
                            pass
                return True
            except Exception:
                return False

        def _kill_process_tree(pid, username):
            """终止进程树（主进程及所有子进程）"""
            system_user = _get_os_user()
            try:
                parent = psutil.Process(pid)
                all_processes = [parent]

                # 收集所有子进程
                try:
                    children = parent.children(recursive=True)
                    for child in children:
                        if child.username() == system_user:
                            all_processes.append(child)
                except Exception:
                    pass

                # 先尝试正常终止
                for proc in all_processes:
                    try:
                        if proc.username() == system_user:
                            proc.terminate()
                    except Exception:
                        pass

                # 等待进程退出
                gone, alive = psutil.wait_procs(all_processes, timeout=3)

                # 强制杀死仍在运行的进程
                for proc in alive:
                    try:
                        if proc.username() == system_user:
                            proc.kill()
                    except Exception:
                        pass

            except Exception as e:
                logging.warning(f"清理进程树时出错: {e}")

        def _kill_torchrun_processes(username, model_name=None):
            """
            清理 torchrun 分布式训练进程

            Args:
                username: 逻辑用户名（dashboard 用户）
                model_name: 可选，指定模型名称，只停止该模型的进程
            """
            system_user = _get_os_user()

            # 1. 清理 torchrun 主进程及进程组
            try:
                result = subprocess.run(
                    ["pgrep", "-u", system_user, "-f", "torchrun"],
                    capture_output=True,
                    text=True,
                    stderr=subprocess.DEVNULL
                )
                if result.stdout:
                    torchrun_pids = result.stdout.strip().split()
                    for torchrun_pid in torchrun_pids:
                        try:
                            torchrun_proc = psutil.Process(int(torchrun_pid))
                            if torchrun_proc.username() == system_user:
                                # 如果指定了 model_name，检查进程命令行是否包含该模型路径
                                if model_name:
                                    cmdline = " ".join(torchrun_proc.cmdline())
                                    # 检查是否包含模型路径
                                    if model_name not in cmdline:
                                        logging.debug(f"跳过 torchrun 进程 {torchrun_pid}，不属于模型 {model_name}")
                                        continue

                                subprocess.run(
                                    ["kill", "-9", torchrun_pid],
                                    stderr=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL
                                )
                                logging.info(f"已终止 torchrun 主进程: {torchrun_pid}")
                        except Exception:
                            pass
            except Exception:
                pass

            # 2. 清理残留的 python run_expid 进程
            try:
                result = subprocess.run(
                    ["pgrep", "-u", system_user, "-f", "python.*run_expid"],
                    capture_output=True,
                    text=True,
                    stderr=subprocess.DEVNULL
                )
                if result.stdout:
                    for proc_pid in result.stdout.strip().split():
                        try:
                            proc = psutil.Process(int(proc_pid))
                            if proc.username() == system_user:
                                # 再次检查 cmdline 避免误杀
                                cmdline = " ".join(proc.cmdline())
                                if "dashboard/app.py" in cmdline or "streamlit" in cmdline:
                                    continue

                                # 如果指定了 model_name，检查进程命令行是否包含该模型路径
                                if model_name:
                                    if model_name not in cmdline:
                                        logging.debug(f"跳过进程 {proc_pid}，不属于模型 {model_name}")
                                        continue

                                subprocess.run(
                                    ["kill", "-9", proc_pid],
                                    stderr=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL
                                )
                                logging.info(f"已终止残留训练进程: {proc_pid}")
                        except Exception:
                            pass
            except Exception:
                pass

            # 3. 清理分布式训练端口进程 (端口占用清理)
            try:
                import re
                result = subprocess.run(
                    ["pgrep", "-u", system_user],
                    capture_output=True,
                    text=True,
                    stderr=subprocess.DEVNULL
                )
                if result.stdout:
                    user_pids = set(result.stdout.strip().split())
                    result = subprocess.run(
                        ["ss", "-tlnp"],
                        capture_output=True,
                        text=True,
                        stderr=subprocess.DEVNULL
                    )
                    if result.stdout:
                        for line in result.stdout.strip().split('\n'):
                            if 'LISTEN' in line:
                                port_match = re.search(r':(\d+)\s', line)
                                pid_match = re.search(r'pid=(\d+)', line)
                                if port_match and pid_match:
                                    port, port_pid = port_match.group(1), pid_match.group(1)
                                    if port_pid in user_pids:
                                        try:
                                            port_proc = psutil.Process(int(port_pid))
                                            # 只清理高位端口 (DDP通常使用 random free port, 但 torchrun 也会用 start_method='spawn')
                                            # 这里保守一点，只清理明确属于当前用户的 python 进程
                                            if port_proc.username() == system_user and int(port) > 10000:
                                                 if "python" in port_proc.name().lower():
                                                    subprocess.run(
                                                        ["kill", "-9", port_pid],
                                                        stderr=subprocess.DEVNULL,
                                                        stdout=subprocess.DEVNULL
                                                    )
                                                    logging.info(f"已终止分布式端口 {port} 的进程: {port_pid}")
                                        except Exception:
                                            pass
            except Exception:
                pass

            # 4. 清理孤儿进程
            try:
                result = subprocess.run(
                    ["pgrep", "-u", username],
                    capture_output=True,
                    text=True,
                    stderr=subprocess.DEVNULL
                )
                if result.stdout:
                    for user_pid in result.stdout.strip().split():
                        try:
                            proc = psutil.Process(int(user_pid))
                            cmdline = " ".join(proc.cmdline())
                            if any(kw in cmdline for kw in ["torchrun", "run_expid", "train", "inference"]):
                                try:
                                    parent = proc.parent()
                                    if not parent or not parent.is_running():
                                        subprocess.run(
                                            ["kill", "-9", user_pid],
                                            stderr=subprocess.DEVNULL,
                                            stdout=subprocess.DEVNULL
                                        )
                                        logging.info(f"已终止孤儿进程: {user_pid}")
                                except Exception:
                                    subprocess.run(
                                        ["kill", "-9", user_pid],
                                        stderr=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL
                                    )
                                    logging.info(f"已终止可能孤儿进程: {user_pid}")
                        except Exception:
                            pass
            except Exception:
                pass

        def _cleanup_lock_files(username, root_dir):
            """清理用户的锁文件"""
            try:
                import glob
                lock_patterns = [
                    f"**/{username}_*.lock",
                    f"**/.inference_lock",
                    f"**/*.lock"
                ]

                for pattern in lock_patterns:
                    for lock_file in glob.glob(os.path.join(root_dir, pattern), recursive=True):
                        try:
                            if os.path.exists(lock_file):
                                file_stat = os.stat(lock_file)
                                # 删除最近1小时内创建的锁文件
                                if time.time() - file_stat.st_mtime < 3600:
                                    os.remove(lock_file)
                                    logging.info(f"已删除锁文件: {lock_file}")
                        except Exception as e:
                            logging.warning(f"删除锁文件 {lock_file} 时出错: {e}")
            except Exception as e:
                logging.warning(f"清理锁文件时出错: {e}")

        def _cleanup_gpu_memory():
            """清理 GPU 内存"""
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # ==================== 主停止函数 ====================
        def stop_process():
            if st.session_state.run_pid:
                import logging
                logging.basicConfig(level=logging.WARNING)

                try:
                    pid = st.session_state.run_pid
                    username = current_user
                    model_name = st.session_state.running_model

                    # 1. 验证进程归属
                    if not _verify_process_ownership(pid, username):
                        st.error(f"无法停止进程 {pid}: 进程不属于当前系统用户，或权限不足。")
                        logging.warning(f"安全警告：进程 {pid} 不属于用户 {username}")
                        return

                    # 2. 使用进程组信号终止（SIGTERM）
                    try:
                        pgid = os.getpgid(pid)
                        if _is_process_group_safe(pgid, username):
                            os.killpg(pgid, signal.SIGTERM)
                            time.sleep(2)
                        else:
                            os.kill(pid, signal.SIGTERM)
                            time.sleep(1)
                    except Exception as e:
                        logging.warning(f"进程组 SIGTERM 失败: {e}")
                        try:
                            os.kill(pid, signal.SIGTERM)
                            time.sleep(1)
                        except Exception:
                            pass

                    # 3. 强制终止（SIGKILL）
                    try:
                        os.kill(pid, 0)  # 检查进程是否还在运行
                        if _verify_process_ownership(pid, username):
                            try:
                                pgid = os.getpgid(pid)
                                if _is_process_group_safe(pgid, username):
                                    os.killpg(pgid, signal.SIGKILL)
                                else:
                                    os.kill(pid, signal.SIGKILL)
                            except Exception:
                                os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass  # 进程已经退出
                    except Exception as e:
                        logging.warning(f"强制终止失败: {e}")

                    # 4. 清理进程树
                    _kill_process_tree(pid, username)

                    # 5. 清理 torchrun 分布式进程（只清理当前模型的进程）
                    _kill_torchrun_processes(username, model_name)

                    # 6. 清理锁文件
                    _cleanup_lock_files(username, ROOT_DIR)

                    # 7. 清理 GPU 内存
                    _cleanup_gpu_memory()

                    # 8. 等待资源释放
                    time.sleep(1)

                except Exception as e:
                    logging.warning(f"停止进程时出错: {e}")

                update_history_record(current_user, st.session_state.run_pid, "stopped")

                # Unregister Task Globally
                remove_task_state(st.session_state.run_pid)

                st.session_state.run_pid = None
                st.session_state.running_model = None

        # Experiment Parameters
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            # Load available experiment IDs from model_config.yaml
            available_expids = []
            try:
                if os.path.exists(model_config_path):
                    with open(model_config_path, 'r') as f:
                        m_conf = yaml.safe_load(f)
                        if m_conf and isinstance(m_conf, dict):
                            available_expids = [k for k in m_conf.keys() if k != 'Base']
            except Exception:
                pass
            
            if available_expids:
                expid = st.selectbox("Experiment ID", options=available_expids, help="选择 model_config.yaml 中的 experiment_id")
            else:
                default_val = selected_model.split('/')[-1] + "_test" if selected_model else "test"
                expid = st.text_input("Experiment ID", value=default_val, help="对应 model_config.yaml 中的 experiment_id")
        with col_p2:
            gpu_opts, gpu_labels = get_gpu_options()
            if not gpu_opts:
                gpu_opts = [-1]
                gpu_labels = ["CPU (-1)"]
            gpu_map = dict(zip(gpu_opts, gpu_labels))

            stored_devices = st.session_state.get("_selected_devices", [])
            stored_devices = [d for d in stored_devices if d in gpu_opts]
            if not stored_devices:
                stored_devices = [gpu_opts[0]]
            st.session_state["_selected_devices"] = stored_devices[:]

            if "device_selector" not in st.session_state:
                st.session_state["device_selector"] = stored_devices[:]

            selected_devices = st.multiselect(
                "计算设备 (可多选)",
                options=gpu_opts,
                default=st.session_state.get("device_selector", stored_devices),
                key="device_selector",
                format_func=lambda x: gpu_map.get(x, str(x)),
                help="选择一个或多个设备；多选时自动启用分布式训练"
            )

            if not selected_devices:
                st.warning("未选择设备，已回退到默认 CPU/GPU。")
                selected_devices = [gpu_opts[0]]

            if -1 in selected_devices and len(selected_devices) > 1:
                st.info("检测到 CPU 与 GPU 同时被选中，已自动忽略 CPU。")
                selected_devices = [d for d in selected_devices if d != -1]
                if not selected_devices:
                    selected_devices = [gpu_opts[0]]

            st.session_state["_selected_devices"] = selected_devices[:]
            use_distributed = len(selected_devices) > 1
            device_summary = ",".join(str(d) for d in selected_devices)
            mkl_threading_layer = determine_mkl_threading_layer(selected_devices)
            if use_distributed:
                st.caption(f"💪 多卡模式：{device_summary}")
            else:
                st.caption(f"当前设备：{gpu_map.get(selected_devices[0], selected_devices[0])}")
            st.caption(f"MKL 线程层自动设置为 {mkl_threading_layer}，以匹配当前设备。")
            
            # 保存到session state以便后续使用
            st.session_state["_use_distributed"] = use_distributed
            st.session_state["_device_summary"] = device_summary
            st.session_state["_mkl_threading_layer"] = mkl_threading_layer
            st.session_state["_device_list"] = selected_devices[:]
        with col_p3:
            num_workers = st.number_input("Num Workers", min_value=0, max_value=16, value=3, help="数据加载线程数，0 表示在主线程加载")
            # 保存到session state
            st.session_state["_num_workers"] = num_workers

        # 从session state获取值，如果不存在则使用当前值
        device_list = st.session_state.get("_device_list", selected_devices[:])
        device_meta_value = st.session_state.get("_device_summary", device_summary)
        multi_gpu_enabled = st.session_state.get("_use_distributed", use_distributed)
        current_mkl_threading_layer = st.session_state.get("_mkl_threading_layer", mkl_threading_layer)
        current_num_workers = st.session_state.get("_num_workers", num_workers)

        def build_run_command(run_mode):
            # 从session state获取最新值
            device_list = st.session_state.get("_device_list", [])
            multi_gpu_enabled = st.session_state.get("_use_distributed", False)
            # 获取当前的expid，需要从外部作用域获取
            current_expid = expid
            
            if not device_list:
                device_list = [gpu_opts[0]] if gpu_opts else [-1]
            
            if multi_gpu_enabled and len(device_list) > 1:
                cuda_visible = ",".join(str(d) for d in device_list)
                nproc = len(device_list)
                torchrun_prefix = f"CUDA_VISIBLE_DEVICES={cuda_visible} torchrun --standalone --nnodes=1 --nproc_per_node={nproc}"
                return f"cd {model_path} && {torchrun_prefix} run_expid.py --distributed --expid {current_expid} --mode {run_mode}"
            return f"cd {model_path} && python run_expid.py --expid {current_expid} --gpu {device_list[0]} --mode {run_mode}"

        st.markdown("#### 操作")
        
        col_train, col_infer, col_stop = st.columns(3)
        
        # Limit Checks
        can_start = True
        limit_msg = ""
        if not current_user:
            can_start = False
            limit_msg = "需要用户名。"
        elif user_task_count >= 3:
            can_start = False
            limit_msg = f"达到用户限制 ({user_task_count}/3)。"
        elif global_task_count >= 10:
            can_start = False
            limit_msg = f"达到全局限制 ({global_task_count}/10)。"

        if col_train.button("开始训练", type="primary", disabled=not can_start):
            if not can_start and limit_msg:
                st.error(limit_msg)
            else:
                config_override_dir = None
            
            # Extract dataset params from session state
            ds_id = st.session_state.get("ds_id_val", "custom_dataset")
            ds_root = st.session_state.get("ds_root_val", "")
            ds_train = st.session_state.get("ds_train_val", "")
            ds_valid = st.session_state.get("ds_valid_val", "")
            ds_test = st.session_state.get("ds_test_val", "")
            ds_infer = st.session_state.get("ds_infer_val", "")
            ds_split = st.session_state.get("ds_split_val", "random")
            ds_train_size = 0.8
            ds_valid_size = 0.1
            ds_test_size = 0.1
            # Normalize data paths with data_root
            ds_train = normalize_data_path(ds_root, ds_train)
            ds_valid = normalize_data_path(ds_root, ds_valid)
            ds_test = normalize_data_path(ds_root, ds_test)
            ds_infer = normalize_data_path(ds_root, ds_infer)

            # Always generate temporary config to support num_workers and dataset override
            timestamp = int(time.time())
            temp_config_dir = os.path.join(LOG_DIR, current_user, "configs", f"{expid}_{timestamp}")
            os.makedirs(temp_config_dir, exist_ok=True)
            config_override_dir = temp_config_dir
            
            try:
                # 1. Load and Modify Model Config
                with open(model_config_path, 'r') as f:
                    model_conf = yaml.safe_load(f)
                
                # Update num_workers
                if 'Base' in model_conf:
                    model_conf['Base']['num_workers'] = num_workers
                # Also update in the specific expid if it exists
                if expid in model_conf:
                    model_conf[expid]['num_workers'] = num_workers
                
                # Update dataset_id if override is enabled
                original_ds_id = None
                if apply_override:
                    # Get original dataset_id from the target experiment
                    if expid in model_conf:
                        original_ds_id = model_conf[expid].get('dataset_id')
                    
                    # Update dataset_id in ALL sections (including template)
                    for key in model_conf:
                        if isinstance(model_conf[key], dict) and 'dataset_id' in model_conf[key]:
                             model_conf[key]['dataset_id'] = ds_id
                
                with open(os.path.join(temp_config_dir, "model_config.yaml"), 'w') as f:
                    yaml.dump(model_conf, f)
                    
                # 2. Handle Dataset Config
                if apply_override:
                    # Try to load existing dataset config to preserve feature_cols and label_col
                    existing_ds_conf = {}
                    try:
                        with open(dataset_config_path, 'r') as f:
                            existing_ds_conf = yaml.safe_load(f) or {}
                    except:
                        pass

                    ds_params = {
                        'data_root': ds_root,
                        'data_format': 'parquet',
                        'train_data': ds_train,
                        'valid_data': ds_valid,
                        'test_data': ds_test,
                        'split_type': ds_split,
                        'train_size': ds_train_size,
                        'valid_size': ds_valid_size,
                        'test_size': ds_test_size
                    }
                    
                    # Copy schema from existing config using ORIGINAL ID
                    if original_ds_id and original_ds_id in existing_ds_conf:
                        if 'feature_cols' in existing_ds_conf[original_ds_id]:
                            ds_params['feature_cols'] = existing_ds_conf[original_ds_id]['feature_cols']
                        if 'label_col' in existing_ds_conf[original_ds_id]:
                            ds_params['label_col'] = existing_ds_conf[original_ds_id]['label_col']
                    
                    # Remove empty fields
                    if not ds_valid:
                        ds_params.pop('valid_data', None)
                    if not ds_test:
                        ds_params.pop('test_data', None)
                    
                    if ds_infer:
                        ds_params['infer_data'] = ds_infer
                        
                    dataset_conf = {
                        ds_id: ds_params
                    }
                    with open(os.path.join(temp_config_dir, "dataset_config.yaml"), 'w') as f:
                        yaml.dump(dataset_conf, f)
                        
                    st.toast(f"使用数据集覆盖：{ds_id} (继承自 {original_ds_id})", icon="⚙️")
                else:
                    # Copy existing dataset config
                    shutil.copy(dataset_config_path, os.path.join(temp_config_dir, "dataset_config.yaml"))
                
            except Exception as e:
                st.error(f"生成配置失败：{e}")
                st.stop()

            cmd = build_run_command("train")
            # Include username in log filename for isolation
            start_process(
                cmd,
                f"{expid}_{current_user}_train.log",
                selected_model,
                config_override_dir,
                meta={
                    "expid": expid,
                    "mode": "train",
                    "gpu": device_meta_value,
                    "num_workers": num_workers
                },
                env_overrides={"MKL_THREADING_LAYER": mkl_threading_layer}
            )
            st.rerun()

        if col_infer.button("开始推理", disabled=not can_start):
            if not can_start and limit_msg:
                st.error(limit_msg)
            else:
                # Inference also needs the config override if we want to use the same data settings
                config_override_dir = None
            
            # Extract dataset params from session state
            ds_id = st.session_state.get("ds_id_val", "custom_dataset")
            ds_root = st.session_state.get("ds_root_val", "")
            ds_train = st.session_state.get("ds_train_val", "")
            ds_valid = st.session_state.get("ds_valid_val", "")
            ds_test = st.session_state.get("ds_test_val", "")
            ds_infer = st.session_state.get("ds_infer_val", "")
            ds_split = st.session_state.get("ds_split_val", "random")
            ds_train_size = 0.8
            ds_valid_size = 0.1
            ds_test_size = 0.1
            # Normalize data paths with data_root
            ds_train = normalize_data_path(ds_root, ds_train)
            ds_valid = normalize_data_path(ds_root, ds_valid)
            ds_test = normalize_data_path(ds_root, ds_test)
            ds_infer = normalize_data_path(ds_root, ds_infer)

            # Always generate temporary config to support num_workers and dataset override
            timestamp = int(time.time())
            temp_config_dir = os.path.join(LOG_DIR, current_user, "configs", f"{expid}_infer_{timestamp}")
            os.makedirs(temp_config_dir, exist_ok=True)
            config_override_dir = temp_config_dir
            
            try:
                # 1. Load and Modify Model Config
                with open(model_config_path, 'r') as f:
                    model_conf = yaml.safe_load(f)
                
                # Update num_workers
                if 'Base' in model_conf:
                    model_conf['Base']['num_workers'] = num_workers
                # Also update in the specific expid if it exists
                if expid in model_conf:
                    model_conf[expid]['num_workers'] = num_workers
                
                # Update dataset_id if override is enabled
                original_ds_id = None
                if apply_override:
                    # Get original dataset_id from the target experiment
                    if expid in model_conf:
                        original_ds_id = model_conf[expid].get('dataset_id')
                    
                    # Update dataset_id in ALL sections (including template)
                    for key in model_conf:
                        if isinstance(model_conf[key], dict) and 'dataset_id' in model_conf[key]:
                             model_conf[key]['dataset_id'] = ds_id
                
                with open(os.path.join(temp_config_dir, "model_config.yaml"), 'w') as f:
                    yaml.dump(model_conf, f)
                
                # 2. Handle Dataset Config
                if apply_override:
                    # Try to load existing dataset config to preserve feature_cols and label_col
                    existing_ds_conf = {}
                    try:
                        with open(dataset_config_path, 'r') as f:
                            existing_ds_conf = yaml.safe_load(f) or {}
                    except:
                        pass

                    ds_params = {
                        'data_root': ds_root,
                        'data_format': 'parquet',
                        'train_data': ds_train,
                        'valid_data': ds_valid,
                        'test_data': ds_test,
                        'split_type': ds_split,
                        'train_size': ds_train_size,
                        'valid_size': ds_valid_size,
                        'test_size': ds_test_size
                    }

                    # Copy schema from existing config using ORIGINAL ID
                    if original_ds_id and original_ds_id in existing_ds_conf:
                        if 'feature_cols' in existing_ds_conf[original_ds_id]:
                            ds_params['feature_cols'] = existing_ds_conf[original_ds_id]['feature_cols']
                        if 'label_col' in existing_ds_conf[original_ds_id]:
                            ds_params['label_col'] = existing_ds_conf[original_ds_id]['label_col']

                    if not ds_valid:
                        ds_params.pop('valid_data', None)
                    if not ds_test:
                        ds_params.pop('test_data', None)
                    if ds_infer:
                        ds_params['infer_data'] = ds_infer

                    dataset_conf = {
                        ds_id: ds_params
                    }
                    with open(os.path.join(temp_config_dir, "dataset_config.yaml"), 'w') as f:
                        yaml.dump(dataset_conf, f)
                else:
                    # Copy existing dataset config
                    shutil.copy(dataset_config_path, os.path.join(temp_config_dir, "dataset_config.yaml"))

            except Exception as e:
                st.error(f"生成配置失败：{e}")
                st.stop()

            cmd = build_run_command("inference")
            # Include username in log filename for isolation
            start_process(
                cmd,
                f"{expid}_{current_user}_inference.log",
                selected_model,
                config_override_dir,
                meta={
                    "expid": expid,
                    "mode": "inference",
                    "gpu": device_meta_value,
                    "num_workers": num_workers
                },
                env_overrides={"MKL_THREADING_LAYER": mkl_threading_layer}
            )
            st.rerun()
            
        if col_stop.button("停止进程", type="secondary", disabled=st.session_state.run_pid is None):
            stop_process()
            st.rerun()

        # Status & Logs Monitoring
        st.markdown("---")
        
        is_running = False
        finished_exit_code = None
        finished_pid = None
        if st.session_state.run_pid:
            try:
                # Try to wait for the process to check if it's a zombie (finished but not reaped)
                pid, status = os.waitpid(st.session_state.run_pid, os.WNOHANG)
                if pid == 0:
                    is_running = True
                else:
                    finished_pid = pid
                    if os.WIFEXITED(status):
                        finished_exit_code = os.WEXITSTATUS(status)
                    elif os.WIFSIGNALED(status):
                        finished_exit_code = -os.WTERMSIG(status)
                    is_running = False
            except ChildProcessError:
                try:
                    os.kill(st.session_state.run_pid, 0)
                    is_running = True
                except OSError:
                    is_running = False
                    finished_pid = st.session_state.run_pid
            except OSError:
                is_running = False
                finished_pid = st.session_state.run_pid

            if is_running:
                if st.session_state.running_model == selected_model:
                    st.success(f"🟢 **运行中** (PID: {st.session_state.run_pid}) | 用户: {current_user}")
                else:
                    # Check if selected_model is running in background (from task states)
                    active_tasks = get_active_tasks()
                    selected_model_task = None
                    for task in active_tasks:
                        if task.get('model') == selected_model and task.get('username') == current_user:
                            selected_model_task = task
                            break
                    if selected_model_task:
                        st.success(f"🟢 **运行中** (PID: {selected_model_task['pid']}) | 用户: {current_user}")
                    else:
                        st.info(f"后台运行中：**{st.session_state.running_model}**")
            else:
                # Cleanup if process finished
                if finished_exit_code is None:
                    status_label = "finished"
                else:
                    status_label = "success" if finished_exit_code == 0 else "failed"
                update_history_record(current_user, st.session_state.run_pid, status_label, exit_code=finished_exit_code)
                remove_task_state(st.session_state.run_pid)
                st.session_state.run_pid = None
                st.session_state.running_model = None
                message = "✅ **已完成**" if finished_exit_code == 0 else "⚠️ **已结束 (检查日志)**"
                st.info(message)
                st.rerun()
        else:
            # Check if there's any task running for selected_model
            active_tasks = get_active_tasks()
            selected_model_task = None
            for task in active_tasks:
                if task.get('model') == selected_model and task.get('username') == current_user:
                    selected_model_task = task
                    break
            if selected_model_task:
                st.success(f"🟢 **运行中** (PID: {selected_model_task['pid']}) | 用户: {current_user}")
            else:
                st.info("⚪ **空闲**")

        # Get logfile for selected model (support multi-task concurrent display)
        selected_logfile = None
        if st.session_state.running_model == selected_model:
            # Current session's running model
            selected_logfile = st.session_state.run_logfile
        else:
            # Check from active tasks
            active_tasks = get_active_tasks()
            for task in active_tasks:
                if task.get('model') == selected_model and task.get('username') == current_user:
                    selected_logfile = task.get('logfile')
                    break

        # Show logs if logfile exists for selected model
        if selected_logfile and os.path.exists(selected_logfile):
            st.subheader("📋 实时日志")

            # Auto-refresh toggle
            auto_refresh = st.checkbox("🔄 自动刷新日志", value=True, help="取消勾选以停止页面刷新（查看 TensorBoard 时很有用）")

            # Use selected_logfile instead of session_state.run_logfile
            if selected_logfile and os.path.exists(selected_logfile):
                with open(selected_logfile, "r") as f:
                    lines = f.readlines()
                    if lines:
                        st.code("".join(lines[-50:]), language="text")
                    else:
                        st.caption("等待日志...")
            else:
                st.caption("暂无日志。")
            
            if is_running and auto_refresh:
                time.sleep(2)
                st.rerun()
        else:
            # Show message that no logfile is available for selected model
            st.caption(f"**{selected_model}** 暂无运行中的任务或日志文件。")


    with tab3:
        st.markdown("### 📂 权重与文件")
        checkpoint_dir = os.path.join(model_path, "checkpoints")
        
        if os.path.exists(checkpoint_dir):
            dataset_dirs = get_subdirectories(checkpoint_dir)
            
            if dataset_dirs:
                # Sort dataset directories by modification time (newest first)
                try:
                    dataset_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
                except Exception:
                    pass # Keep default order if sorting fails
                
                selected_dataset_dir = st.selectbox("选择数据集目录", dataset_dirs, index=0, help="默认选中最新修改的目录")
                
                if selected_dataset_dir:
                    target_dir = os.path.join(checkpoint_dir, selected_dataset_dir)
                    files = os.listdir(target_dir)
                    
                    # Dataframe for files
                    file_data = []
                    for f in files:
                        # 跳过 TensorBoard 事件文件，避免干扰
                        if f.startswith("events"):
                            continue
                        fp = os.path.join(target_dir, f)
                        stat = os.stat(fp)
                        size_mb = stat.st_size / (1024 * 1024)
                        mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
                        file_data.append({"文件名": f, "大小 (MB)": f"{size_mb:.2f}", "最后修改时间": mod_time})
                    
                    # Sort files for display (optional, but good for UX)
                    file_data.sort(key=lambda x: x["最后修改时间"], reverse=True)
                    
                    df = pd.DataFrame(file_data)
                    st.dataframe(df, use_container_width=True)
            else:
                st.warning("在 checkpoints 中未找到数据集目录。")
        else:
            st.warning("尚未找到 checkpoints 目录。请先运行训练任务。")

        # Log Preview Section (per-user log dir to avoid覆盖)
        user_log_dir = os.path.join(LOG_DIR, current_user)
        log_files = []
        if os.path.exists(user_log_dir):
            for f in os.listdir(user_log_dir):
                if f.endswith(".log"):
                    log_files.append(f)
        # 优先按 expid 过滤；如果过滤后为空，则回退展示全部，避免“未找到”误报
        filtered_logs = []
        if expid:
            filtered_logs = [f for f in log_files if expid in f]
        if not filtered_logs:
            filtered_logs = log_files[:]

        if filtered_logs:
            try:
                filtered_logs.sort(key=lambda x: os.path.getmtime(os.path.join(user_log_dir, x)), reverse=True)
            except Exception:
                pass

            st.markdown("---")
            st.subheader("📜 日志查看器")
            selected_log = st.selectbox("选择日志文件", filtered_logs, index=0, help="默认选中最新修改的日志（若按 expid 过滤为空则展示全部）")
            if selected_log:
                log_path = os.path.join(user_log_dir, selected_log)
                log_tail = _tail_text_file(log_path, max_bytes=80000, max_lines=800)
                st.code(log_tail or "日志为空或不可读。", language="text")
                st.caption(f"路径: `{log_path}`")
        else:
            st.caption("未找到当前用户的日志文件。")

    with tab4:
        st.header("📈 TensorBoard 可视化")
        checkpoint_dir = os.path.join(model_path, "checkpoints")
        
        if os.path.exists(checkpoint_dir):
            st.markdown('<div class="css-1r6slb0">', unsafe_allow_html=True)
            
            st.subheader("🔌 连接信息")
            st.caption("日志目录源：")
            st.code(checkpoint_dir, language="bash")
            
            st.markdown("---")
            
            # TensorBoard URL (server-friendly). Attempt auto-detect, fallback to env/default.
            detected_url, detected_pid, detected_port = detect_tensorboard_process()
            default_tb_host = os.environ.get("TB_HOST") or _guess_public_host()
            default_tb_port = os.environ.get("TB_PORT", "6006")
            default_tb_url = detected_url or (f"http://{default_tb_host}:{default_tb_port}" if default_tb_host else "")

            # Allow prefill before widget instantiation to avoid Streamlit mutation error.
            if "tb_url_prefill" in st.session_state:
                st.session_state.tb_url_input = st.session_state.tb_url_prefill
                del st.session_state.tb_url_prefill
            if "tb_url_input" not in st.session_state:
                st.session_state.tb_url_input = default_tb_url

            tb_url = st.text_input(
                "可视化访问地址",
                value=st.session_state.tb_url_input,
                help="远程访问请填 http://<服务器IP或域名>:<端口>（不要 localhost，确保端口已放行）",
                placeholder="http://<服务器IP或域名>:6006",
                key="tb_url_input",
            )

            status_msg = "未检测到运行中的 TensorBoard 进程。"
            if detected_url:
                status_msg = f"检测到 TensorBoard 进程 (PID {detected_pid})，监听 {detected_port}。"
            st.caption(status_msg)
            if not detected_url and not default_tb_host:
                st.caption("提示：服务器访问请填公网 IP/域名，例如 http://<server-ip>:6006。")

            col_launch, col_open = st.columns(2)
            
            with col_launch:
                if st.button("🚀 启动服务 (端口 6006)", type="primary", use_container_width=True):
                    cmd = f"tensorboard --logdir {checkpoint_dir} --port 6006"
                    subprocess.Popen(cmd, shell=True)
                    st.toast("TensorBoard 服务已启动！", icon="✅")
                    time.sleep(1)
            
            with col_open:
                st.markdown(
                    f"""
                    <a href="{tb_url}" target="_blank" style="text-decoration: none;">
                        <div style="
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            width: 100%;
                            background-color: #EFF6FF;
                            border: 1px solid #BFDBFE;
                            color: #1E40AF;
                            padding: 0.55rem;
                            border-radius: 8px;
                            font-weight: 600;
                            text-decoration: none;
                            transition: all 0.2s;
                        ">
                            🔗 打开界面
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True
                )

            # Auto-detect and auto-fill button
            if st.button("🔍 自动检测端口并填充", use_container_width=True):
                if detected_url:
                    st.session_state.tb_url_prefill = detected_url
                    st.rerun()
                else:
                    st.warning("未检测到 TensorBoard 进程，请先启动再重试。")

            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("### 💡 快速指南")
            st.markdown("""
            - **第一步**：点击 **启动服务** 以开启后台进程。
            - **第二步**：点击 **打开界面** 以查看指标。
            - **注意**：如果切换模型，您可能需要重启服务或刷新 TensorBoard。
            """)
        else:
            st.warning("⚠️ 未找到 checkpoints 目录。请先运行训练任务。")

    with tab5:
        st.header("🗂️ 历史运行记录")
        st.caption("此处仅展示当前左侧所选用户的运行记录，切换侧边栏用户名即可查看自己的历史。每条记录支持删除。")
        st.markdown("""
            <style>
            .history-row {
                padding: 8px 10px;
                background: #F9FAFB;
                border: 1px solid #E5E7EB;
                border-radius: 10px;
                margin-bottom: 6px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.02);
                font-size: 0.9rem;
            }
            .history-row strong { color: #111827; }
            .history-meta { color: #4B5563; font-size: 0.85rem; }
            </style>
        """, unsafe_allow_html=True)

        target_user = current_user
        user_history = load_history(target_user)
        
        # 添加批量删除功能
        if user_history:
            col_del1, col_del2 = st.columns([3, 1])
            with col_del1:
                st.markdown("**操作**：")
            with col_del2:
                if st.button("🗑️ 删除所有历史记录", type="secondary", use_container_width=True):
                    if delete_all_history(target_user):
                        st.toast("已删除所有历史记录！", icon="✅")
                        st.rerun()
        
        if user_history:
            for i, rec in enumerate(user_history[:100]):
                start_ts = rec.get('start_time')
                start_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_ts)) if start_ts else "-"
                duration_val = rec.get('duration')
                now_ts = time.time()
                if rec.get('status') == 'running' and start_ts:
                    duration_val = now_ts - start_ts
                
                gpu_val = rec.get('gpu')
                if gpu_val in (None, -1, "-1"):
                    gpu_display = 'CPU'
                else:
                    gpu_display = str(gpu_val)
                
                success_flag = rec.get('success')
                success_display = '✅' if success_flag else ('❌' if success_flag is False else '—')
                # 优先使用历史记录中已存储的指标（在任务完成时已从CSV读取并存储）
                val_metrics = rec.get('val_metrics')
                test_metrics = rec.get('test_metrics')
                if val_metrics is not None or test_metrics is not None:
                    # 使用存储的指标
                    metrics_summary = ""
                    if val_metrics:
                        metrics_summary += f"val: {val_metrics}"
                    if test_metrics:
                        if metrics_summary:
                            metrics_summary += " | "
                        metrics_summary += f"test: {test_metrics}"
                else:
                    # 回退到从CSV文件读取指标，如果不存在则从日志提取
                    val_metrics, test_metrics = get_metrics_from_csv(rec.get('expid'), os.path.join(MODEL_ZOO_DIR, rec.get('model')))
                    if val_metrics != "-" or test_metrics != "-":
                        metrics_summary = f"val: {val_metrics}"
                        if test_metrics != "-":
                            metrics_summary += f" | test: {test_metrics}"
                    else:
                        metrics_summary = extract_latest_metrics(rec.get('logfile'))
                
                # 单行摘要
                summary = (
                    f"<div class='history-row'>"
                    f"<strong>{start_str}</strong> · {rec.get('model', '-')}/{rec.get('expid', '-')} "
                    f"<span class='history-meta'>｜模式 {rec.get('mode', '-')} ｜GPU {gpu_display} ｜时长 {format_duration(duration_val)} "
                    f"｜状态 {rec.get('status', '-')} {success_display} ｜指标 {metrics_summary}</span>"
                    f"</div>"
                )

                col1, col3 = st.columns([10, 1], gap="small")
                with col1:
                    st.markdown(summary, unsafe_allow_html=True)
                with col3:
                    pid = rec.get('pid')
                    if pid and st.button("🗑️", key=f"delete_{i}", help="删除此记录"):
                        if delete_history_record(target_user, pid):
                            st.toast(f"已删除记录 PID: {pid}", icon="✅")
                            st.rerun()
        else:
            st.info(f"用户 {target_user} 暂无历史记录")
