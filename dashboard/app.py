import streamlit as st
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
import psutil
from code_editor import code_editor

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
    """Return (util%, bytes) summed across all GPUs touched by given pids.
    If no specific process matches, return total GPU utilization across all devices."""
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

    total_util = 0.0
    total_mem = 0
    matched_any = False
    try:
        device_count = pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError:
        return None, None

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
                total_util += util
                total_mem += device_mem
    
    # If no specific process matched, return total GPU utilization across all devices
    # This is useful for distributed training where child processes might not be detected
    if not matched_any:
        for idx in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_util += util
                total_mem += mem_info.used
                matched_any = True
            except pynvml.NVMLError:
                continue
    
    if not matched_any:
        return None, None
    
    # Normalize utilization to percentage (already in percentage)
    # Round to 1 decimal place for display
    total_util = round(total_util, 1)
    return total_util, total_mem

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
            updated = True
            break
    if updated:
        save_history(username, history)

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

    # Ensure prev_user is in options
    default_index = 0
    if st.session_state.prev_user in USER_OPTIONS:
        default_index = USER_OPTIONS.index(st.session_state.prev_user)

    current_user = st.selectbox("用户名", USER_OPTIONS, index=default_index, help="用于任务限制（每位用户最多 3 个任务）。")
    
    # Detect User Switch
    if current_user != st.session_state.prev_user:
        st.session_state.prev_user = current_user
        # Clear session state to prevent leaking previous user's task info
        st.session_state.run_pid = None
        st.session_state.run_logfile = None
        st.session_state.running_model = None
        st.rerun()

    if not current_user:
        st.warning("请输入用户名。")

    st.markdown("### 📍 模型选择")
    models = get_models(MODEL_ZOO_DIR)
    selected_model = st.selectbox("选择模型", models, label_visibility="collapsed")
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
                path = os.path.join(relative_data_path, d)
                st.session_state.ds_train_val = path
                st.session_state.ds_valid_val = path
                st.session_state.ds_test_val = path
                st.session_state.ds_infer_val = path
                st.session_state.ds_root_val = relative_data_path

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
                    
                    st.selectbox(label, options, key=key, help=help_msg)
                else:
                    st.text_input(label, key=key, help=help_msg)

            render_file_selector("Train Data", "ds_train_val", "训练数据文件路径")
            render_file_selector("Valid Data", "ds_valid_val", "验证数据文件路径")
            render_file_selector("Test Data", "ds_test_val", "测试数据文件路径")

            st.text_input("Infer Data", key="ds_infer_val", help="推理数据文件路径 (可选，留空则忽略)")
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
        def render_editor_section(title, file_key, content, lang, lines, key_suffix, download_mime, is_custom, reset_func):
            # Header with integrated buttons
            is_fullscreen = st.session_state.fullscreen_section == key_suffix
            # Using 🗗 (Window Restore) for exit, and ⛶ (Square Four Corners) for enter
            fs_icon = "🗗" if is_fullscreen else "⛶"
            fs_help = "退出全屏" if is_fullscreen else "全屏编辑"
            
            # Custom CSS for flat, modern, bold buttons in the header
            st.markdown("""
                <style>
                /* Target buttons inside columns for the header actions */
                div[data-testid="column"] button {
                    border: 1px solid transparent; /* Flat style */
                    background-color: #f8f9fa; /* Very light gray */
                    font-weight: 900 !important; /* Bold icons */
                    border-radius: 8px; /* Rounded corners */
                    transition: all 0.2s ease;
                    padding: 0.4rem 0.8rem; /* Increased padding for better touch target and look */
                    margin: 0px;
                    width: 100%; /* Force button to fill the small column */
                }
                div[data-testid="column"] button:hover {
                    border: 1px solid #e0e0e0;
                    background-color: #ffffff;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    color: #000;
                }
                </style>
            """, unsafe_allow_html=True)

            # Dynamic column ratios based on fullscreen state
            # Fullscreen: Container is wide, use small ratios for buttons to keep them compact
            # Normal: Container is narrow, give buttons more relative width to prevent squashing
            if is_fullscreen:
                col_ratios = [1, 0.04, 0.04, 0.04]
            else:
                col_ratios = [1, 0.12, 0.12, 0.12]
                
            header_cols = st.columns(col_ratios, gap="small")
            
            with header_cols[0]:
                st.markdown(f"### **{title}**") # Bold title
            
            with header_cols[1]:
                st.download_button(
                    label="📥",
                    data=content,
                    file_name=title,
                    mime=download_mime,
                    key=f"dl_{key_suffix}_{selected_model}",
                    help="导出文件"
                )
                
            with header_cols[2]:
                if is_custom:
                    if st.button("↺", key=f"reset_{key_suffix}_{selected_model}", help="重置为系统默认"):
                        reset_func(current_user, selected_model, title)
                        st.rerun()
                else:
                    st.write("") # Placeholder
            
            with header_cols[3]:
                if st.button(fs_icon, key=f"fs_btn_{key_suffix}", help=fs_help):
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
                    st.success("已保存！")
                    st.rerun()

            # Editor
            # Try to preserve edits across reruns by checking session state
            editor_key = f"editor_{key_suffix}_{selected_model}"
            
            # Fix AttributeError: 'NoneType' object has no attribute 'get'
            # st.session_state.get(editor_key) might return None if initialized but empty
            saved_state = st.session_state.get(editor_key)
            if saved_state and isinstance(saved_state, dict):
                initial_text = saved_state.get('text', content)
            else:
                initial_text = content
            
            # Use minLines/maxLines to enforce height and alignment
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
                response_mode="debounce",  # ensure latest text is returned without needing an explicit submit
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
            # If the editor hasn't emitted any event yet (type/text empty), fallback to the current visible text
            if isinstance(res, dict):
                if res.get("text") == "" and res.get("type", "") == "":
                    return initial_text
                return res.get("text", initial_text)
            return initial_text

        # Load contents
        ds_content = load_file_content(dataset_config_path)
        md_content = load_file_content(model_config_path)
        script_content = load_file_content(run_expid_path)
        
        new_ds_content = ds_content
        new_md_content = md_content
        new_script_content = script_content

        # Layout Logic
        fs = st.session_state.fullscreen_section
        
        if fs == "dataset":
            new_ds_content = render_editor_section(
                "dataset_config.yaml", "dataset_config.yaml", ds_content, "yaml", 35, "dataset", "application/x-yaml", 
                config_info["dataset_config.yaml"]["type"] == "custom", reset_user_config
            )
        elif fs == "model":
            new_md_content = render_editor_section(
                "model_config.yaml", "model_config.yaml", md_content, "yaml", 35, "model", "application/x-yaml",
                config_info["model_config.yaml"]["type"] == "custom", reset_user_config
            )
        elif fs == "script":
            new_script_content = render_editor_section(
                "run_expid.py", "run_expid.py", script_content, "python", 35, "script", "text/x-python",
                config_info["run_expid.py"]["type"] == "custom", reset_user_config
            )
        else:
            # Normal View
            col1, col2 = st.columns(2)
            with col1:
                new_ds_content = render_editor_section(
                    "dataset_config.yaml", "dataset_config.yaml", ds_content, "yaml", 15, "dataset", "application/x-yaml",
                    config_info["dataset_config.yaml"]["type"] == "custom", reset_user_config
                )
            with col2:
                new_md_content = render_editor_section(
                    "model_config.yaml", "model_config.yaml", md_content, "yaml", 15, "model", "application/x-yaml",
                    config_info["model_config.yaml"]["type"] == "custom", reset_user_config
                )
            
            st.markdown("---")
            new_script_content = render_editor_section(
                "run_expid.py", "run_expid.py", script_content, "python", 15, "script", "text/x-python",
                config_info["run_expid.py"]["type"] == "custom", reset_user_config
            )

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
                        cpu_total, mem_total, gpu_util, gpu_mem = _aggregate_process_usage(t['pid'])
                        if cpu_total is not None:
                            cpu_usage = f"{cpu_total:.1f}%"
                        if mem_total is not None:
                            mem_usage = f"{mem_total / (1024 * 1024):.0f} MB"
                        gpu_parts = []
                        if gpu_util is not None:
                            gpu_parts.append(f"{gpu_util:.0f}%")
                        if gpu_mem is not None:
                            gpu_parts.append(f"{gpu_mem / (1024 * 1024):.0f} MB")
                        if gpu_parts:
                            gpu_usage = " / ".join(gpu_parts)

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
            log_path = os.path.join(LOG_DIR, log_filename)
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
                "end_time": None,
                "duration": None,
                "status": "running",
                "success": None,
                "logfile": log_path
            })
            
        def stop_process():
            if st.session_state.run_pid:
                import logging
                logging.basicConfig(level=logging.WARNING)
                
                try:
                    pid = st.session_state.run_pid
                    
                    # 1. 先尝试正常终止（SIGTERM）
                    try:
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(1)  # 给进程时间正常退出
                    except Exception:
                        pass
                    
                    # 2. 如果还在运行，强制终止（SIGKILL）
                    try:
                        os.kill(pid, 0)  # 检查进程是否存在
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass  # 进程已经退出
                    
                    # 3. 清理可能的子进程（分布式训练的子进程）
                    try:
                        parent = psutil.Process(pid)
                        children = parent.children(recursive=True)
                        for child in children:
                            try:
                                child.terminate()
                            except:
                                pass
                        
                        # 等待子进程退出
                        gone, alive = psutil.wait_procs(children, timeout=3)
                        for child in alive:
                            try:
                                child.kill()
                            except:
                                pass
                    except:
                        pass
                    
                    # 4. 安全地清理当前用户的TCPStore相关进程和端口占用
                    try:
                        import subprocess
                        # 使用前端选择的用户名，而不是系统用户名
                        current_username = current_user
                        
                        # 安全方法1：只清理当前用户的进程
                        # 使用pgrep查找当前用户的进程，然后逐个杀死
                        try:
                            # 查找当前用户的torchrun进程
                            result = subprocess.run(["pgrep", "-u", current_username, "-f", "torchrun"], 
                                                  capture_output=True, text=True, stderr=subprocess.DEVNULL)
                            if result.stdout:
                                pids = result.stdout.strip().split()
                                for proc_pid in pids:
                                    try:
                                        # 检查是否是当前进程的子进程
                                        parent = psutil.Process(pid)
                                        children = parent.children(recursive=True)
                                        child_pids = [str(child.pid) for child in children]
                                        if proc_pid in child_pids or proc_pid == str(pid):
                                            subprocess.run(["kill", "-9", proc_pid], 
                                                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                                    except:
                                        pass
                        except:
                            pass
                        
                        try:
                            # 查找当前用户的python run_expid进程
                            result = subprocess.run(["pgrep", "-u", current_username, "-f", "python.*run_expid"], 
                                                  capture_output=True, text=True, stderr=subprocess.DEVNULL)
                            if result.stdout:
                                pids = result.stdout.strip().split()
                                for proc_pid in pids:
                                    try:
                                        # 检查是否是当前进程的子进程
                                        parent = psutil.Process(pid)
                                        children = parent.children(recursive=True)
                                        child_pids = [str(child.pid) for child in children]
                                        if proc_pid in child_pids or proc_pid == str(pid):
                                            subprocess.run(["kill", "-9", proc_pid], 
                                                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                                    except:
                                        pass
                        except:
                            pass
                        
                        # 安全方法2：只清理当前进程打开的端口
                        try:
                            # 查找当前进程及其子进程打开的端口
                            def get_process_ports(process_pid):
                                """获取进程及其子进程打开的所有端口"""
                                ports = set()
                                try:
                                    proc = psutil.Process(process_pid)
                                    # 获取当前进程的连接
                                    for conn in proc.connections():
                                        if conn.laddr:
                                            ports.add(conn.laddr.port)
                                    # 获取子进程的连接
                                    for child in proc.children(recursive=True):
                                        try:
                                            for conn in child.connections():
                                                if conn.laddr:
                                                    ports.add(conn.laddr.port)
                                        except:
                                            pass
                                except:
                                    pass
                                return ports
                            
                            # 获取当前进程的所有端口
                            process_ports = get_process_ports(pid)
                            
                            # 清理这些端口
                            for port in process_ports:
                                try:
                                    result = subprocess.run(["lsof", f"-ti:{port}"], 
                                                          capture_output=True, text=True, stderr=subprocess.DEVNULL)
                                    if result.stdout:
                                        port_pids = result.stdout.strip().split()
                                        for port_pid in port_pids:
                                            # 检查端口进程是否是当前进程的子进程
                                            try:
                                                port_proc = psutil.Process(int(port_pid))
                                                if port_proc.username() == current_username:
                                                    # 进一步检查是否是当前进程树
                                                    parent = psutil.Process(pid)
                                                    children = parent.children(recursive=True)
                                                    child_pids = [child.pid for child in children]
                                                    if int(port_pid) in child_pids or int(port_pid) == pid:
                                                        subprocess.run(["kill", "-9", port_pid], 
                                                                     stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                                            except:
                                                pass
                                except:
                                    pass
                        except:
                            pass
                        
                    except Exception as e:
                        logging.warning(f"清理进程时出错: {e}")
                    
                    # 5. 清理GPU内存
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
                    
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
        with col_p3:
            num_workers = st.number_input("Num Workers", min_value=1, max_value=16, value=3, help="数据加载线程数")

        device_list = selected_devices[:]  # snapshot for command builders
        device_meta_value = device_summary
        multi_gpu_enabled = use_distributed

        def build_run_command(run_mode):
            if multi_gpu_enabled:
                cuda_visible = ",".join(str(d) for d in device_list)
                nproc = len(device_list)
                torchrun_prefix = f"CUDA_VISIBLE_DEVICES={cuda_visible} torchrun --standalone --nnodes=1 --nproc_per_node={nproc}"
                return f"cd {model_path} && {torchrun_prefix} run_expid.py --distributed --expid {expid} --mode {run_mode}"
            return f"cd {model_path} && python run_expid.py --expid {expid} --gpu {device_list[0]} --mode {run_mode}"

        st.markdown("#### 操作")
        
        col_train, col_infer, col_stop = st.columns(3)
        
        # Check if current selected model matches the running model
        is_running_other_model = st.session_state.run_pid is not None and st.session_state.running_model != selected_model
        
        if is_running_other_model:
            st.warning(f"⚠️ 另一个模型 (**{st.session_state.running_model}**) 正在运行。请在开始新任务前停止它。")

        # Limit Checks
        can_start = True
        limit_msg = ""
        if not current_user:
            can_start = False
            limit_msg = "需要用户名。"
        elif st.session_state.run_pid is not None:
            can_start = False # Already running in this session
        elif user_task_count >= 3:
            can_start = False
            limit_msg = f"达到用户限制 ({user_task_count}/3)。"
        elif global_task_count >= 10:
            can_start = False
            limit_msg = f"达到全局限制 ({global_task_count}/10)。"

        if col_train.button("🔥 开始训练", type="primary", disabled=not can_start):
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

            # Always generate temporary config to support num_workers and dataset override
            timestamp = int(time.time())
            temp_config_dir = os.path.join(LOG_DIR, "configs", f"{expid}_{timestamp}")
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

        if col_infer.button("🔮 开始推理", disabled=not can_start):
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

            # Always generate temporary config to support num_workers and dataset override
            timestamp = int(time.time())
            temp_config_dir = os.path.join(LOG_DIR, "configs", f"{expid}_infer_{timestamp}")
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
            
        if col_stop.button("🛑 停止进程", type="secondary", disabled=st.session_state.run_pid is None):
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
            st.info("⚪ **空闲**")

        # Only show logs if the selected model is the one running
        if st.session_state.running_model == selected_model or st.session_state.running_model is None:
            st.subheader("📋 实时日志")
            
            # Auto-refresh toggle
            auto_refresh = st.checkbox("🔄 自动刷新日志", value=True, help="取消勾选以停止页面刷新（查看 TensorBoard 时很有用）")

            if st.session_state.run_logfile and os.path.exists(st.session_state.run_logfile):
                with open(st.session_state.run_logfile, "r") as f:
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
            st.caption(f"**{st.session_state.running_model}** 的日志已隐藏。切换回该模型以查看实时日志。")

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
                    log_files = []
                    for f in files:
                        fp = os.path.join(target_dir, f)
                        stat = os.stat(fp)
                        size_mb = stat.st_size / (1024 * 1024)
                        mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
                        file_data.append({"文件名": f, "大小 (MB)": f"{size_mb:.2f}", "最后修改时间": mod_time})
                        if f.endswith(".log"):
                            log_files.append(f)
                    
                    # Sort files for display (optional, but good for UX)
                    file_data.sort(key=lambda x: x["最后修改时间"], reverse=True)
                    
                    df = pd.DataFrame(file_data)
                    st.dataframe(df, use_container_width=True)

                    # Log Preview Section
                    if log_files:
                        # Sort log files by modification time (newest first)
                        try:
                            log_files.sort(key=lambda x: os.path.getmtime(os.path.join(target_dir, x)), reverse=True)
                        except Exception:
                            pass

                        st.markdown("---")
                        st.subheader("📜 日志查看器")
                        selected_log = st.selectbox("选择日志文件", log_files, index=0, help="默认选中最新修改的日志")
                        if selected_log:
                            log_path = os.path.join(target_dir, selected_log)
                            with open(log_path, "r") as f:
                                st.code(f.read(), language="text")
            else:
                st.warning("在 checkpoints 中未找到数据集目录。")
        else:
            st.warning("尚未找到 checkpoints 目录。请先运行训练任务。")

    with tab4:
        st.header("📈 TensorBoard 可视化")
        checkpoint_dir = os.path.join(model_path, "checkpoints")
        
        if os.path.exists(checkpoint_dir):
            st.markdown('<div class="css-1r6slb0">', unsafe_allow_html=True)
            
            st.subheader("🔌 连接信息")
            st.caption("日志目录源：")
            st.code(checkpoint_dir, language="bash")
            
            st.markdown("---")
            
            col_launch, col_open = st.columns(2)
            
            with col_launch:
                if st.button("🚀 启动服务 (端口 6006)", type="primary", use_container_width=True):
                    cmd = f"tensorboard --logdir {checkpoint_dir} --port 6006"
                    subprocess.Popen(cmd, shell=True)
                    st.toast("TensorBoard 服务已启动！", icon="✅")
                    time.sleep(1)
            
            with col_open:
                st.markdown(
                    """
                    <a href="http://localhost:6006" target="_blank" style="text-decoration: none;">
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
        st.caption("此处仅展示当前左侧所选用户的运行记录，切换侧边栏用户名即可查看自己的历史。")

        target_user = current_user
        user_history = load_history(target_user)
        if user_history:
            detail_rows = []
            now_ts = time.time()
            for rec in user_history[:100]:
                start_ts = rec.get('start_time')
                start_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_ts)) if start_ts else "-"
                duration_val = rec.get('duration')
                if rec.get('status') == 'running' and start_ts:
                    duration_val = now_ts - start_ts
                gpu_val = rec.get('gpu')
                if gpu_val in (None, -1, "-1"):
                    gpu_display = 'CPU'
                else:
                    gpu_display = str(gpu_val)
                success_flag = rec.get('success')
                success_display = '✅' if success_flag else ('❌' if success_flag is False else '—')
                detail_rows.append({
                    "开始时间": start_str,
                    "模型": rec.get('model', '-') or '-',
                    "实验": rec.get('expid', '-') or '-',
                    "模式": rec.get('mode', '-') or '-',
                    "GPU": gpu_display,
                    "时长": format_duration(duration_val),
                    "状态": rec.get('status', '-') or '-',
                    "成功": success_display
                })
            st.dataframe(detail_rows, use_container_width=True, hide_index=True)
        else:
            st.info(f"用户 {target_user} 暂无历史记录")
