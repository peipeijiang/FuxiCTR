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

# Set page config
st.set_page_config(
    page_title="FuxiCTR 实验平台",
    page_icon="🍭",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
</style>
""", unsafe_allow_html=True)

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_ZOO_DIR = os.path.join(ROOT_DIR, "model_zoo")
DATA_DIR = os.path.join(ROOT_DIR, "data")
LOG_DIR = os.path.join(ROOT_DIR, "dashboard", "logs")
TASK_STATE_DIR = os.path.join(ROOT_DIR, "dashboard", "state", "tasks")
USER_CONFIG_DIR = os.path.join(ROOT_DIR, "dashboard", "user_configs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TASK_STATE_DIR, exist_ok=True)
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
    st.title("📚 FuxiCTR 平台使用指南")
    
    if st.button("🔙 返回主页"):
        st.session_state.show_tutorial = False
        st.rerun()
        
    st.markdown("---")
    
    st.markdown("""
    ## 1. 平台简介
    FuxiCTR 是一个可配置、模块化、高性能的 CTR 预估库。本平台（FuxiCTR Studio）提供了一个可视化的界面，用于管理实验、配置参数、监控任务和分析结果。
    
    ## 2. 快速入门 (App 使用流程)
    
    ### 第一步：身份设置
    在左侧边栏的 **"用户身份"** 区域输入您的用户名。
    *   **作用**：用于区分不同用户的任务，防止日志冲突，并进行资源配额管理（每人同时限跑 1 个任务）。
    
    ### 第二步：选择模型
    在左侧边栏选择您要实验的模型（例如 `DeepFM` 或 `DCN`）。
    *   选择后，主界面会自动加载该模型的配置文件。
    
    ### 第三步：数据配置
    您有两种方式配置数据：
    1.  **快速覆盖 (推荐)**：在侧边栏勾选 `✅ 启用数据集覆盖`，然后选择一个预设的数据集（如 `tiny_csv`）。系统会自动生成临时的配置文件。
    2.  **手动配置**：在主界面的 `🛠️ 配置管理` 标签页中，直接编辑 `dataset_config.yaml`。
    
    ### 第四步：启动任务
    切换到 `▶️ 任务执行` 标签页：
    1.  设置 **实验ID** (Experiment ID)。
    2.  选择 **GPU 设备** (或使用 CPU)。
    3.  点击 `🔥 开始训练` 或 `🔮 开始推理`。
    
    ### 第五步：监控与分析
    *   **实时日志**：任务启动后，下方会自动显示运行日志。
    *   **任务监控**：展开 `📡 服务器活动与任务监控` 面板，查看当前服务器负载和您的任务状态。
    *   **可视化**：训练完成后，切换到 `📈 可视化` 标签页，一键启动 TensorBoard 查看 Loss 和 AUC 曲线。
    
    ---
    
    ## 3. 核心配置详解
    
    ### 🛠 dataset_config.yaml (数据配置)
    此文件定义了数据集的路径、格式和特征处理方式。
    
    ```yaml
    dataset_id:
        data_root: ../data/  # 数据根目录
        data_format: csv     # 数据格式: csv, h5, parquet 等
        train_data: ../data/train.csv
        valid_data: ../data/valid.csv
        test_data: ../data/test.csv
        min_categr_count: 1
        feature_cols:        # 特征定义列表
            - {name: user_id, active: True, dtype: str, type: categorical}
            - {name: item_id, active: True, dtype: str, type: categorical}
            - {name: age, active: True, dtype: float, type: numeric}
        label_col: {name: click, dtype: float}
    ```
    
    ### ⚙️ model_config.yaml (模型配置)
    此文件定义了模型的超参数、优化器和训练设置。
    
    ```yaml
    Base: # 所有模型的基类配置
        model_root: './checkpoints/'
        workers: 3
        verbose: 1
        patience: 2
        pickle_feature_encoder: True
        use_hdf5: True
        save_best_only: True
        every_x_epochs: 1
        debug: False

    DeepFM_test: # 特定实验配置
        model: DeepFM
        dataset_id: tiny_csv # 关联 dataset_config 中的 ID
        loss: 'binary_crossentropy'
        metrics: ['logloss', 'AUC']
        task: binary_classification
        optimizer: adam
        learning_rate: 1.e-3
        embedding_regularizer: 1.e-8
        net_regularizer: 0
        batch_size: 128
        embedding_dim: 4
        epochs: 1
        shuffle: True
        seed: 2019
        monitor: 'AUC'
        monitor_mode: 'max'
    ```
    
    ## 4. 常见问题
    *   **Q: 为什么无法启动任务？**
        *   A: 请检查是否已输入用户名，或者是否已达到个人/全局任务数量限制。
    *   **Q: 如何查看历史日志？**
        *   A: 在 `📊 模型权重` 标签页中，选择对应的数据集目录，可以查看和预览历史日志文件。
    """)
    st.stop() # Stop execution here to show only tutorial

# Header
if st.session_state.show_tutorial:
    render_tutorial()

col_main, col_help = st.columns([6, 1])
with col_main:
    st.title("FuxiCTR 实验平台")
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

    # Define user list from provided images
    user_options = [
        "yeshao",
        "chenzeng2", "cywang50", "gjwang5", "gxwang9", 
        "hkhu3", "junzhang56", "mxsong", "qiancao6", 
        "taozhang48", "wenzhang33", "yangzhou23", "ymbo2"
    ]
    
    # Ensure prev_user is in options
    default_index = 0
    if st.session_state.prev_user in user_options:
        default_index = user_options.index(st.session_state.prev_user)

    current_user = st.selectbox("用户名", user_options, index=default_index, help="用于任务限制（每位用户最多 1 个任务）。")
    
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
            st.text_input("Train Data", key="ds_train_val", help="训练数据文件路径")
            st.text_input("Valid Data", key="ds_valid_val", help="验证数据文件路径")
            st.text_input("Test Data", key="ds_test_val", help="测试数据文件路径")
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
    tab1, tab2, tab3, tab4 = st.tabs(["🛠️ 配置管理", "▶️ 任务执行", "📊 模型权重", "📈 可视化"])

    with tab1:
        st.markdown("### 📝 配置编辑器")
        
        # Check if any custom config is active
        has_custom = any(config_info[k]["type"] == "custom" for k in config_info)
        if has_custom:
            st.info(f"💡 当前正在编辑 **{current_user}** 的自定义配置。")
        else:
            st.info("💡 当前显示的是系统默认配置。保存修改后将自动创建您的个人副本。")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dataset Config Section
            ds_info = config_info["dataset_config.yaml"]
            is_custom_ds = ds_info["type"] == "custom"
            
            header_cols = st.columns([3, 1, 1])
            with header_cols[0]:
                st.markdown("### dataset_config.yaml")
            with header_cols[1]:
                st.download_button(
                    label="⬇️ 导出",
                    data=load_file_content(dataset_config_path),
                    file_name="dataset_config.yaml",
                    mime="application/x-yaml",
                    key=f"dl_ds_{selected_model}"
                )
            with header_cols[2]:
                if is_custom_ds:
                    if st.button("🔄 重置", key=f"reset_ds_{selected_model}", help="删除自定义配置，恢复系统默认"):
                        reset_user_config(current_user, selected_model, "dataset_config.yaml")
                        st.rerun()
            
            if is_custom_ds:
                st.caption("✅ 使用中：个人自定义配置")
            else:
                st.caption("🔒 使用中：系统默认配置")

            with st.expander("📂 上传 / 替换文件"):
                uploaded_dataset = st.file_uploader("上传 dataset_config.yaml", type=["yaml", "yml"], key=f"dataset_uploader_{selected_model}")
                if uploaded_dataset is not None:
                    content = uploaded_dataset.read().decode("utf-8")
                    # Always save to user config dir
                    save_path = os.path.join(user_config_save_dir, "dataset_config.yaml")
                    save_file_content(save_path, content)
                    st.success("已保存到个人配置！")
                    st.rerun()
            
            dataset_content = load_file_content(dataset_config_path)
            new_dataset_content = st.text_area("内容", dataset_content, height=400, key=f"dataset_editor_{selected_model}", label_visibility="collapsed")
            
        with col2:
            # Model Config Section
            md_info = config_info["model_config.yaml"]
            is_custom_md = md_info["type"] == "custom"
            
            header_cols_m = st.columns([3, 1, 1])
            with header_cols_m[0]:
                st.markdown("### model_config.yaml")
            with header_cols_m[1]:
                st.download_button(
                    label="⬇️ 导出",
                    data=load_file_content(model_config_path),
                    file_name="model_config.yaml",
                    mime="application/x-yaml",
                    key=f"dl_md_{selected_model}"
                )
            with header_cols_m[2]:
                if is_custom_md:
                    if st.button("🔄 重置", key=f"reset_md_{selected_model}", help="删除自定义配置，恢复系统默认"):
                        reset_user_config(current_user, selected_model, "model_config.yaml")
                        st.rerun()

            if is_custom_md:
                st.caption("✅ 使用中：个人自定义配置")
            else:
                st.caption("🔒 使用中：系统默认配置")
            
            with st.expander("📂 上传 / 替换文件"):
                uploaded_model = st.file_uploader("上传 model_config.yaml", type=["yaml", "yml"], key=f"model_uploader_{selected_model}")
                if uploaded_model is not None:
                    content = uploaded_model.read().decode("utf-8")
                    # Always save to user config dir
                    save_path = os.path.join(user_config_save_dir, "model_config.yaml")
                    save_file_content(save_path, content)
            model_content = load_file_content(model_config_path)
            new_model_content = st.text_area("内容", model_content, height=400, key=f"model_editor_{selected_model}", label_visibility="collapsed")

        st.markdown("---")
        
        # Run Script Config Section
        script_info = config_info["run_expid.py"]
        is_custom_script = script_info["type"] == "custom"
        
        header_cols_s = st.columns([3, 1, 1])
        with header_cols_s[0]:
            st.markdown("### 📜 run_expid.py")
        with header_cols_s[1]:
            st.download_button(
                label="⬇️ 导出",
                data=load_file_content(run_expid_path),
                file_name="run_expid.py",
                mime="text/x-python",
                key=f"dl_script_{selected_model}"
            )
        with header_cols_s[2]:
            if is_custom_script:
                if st.button("🔄 重置", key=f"reset_script_{selected_model}", help="删除自定义脚本，恢复系统默认"):
                    reset_user_config(current_user, selected_model, "run_expid.py")
                    st.rerun()
        
        if is_custom_script:
            st.caption("✅ 使用中：个人自定义脚本")
        else:
            st.caption("🔒 使用中：系统默认脚本")

        with st.expander("📂 上传 / 替换脚本"):
            uploaded_script = st.file_uploader("上传 run_expid.py", type=["py"], key=f"script_uploader_{selected_model}")
            if uploaded_script is not None:
                content = uploaded_script.read().decode("utf-8")
                save_path = os.path.join(user_config_save_dir, "run_expid.py")
                save_file_content(save_path, content)
                st.success("脚本已更新到个人配置！")
                st.rerun()

        run_expid_content = load_file_content(run_expid_path)
        new_run_expid_content = st.text_area("内容", run_expid_content, height=300, key=f"script_editor_{selected_model}", label_visibility="collapsed")

        if st.button("💾 保存所有配置", type="primary"):
            # Save configs to user directory
            save_file_content(os.path.join(user_config_save_dir, "dataset_config.yaml"), new_dataset_content)
            save_file_content(os.path.join(user_config_save_dir, "model_config.yaml"), new_model_content)
            save_file_content(os.path.join(user_config_save_dir, "run_expid.py"), new_run_expid_content)
            
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
                st.metric("全局负载", f"{global_task_count} / 3", help="服务器上的总活跃任务数")
            
            with col_m2:
                delta_color = "normal" if user_task_count == 0 else "off"
                st.metric("您的配额", f"{user_task_count} / 1", "活跃任务", delta_color=delta_color, help="您同时最多只能运行 1 个任务")
            
            with col_m3:
                if active_tasks:
                    task_data = []
                    for t in active_tasks:
                        duration = int(time.time() - t['start_time'])
                        mins, secs = divmod(duration, 60)
                        hours, mins = divmod(mins, 60)
                        dur_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m {secs}s"
                        task_data.append({
                            "用户": t['username'],
                            "模型": t['model'],
                            "PID": t['pid'],
                            "运行时长": dur_str
                        })
                    st.dataframe(task_data, hide_index=True, use_container_width=True)
                else:
                    st.info("暂无活跃任务")
                            
        def start_process(command, log_filename, model_name, config_override_path=None):
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
            
            # Use start_new_session=True to create a process group, so we can kill the whole tree later
            p = subprocess.Popen(final_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            f.close()
            
            # Update Session State
            st.session_state.run_pid = p.pid
            st.session_state.run_logfile = log_path
            st.session_state.running_model = model_name
            
            # Register Task Globally
            save_task_state(current_user, p.pid, model_name, log_path)
            
        def stop_process():
            if st.session_state.run_pid:
                try:
                    # Kill the process group to ensure child processes (like the python script) are also killed
                    # Since start_new_session=True was used, the PID is the PGID.
                    # We use SIGKILL (9) to ensure it stops, and avoid os.getpgid lookup which fails if parent is dead.
                    os.killpg(st.session_state.run_pid, signal.SIGKILL)
                except Exception:
                    pass
                
                # Unregister Task Globally
                remove_task_state(st.session_state.run_pid)
                
                st.session_state.run_pid = None
                st.session_state.running_model = None

        # Experiment Parameters
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            expid = st.text_input("Experiment ID", value=selected_model.split('/')[-1] + "_test" if selected_model else "test", help="对应 model_config.yaml 中的 experiment_id")
        with col_p2:
            gpu = st.selectbox("GPU Device", options=[0, 1, 2, 3, 4, 5, 6, 7], index=0, help="选择使用的 GPU 设备 ID")

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
        elif user_task_count >= 1:
            can_start = False
            limit_msg = f"达到用户限制 ({user_task_count}/1)。"
        elif global_task_count >= 3:
            can_start = False
            limit_msg = f"达到全局限制 ({global_task_count}/3)。"

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

            if apply_override:
                # Generate temporary config override
                timestamp = int(time.time())
                temp_config_dir = os.path.join(LOG_DIR, "configs", f"{expid}_{timestamp}")
                os.makedirs(temp_config_dir, exist_ok=True)
                
                # 1. Load and Modify Model Config
                try:
                    with open(model_config_path, 'r') as f:
                        model_conf = yaml.safe_load(f)
                    
                    # Get original dataset_id from the target experiment
                    original_ds_id = None
                    if expid in model_conf:
                        original_ds_id = model_conf[expid].get('dataset_id')
                    
                    # Update dataset_id in ALL sections (including template)
                    for key in model_conf:
                        if isinstance(model_conf[key], dict) and 'dataset_id' in model_conf[key]:
                             model_conf[key]['dataset_id'] = ds_id
                    
                    with open(os.path.join(temp_config_dir, "model_config.yaml"), 'w') as f:
                        yaml.dump(model_conf, f)
                        
                    # 2. Generate Dataset Config
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
                    
                    # Remove empty fields to avoid "Invalid data path: *.parquet" error
                    # FuxiCTR will try to glob "*.parquet" if the path is empty, which fails.
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
                        
                    config_override_dir = temp_config_dir
                    st.toast(f"使用数据集覆盖：{ds_id} (继承自 {original_ds_id})", icon="⚙️")
                    
                except Exception as e:
                    st.error(f"生成配置覆盖失败：{e}")
                    st.stop()

            cmd = f"cd {model_path} && python run_expid.py --expid {expid} --gpu {gpu} --mode train"
            # Include username in log filename for isolation
            start_process(cmd, f"{expid}_{current_user}_train.log", selected_model, config_override_dir)
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

            if apply_override:
                 # Generate temporary config override (Same logic as training)
                timestamp = int(time.time())
                temp_config_dir = os.path.join(LOG_DIR, "configs", f"{expid}_infer_{timestamp}")
                os.makedirs(temp_config_dir, exist_ok=True)
                
                try:
                    with open(model_config_path, 'r') as f:
                        model_conf = yaml.safe_load(f)
                    
                    # Get original dataset_id from the target experiment
                    original_ds_id = None
                    if expid in model_conf:
                        original_ds_id = model_conf[expid].get('dataset_id')
                    
                    # Update dataset_id in ALL sections (including template)
                    for key in model_conf:
                        if isinstance(model_conf[key], dict) and 'dataset_id' in model_conf[key]:
                             model_conf[key]['dataset_id'] = ds_id
                    
                    with open(os.path.join(temp_config_dir, "model_config.yaml"), 'w') as f:
                        yaml.dump(model_conf, f)
                    
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
                    config_override_dir = temp_config_dir
                except Exception as e:
                    st.error(f"生成配置覆盖失败：{e}")
                    st.stop()

            cmd = f"cd {model_path} && python run_expid.py --expid {expid} --gpu {gpu} --mode inference"
            # Include username in log filename for isolation
            start_process(cmd, f"{expid}_{current_user}_inference.log", selected_model, config_override_dir)
            st.rerun()
            
        if col_stop.button("🛑 停止进程", type="secondary", disabled=st.session_state.run_pid is None):
            stop_process()
            st.rerun()

        # Status & Logs Monitoring
        st.markdown("---")
        
        is_running = False
        if st.session_state.run_pid:
            try:
                # Try to wait for the process to check if it's a zombie (finished but not reaped)
                # os.WNOHANG ensures we don't block if it's still running
                pid, status = os.waitpid(st.session_state.run_pid, os.WNOHANG)
                if pid == 0:
                    # Process is still running
                    is_running = True
                else:
                    # Process exited and was reaped
                    is_running = False
            except ChildProcessError:
                # Not a child of this process (e.g. restored from session state after restart)
                # Fallback to os.kill check
                try:
                    os.kill(st.session_state.run_pid, 0)
                    is_running = True
                except OSError:
                    is_running = False
            except OSError:
                is_running = False

            if is_running:
                if st.session_state.running_model == selected_model:
                    st.success(f"🟢 **运行中** (PID: {st.session_state.run_pid}) | 用户: {current_user}")
                else:
                    st.info(f"后台运行中：**{st.session_state.running_model}**")
            else:
                # Cleanup if process finished
                remove_task_state(st.session_state.run_pid)
                st.session_state.run_pid = None
                st.session_state.running_model = None
                st.info("✅ **已完成**")
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
                selected_dataset_dir = st.selectbox("选择数据集目录", dataset_dirs)
                
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
                    
                    df = pd.DataFrame(file_data)
                    st.dataframe(df, use_container_width=True)

                    # Log Preview Section
                    if log_files:
                        st.markdown("---")
                        st.subheader("📜 日志查看器")
                        selected_log = st.selectbox("选择日志文件", log_files)
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
