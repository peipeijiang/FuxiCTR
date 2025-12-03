import streamlit as st
import os
import subprocess
import sys
import time
import signal
import pandas as pd

# Set page config
st.set_page_config(
    page_title="FuxiCTR Studio",
    page_icon="🍭",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
</style>
""", unsafe_allow_html=True)

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_ZOO_DIR = os.path.join(ROOT_DIR, "model_zoo")
DATA_DIR = os.path.join(ROOT_DIR, "data")
LOG_DIR = os.path.join(ROOT_DIR, "dashboard", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

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
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

# Header
st.title("FuxiCTR Studio")
st.markdown("Professional CTR Model Training & Inference Platform")

st.markdown("---")

# Sidebar for Selection
with st.sidebar:
    st.header("🎛️ Project Settings")
    
    st.markdown("### 📍 Model Selection")
    models = get_models(MODEL_ZOO_DIR)
    selected_model = st.selectbox("Select Model", models, label_visibility="collapsed")
    if selected_model:
        st.caption(f"Path: `model_zoo/{selected_model}`")

    st.markdown("### 💾 Dataset Selection")
    datasets = get_subdirectories(DATA_DIR)
    selected_dataset = st.selectbox("Select Dataset", datasets, label_visibility="collapsed")

    if selected_dataset:
        st.info(f"**Active Dataset:**\n\n`{selected_dataset}`")
    
    st.markdown("---")
    st.markdown("### ⚙️ System Info")
    st.text(f"Python: {sys.version.split()[0]}")
    st.text(f"CWD: {os.getcwd()}")

if selected_model:
    model_path = os.path.join(MODEL_ZOO_DIR, selected_model)
    config_dir = os.path.join(model_path, "config")
    
    # Ensure config dir exists
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)

    dataset_config_path = os.path.join(config_dir, "dataset_config.yaml")
    model_config_path = os.path.join(config_dir, "model_config.yaml")
    run_expid_path = os.path.join(model_path, "run_expid.py")

    # Tabs with Icons
    tab1, tab2, tab3, tab4 = st.tabs(["🛠️ Configuration", "▶️ Execution", "📊 Checkpoints", "📈 TensorBoard"])

    with tab1:
        st.markdown("### 📝 Configuration Editor")
        st.info("Edit the configuration files below. Changes are applied immediately upon saving.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("dataset_config.yaml")
            with st.expander("📂 Upload / Replace File"):
                uploaded_dataset = st.file_uploader("Upload dataset_config.yaml", type=["yaml", "yml"], key="dataset_uploader")
                if uploaded_dataset is not None:
                    content = uploaded_dataset.read().decode("utf-8")
                    save_file_content(dataset_config_path, content)
                    st.success("File updated!")
                    st.rerun()
            
            dataset_content = load_file_content(dataset_config_path)
            new_dataset_content = st.text_area("Content", dataset_content, height=400, key="dataset_editor", label_visibility="collapsed")
            
        with col2:
            st.subheader("model_config.yaml")
            with st.expander("📂 Upload / Replace File"):
                uploaded_model = st.file_uploader("Upload model_config.yaml", type=["yaml", "yml"], key="model_uploader")
                if uploaded_model is not None:
                    content = uploaded_model.read().decode("utf-8")
                    save_file_content(model_config_path, content)
                    st.success("File updated!")
                    st.rerun()

            model_content = load_file_content(model_config_path)
            new_model_content = st.text_area("Content", model_content, height=400, key="model_editor", label_visibility="collapsed")

        st.markdown("---")
        st.subheader("📜 run_expid.py")
        with st.expander("📂 Upload / Replace Script"):
            uploaded_script = st.file_uploader("Upload run_expid.py", type=["py"], key="script_uploader")
            if uploaded_script is not None:
                content = uploaded_script.read().decode("utf-8")
                save_file_content(run_expid_path, content)
                st.success("Script updated!")
                st.rerun()

        run_expid_content = load_file_content(run_expid_path)
        new_run_expid_content = st.text_area("Content", run_expid_content, height=300, key="script_editor", label_visibility="collapsed")

        if st.button("💾 Save All Configurations", type="primary"):
            save_file_content(dataset_config_path, new_dataset_content)
            save_file_content(model_config_path, new_model_content)
            save_file_content(run_expid_path, new_run_expid_content)
            st.toast("All configurations saved successfully!", icon="✅")

    with tab2:
        st.markdown("### 🚀 Experiment Control")
        
        with st.container():
            col_params1, col_params2 = st.columns(2)
            with col_params1:
                expid = st.text_input("🏷️ Experiment ID", value=f"{selected_model.split('/')[-1]}_test")
            with col_params2:
                gpu = st.selectbox("🖥️ GPU Device", [0, -1], index=0, help="-1 for CPU")
        
        # State management
        if "run_pid" not in st.session_state:
            st.session_state.run_pid = None
        if "run_logfile" not in st.session_state:
            st.session_state.run_logfile = None
        if "running_model" not in st.session_state:
            st.session_state.running_model = None

        def start_process(command, log_filename, model_name):
            log_path = os.path.join(LOG_DIR, log_filename)
            f = open(log_path, "w")
            p = subprocess.Popen(command, shell=True, stdout=f, stderr=subprocess.STDOUT)
            f.close()
            st.session_state.run_pid = p.pid
            st.session_state.run_logfile = log_path
            st.session_state.running_model = model_name
            
        def stop_process():
            if st.session_state.run_pid:
                try:
                    os.kill(st.session_state.run_pid, signal.SIGTERM)
                except Exception:
                    pass
                st.session_state.run_pid = None
                st.session_state.running_model = None

        st.markdown("#### Actions")
        col_train, col_infer, col_stop = st.columns(3)
        
        # Check if current selected model matches the running model
        is_running_other_model = st.session_state.run_pid is not None and st.session_state.running_model != selected_model
        
        if is_running_other_model:
            st.warning(f"⚠️ Another model (**{st.session_state.running_model}**) is currently running. Please stop it before starting a new task.")

        if col_train.button("🔥 Start Training", type="primary", disabled=st.session_state.run_pid is not None):
            cmd = f"cd {model_path} && python run_expid.py --expid {expid} --gpu {gpu} --mode train"
            start_process(cmd, f"{expid}_train.log", selected_model)
            st.rerun()

        if col_infer.button("🔮 Start Inference", disabled=st.session_state.run_pid is not None):
            cmd = f"cd {model_path} && python run_expid.py --expid {expid} --gpu {gpu} --mode inference"
            start_process(cmd, f"{expid}_inference.log", selected_model)
            st.rerun()
            
        if col_stop.button("🛑 Stop Process", type="secondary", disabled=st.session_state.run_pid is None):
            stop_process()
            st.rerun()

        # Status & Logs Monitoring
        st.markdown("---")
        
        is_running = False
        if st.session_state.run_pid:
            try:
                os.kill(st.session_state.run_pid, 0)
                is_running = True
                if st.session_state.running_model == selected_model:
                    st.success(f"🟢 **Running** (PID: {st.session_state.run_pid})")
                else:
                    st.info(f"Running in background: **{st.session_state.running_model}**")
            except OSError:
                is_running = False
                st.session_state.run_pid = None
                st.session_state.running_model = None
                st.info("✅ **Finished**")
                st.rerun()
        else:
            st.info("⚪ **Idle**")

        # Only show logs if the selected model is the one running
        if st.session_state.running_model == selected_model or st.session_state.running_model is None:
            st.subheader("📋 Live Logs")
            
            # Auto-refresh toggle
            auto_refresh = st.checkbox("🔄 Auto-refresh Logs", value=True, help="Uncheck to stop page reload (useful for viewing TensorBoard)")

            if st.session_state.run_logfile and os.path.exists(st.session_state.run_logfile):
                with open(st.session_state.run_logfile, "r") as f:
                    lines = f.readlines()
                    if lines:
                        st.code("".join(lines[-50:]), language="text")
                    else:
                        st.caption("Waiting for logs...")
            else:
                st.caption("No logs available.")
            
            if is_running and auto_refresh:
                time.sleep(2)
                st.rerun()
        else:
            st.caption(f"Logs for **{st.session_state.running_model}** are hidden. Switch back to that model to view live logs.")

    with tab3:
        st.markdown("### 📂 Checkpoints & Files")
        checkpoint_dir = os.path.join(model_path, "checkpoints")
        
        if os.path.exists(checkpoint_dir):
            dataset_dirs = get_subdirectories(checkpoint_dir)
            
            if dataset_dirs:
                selected_dataset_dir = st.selectbox("Select Dataset Directory", dataset_dirs)
                
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
                        file_data.append({"Filename": f, "Size (MB)": f"{size_mb:.2f}", "Last Modified": mod_time})
                        if f.endswith(".log"):
                            log_files.append(f)
                    
                    df = pd.DataFrame(file_data)
                    st.dataframe(df, use_container_width=True)

                    # Log Preview Section
                    if log_files:
                        st.markdown("---")
                        st.subheader("📜 Log Viewer")
                        selected_log = st.selectbox("Select Log File", log_files)
                        if selected_log:
                            log_path = os.path.join(target_dir, selected_log)
                            with open(log_path, "r") as f:
                                st.code(f.read(), language="text")
            else:
                st.warning("No dataset directories found in checkpoints.")
        else:
            st.warning("No checkpoints directory found yet. Run a training session first.")

    with tab4:
        st.header("📈 TensorBoard Visualization")
        checkpoint_dir = os.path.join(model_path, "checkpoints")
        
        if os.path.exists(checkpoint_dir):
            st.markdown('<div class="css-1r6slb0">', unsafe_allow_html=True)
            
            st.subheader("🔌 Connection")
            st.caption("Log Directory Source:")
            st.code(checkpoint_dir, language="bash")
            
            st.markdown("---")
            
            col_launch, col_open = st.columns(2)
            
            with col_launch:
                if st.button("🚀 Launch Server (Port 6006)", type="primary", use_container_width=True):
                    cmd = f"tensorboard --logdir {checkpoint_dir} --port 6006"
                    subprocess.Popen(cmd, shell=True)
                    st.toast("TensorBoard Service Started!", icon="✅")
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
                            🔗 Open Interface
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("### 💡 Quick Guide")
            st.markdown("""
            - **Step 1**: Click **Launch Server** to start the background process.
            - **Step 2**: Click **Open Interface** to view the metrics.
            - **Note**: If you switch models, you may need to restart the server or refresh TensorBoard.
            """)
        else:
            st.warning("⚠️ No checkpoints directory found. Run a training session first.")
