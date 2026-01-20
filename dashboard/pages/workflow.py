import streamlit as st
import requests
import os
import yaml
from datetime import datetime

API_BASE = "http://localhost:8001"

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
USER_CONFIG_DIR = os.path.join(ROOT_DIR, "dashboard", "user_configs")
MODEL_ZOO_DIR = os.path.join(ROOT_DIR, "model_zoo")

# User Options
USER_OPTIONS = [
    "yeshao",
    "chenzeng2", "cywang50", "gjwang5", "gxwang9",
    "hkhu3", "junzhang56", "mxsong", "qiancao6",
    "taozhang48", "wenzhang33", "yangzhou23", "ymbo2"
]


def get_models(root_dir):
    """Get available models for a given user directory."""
    models = []
    if not os.path.exists(root_dir):
        return []

    for d in os.listdir(root_dir):
        path = os.path.join(root_dir, d)
        if os.path.isdir(path) and not d.startswith(".") and not d.startswith("__"):
            if os.path.exists(os.path.join(path, "run_expid.py")):
                models.append(d)
            else:
                for sub_d in os.listdir(path):
                    sub_path = os.path.join(path, sub_d)
                    if os.path.isdir(sub_path) and not sub_d.startswith(".") and not sub_d.startswith("__"):
                        if os.path.exists(os.path.join(sub_path, "run_expid.py")):
                            models.append(f"{d}/{sub_d}")
    return sorted(models)


def get_model_config_path(model_name, username):
    """Get the path to model_config.yaml for a given model and user."""
    # First check user config
    user_config_dir = os.path.join(USER_CONFIG_DIR, username, model_name)
    user_config_path = os.path.join(user_config_dir, "model_config.yaml")

    if os.path.exists(user_config_path):
        return user_config_path

    # Fallback to model zoo config
    model_dir = os.path.join(MODEL_ZOO_DIR, model_name)
    default_config_path = os.path.join(model_dir, "config", "model_config.yaml")

    return default_config_path


def get_experiment_ids(model_name, username):
    """Get available experiment IDs from model_config.yaml."""
    model_config_path = get_model_config_path(model_name, username)
    experiment_ids = []

    try:
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config and isinstance(config, dict):
                    experiment_ids = [k for k in config.keys() if k != 'Base']
    except Exception:
        pass

    return experiment_ids


def render_status_badge(status: str) -> str:
    """Render modern status badge with softer colors."""
    status_styles = {
        "pending": "background-color: #f3f4f6; color: #4b5563; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.02em;",
        "running": "background-color: #eff6ff; color: #2563eb; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.02em;",
        "completed": "background-color: #ecfdf5; color: #059669; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.02em;",
        "failed": "background-color: #fef2f2; color: #dc2626; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.02em;",
        "cancelled": "background-color: #f9fafb; color: #6b7280; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.02em;"
    }

    style = status_styles.get(status.lower(), status_styles["pending"])
    status_text = {
        "pending": "待处理",
        "running": "运行中",
        "completed": "已完成",
        "failed": "失败",
        "cancelled": "已取消"
    }.get(status.lower(), status.upper())

    if status.lower() == "running":
        return f'<span style="{style}" class="status-running">{status_text}</span>'
    return f'<span style="{style}">{status_text}</span>'


def render_step_status_badge(status: str) -> str:
    """Render HTML status badge for workflow steps with Chinese text."""
    status_styles = {
        "pending": "background-color: #f3f4f6; color: #4b5563; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.02em;",
        "running": "background-color: #eff6ff; color: #2563eb; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.02em;",
        "completed": "background-color: #ecfdf5; color: #059669; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.02em;",
        "failed": "background-color: #fef2f2; color: #dc2626; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.02em;"
    }

    style = status_styles.get(status.lower(), status_styles["pending"])
    status_text = {
        "pending": "待处理",
        "running": "运行中",
        "completed": "已完成",
        "failed": "失败"
    }.get(status.lower(), status.upper())

    return f'<span style="{style}">{status_text}</span>'


def get_step_name_chinese(step_name: str) -> str:
    """Translate step name from English to Chinese."""
    step_names = {
        "data_fetch": "数据获取",
        "train": "模型训练",
        "infer": "模型推理",
        "monitor": "监控",
        "transport": "传输",
        "upload": "上传"
    }
    return step_names.get(step_name.lower(), step_name.upper())


# Add custom CSS for animations and styling
st.markdown("""
<style>
/* Running status pulse animation */
@keyframes pulse-ring {
    0% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.4); }
    70% { box-shadow: 0 0 0 6px rgba(37, 99, 235, 0); }
    100% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }
}

/* Status badge styling */
span[class*="status-running"] {
    position: relative !important;
    padding-left: 18px !important;
}

span[class*="status-running"]::before {
    content: "";
    position: absolute !important;
    left: 6px !important;
    width: 6px !important;
    height: 6px !important;
    background: #2563eb !important;
    border-radius: 50% !important;
    animation: pulse-ring 2s ease-out infinite !important;
}

/* Task row styling */
.task-row {
    display: flex;
    align-items: center;
    padding: 14px 16px;
    border-bottom: 1px solid #f3f4f6;
    transition: background-color 0.15s ease;
    min-height: 60px;
}
.task-row:hover {
    background-color: #f9fafb;
}
.task-row:last-child {
    border-bottom: none;
}
.section-divider {
    margin: 24px 0;
}
.task-name {
    font-weight: 600;
    font-size: 14px;
    color: #111827;
}
.task-meta {
    font-size: 13px;
    color: #6b7280;
}

/* Modern icon button styling */
.stButton > button {
    border: 1px solid #e5e7eb;
    background: white;
    color: #374151;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.15s ease;
}

.stButton > button:hover {
    background: #f9fafb;
    border-color: #d1d5db;
}

.stButton > button[kind="primary"] {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 6px 16px;
    font-weight: 500;
}

.stButton > button[kind="primary"]:hover {
    background-color: #1d4ed8;
}
</style>
""", unsafe_allow_html=True)

# Get query params
query_params = st.query_params
task_id = query_params.get("task_id", None)

st.title("全流程管理")

# ========== TASK DETAIL VIEW ==========
if task_id:
    # Fetch task details
    response = requests.get(f"{API_BASE}/api/workflow/tasks/{task_id}")

    if response.status_code != 200:
        st.error("任务不存在")
        if st.button("返回任务列表"):
            st.query_params.clear()
            st.rerun()
    else:
        task = response.json()

        # Header with back button
        col_back, col_title = st.columns([1, 5])
        with col_back:
            if st.button("←", key="back_btn", help="返回任务列表"):
                st.query_params.clear()
                st.rerun()
        with col_title:
            st.markdown(f"""
                <h1 style="font-size: 22px; font-weight: 600; color: #111827; margin: 0 0 8px 0;">
                    {task['name']}
                </h1>
            """, unsafe_allow_html=True)

        # Task metadata row (status, time, user, model)
        st.markdown(f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 20px;
                padding: 10px 0;
                border-bottom: 1px solid #e5e7eb;
                margin-bottom: 20px;
            ">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 13px; color: #6b7280;">状态</span>
                    {render_status_badge(task['status'])}
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 13px; color: #6b7280;">创建</span>
                    <span style="font-size: 13px; color: #374151;">{task['created_at'][:16] if task['created_at'] else '-'}</span>
                </div>
                {f'''<div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 13px; color: #6b7280;">用户</span>
                    <span style="font-size: 13px; color: #374151;">{task['user']}</span>
                </div>''' if task.get('user') else ''}
                {f'''<div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 13px; color: #6b7280;">模型</span>
                    <span style="font-size: 13px; color: #374151;">{task['model']}</span>
                </div>''' if task.get('model') else ''}
                {f'''<div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 13px; color: #6b7280;">Exp ID</span>
                    <span style="font-size: 13px; color: #374151;">{task['experiment_id']}</span>
                </div>''' if task.get('experiment_id') else ''}
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Configuration Form
        st.subheader("任务配置")

        with st.form("task_config_form"):
            # Model Configuration Section
            st.markdown("""
                <div style="
                    margin-top: 0;
                    margin-bottom: 16px;
                    padding-bottom: 8px;
                    border-bottom: 2px solid #e5e7eb;
                ">
                    <span style="
                        font-size: 15px;
                        font-weight: 600;
                        color: #1f2937;
                    ">模型配置</span>
                </div>
            """, unsafe_allow_html=True)

            # User and Model selection - primary fields
            col1, col2 = st.columns(2)
            with col1:
                default_user_idx = USER_OPTIONS.index(task['user']) if task['user'] in USER_OPTIONS else 0
                current_user = st.selectbox(
                    "用户名 *",
                    USER_OPTIONS,
                    index=default_user_idx,
                    key="detail_user",
                    help="选择任务所属用户",
                    label_visibility="visible"
                )

            with col2:
                user_config_dir = os.path.join(USER_CONFIG_DIR, current_user)
                models = get_models(user_config_dir)
                default_model_idx = models.index(task['model']) if task['model'] in models else 0
                selected_model = st.selectbox(
                    "选择模型 *",
                    models if models else ["无可用模型"],
                    index=default_model_idx if models else 0,
                    key="detail_model",
                    help="选择要运行的模型",
                    label_visibility="visible"
                )

            # Experiment ID - full width
            st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
            if models and selected_model != "无可用模型":
                available_expids = get_experiment_ids(selected_model, current_user)
                if available_expids:
                    default_exp_idx = available_expids.index(task['experiment_id']) if task['experiment_id'] in available_expids else 0
                    experiment_id = st.selectbox(
                        "Experiment ID *",
                        available_expids,
                        index=default_exp_idx,
                        key="detail_expid",
                        help="从配置文件中选择实验ID"
                    )
                else:
                    experiment_id = st.text_input(
                        "Experiment ID *",
                        value=task['experiment_id'] or selected_model.split('/')[-1] + "_test",
                        key="detail_expid_input",
                        help="未找到预配置的实验ID，请手动输入"
                    )
            else:
                experiment_id = st.text_input(
                    "Experiment ID *",
                    value=task['experiment_id'] or "",
                    key="detail_expid_input2",
                    help="请输入实验ID"
                )

            # SQL Configuration Section
            st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
            st.markdown("""
                <div style="
                    margin-bottom: 16px;
                    padding-bottom: 8px;
                    border-bottom: 2px solid #e5e7eb;
                ">
                    <span style="
                        font-size: 15px;
                        font-weight: 600;
                        color: #1f2937;
                    ">SQL 配置</span>
                </div>
            """, unsafe_allow_html=True)

            sql_col1, sql_col2 = st.columns(2)
            with sql_col1:
                sample_sql = st.text_area(
                    "样本数据 SQL",
                    value=task['sample_sql'] or "",
                    height=140,
                    key="detail_sample_sql",
                    help="从HDFS导出样本数据的SQL语句",
                    label_visibility="visible"
                )

            with sql_col2:
                infer_sql = st.text_area(
                    "推理数据 SQL",
                    value=task['infer_sql'] or "",
                    height=140,
                    key="detail_infer_sql",
                    help="从HDFS导出推理数据的SQL语句",
                    label_visibility="visible"
                )

            # Path Configuration Section
            st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
            st.markdown("""
                <div style="
                    margin-bottom: 16px;
                    padding-bottom: 8px;
                    border-bottom: 2px solid #e5e7eb;
                ">
                    <span style="
                        font-size: 14px;
                        font-weight: 600;
                        color: #6b7280;
                    ">路径配置</span>
                </div>
            """, unsafe_allow_html=True)

            path_col1, path_col2 = st.columns(2)
            with path_col1:
                hdfs_path = st.text_input(
                    "HDFS 路径",
                    value=task['hdfs_path'] or "/hdfs/data/",
                    key="detail_hdfs_path",
                    help="HDFS存储路径",
                    label_visibility="visible"
                )
            with path_col2:
                hive_table = st.text_input(
                    "Hive 表",
                    value=task['hive_table'] or "hive.result",
                    key="detail_hive_table",
                    help="目标Hive表名",
                    label_visibility="visible"
                )

            # Buttons
            col_save, col_run = st.columns(2)
            with col_save:
                save_submitted = st.form_submit_button("保存配置")
            with col_run:
                run_submitted = st.form_submit_button("保存并运行", type="primary")

            if save_submitted or run_submitted:
                # Update task via API (would need update endpoint)
                # For now, we'll create a new task execution
                payload = {
                    "name": task['name'],
                    "user": current_user,
                    "model": selected_model,
                    "experiment_id": experiment_id,
                    "sample_sql": sample_sql,
                    "infer_sql": infer_sql,
                    "hdfs_path": hdfs_path,
                    "hive_table": hive_table
                }

                if run_submitted:
                    # Create new execution
                    exec_response = requests.post(f"{API_BASE}/api/workflow/tasks", json=payload)
                    if exec_response.status_code == 200:
                        st.success(f"任务已启动! Execution ID: {exec_response.json()['task_id']}")
                        st.session_state["running_task_id"] = exec_response.json()['task_id']
                    else:
                        st.error(f"启动失败: {exec_response.text}")
                else:
                    st.info("配置已保存（功能开发中）")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Real-time logs section
        st.subheader("实时日志")

        # Show logs for current task or running task
        log_task_id = st.session_state.get("running_task_id", task_id)

        if st.button("刷新日志", key=f"refresh_{task_id}"):
            st.rerun()

        log_placeholder = st.empty()

        # Fetch task steps
        steps_response = requests.get(f"{API_BASE}/api/workflow/tasks/{log_task_id}/steps")
        if steps_response.status_code == 200:
            steps = steps_response.json()
            if steps:
                for step in steps:
                    # Get Chinese step name and status badge
                    step_name_cn = get_step_name_chinese(step['step_name'])
                    status_text = {
                        "pending": "待处理",
                        "running": "运行中",
                        "completed": "已完成",
                        "failed": "失败"
                    }.get(step['status'].lower(), step['status'].upper())

                    # Display with status icon and Chinese name
                    status_icon = {
                        "pending": "⏳",
                        "running": "🔄",
                        "completed": "✅",
                        "failed": "❌"
                    }.get(step['status'].lower(), "⏳")

                    with st.expander(f"{status_icon} {step_name_cn} ({status_text})"):
                        st.write(f"**开始时间:** {step.get('started_at', '未开始')}")
                        st.write(f"**完成时间:** {step.get('completed_at', '未完成')}")
                        if step.get('error_message'):
                            st.error(f"**错误:** {step['error_message']}")
        else:
            log_placeholder.info("暂无步骤信息，等待任务启动...")


# ========== TASK LIST VIEW ==========
else:
    # Header with create button
    col_header, col_create = st.columns([5, 1])

    with col_header:
        st.markdown('<h2 style="font-size: 18px; font-weight: 600; color: #1f2937; margin: 0;">任务列表</h2>', unsafe_allow_html=True)

    with col_create:
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        if st.button("+ 新建", key="create_task_btn", type="primary", use_container_width=True):
            st.session_state["show_create_form"] = not st.session_state.get("show_create_form", False)
            st.rerun()

    # Quick create task form (inline when triggered)
    if st.session_state.get("show_create_form", False):
        with st.container():
            st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
            with st.form("quick_create_task", clear_on_submit=True):
                col_name, col_submit = st.columns([5, 1])
                with col_name:
                    name = st.text_input("任务名称", placeholder="输入任务名称...", label_visibility="collapsed")
                with col_submit:
                    st.markdown('<div style="height: 26px;"></div>', unsafe_allow_html=True)
                    col_sub, col_can = st.columns(2)
                    with col_sub:
                        submitted = st.form_submit_button("创建", type="primary", use_container_width=True)
                    with col_can:
                        if st.form_submit_button("取消", use_container_width=True):
                            st.session_state["show_create_form"] = False
                            st.rerun()

                if submitted and name:
                    response = requests.post(f"{API_BASE}/api/workflow/tasks", json={
                        "name": name,
                        "user": "", "model": "", "experiment_id": "",
                        "sample_sql": "", "infer_sql": "",
                        "hdfs_path": "/hdfs/data/", "hive_table": "hive.result"
                    })
                    if response.status_code == 200:
                        st.session_state["show_create_form"] = False
                        st.rerun()
                    else:
                        st.error(f"创建失败: {response.text}")

    # Task list
    response = requests.get(f"{API_BASE}/api/workflow/tasks")

    if response.status_code == 200:
        tasks = response.json()

        if not tasks:
            st.markdown("""
                <div style="
                    text-align: center;
                    padding: 80px 20px;
                    background: linear-gradient(135deg, #f9fafb 0%, #ffffff 100%);
                    border-radius: 12px;
                    border: 1px solid #e5e7eb;
                    margin-top: 24px;
                ">
                    <div style="
                        width: 64px;
                        height: 64px;
                        margin: 0 auto 20px;
                        background: #f3f4f6;
                        border-radius: 16px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="1.5">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
                        </svg>
                    </div>
                    <h3 style="
                        color: #111827;
                        font-size: 16px;
                        font-weight: 600;
                        margin: 0 0 8px 0;
                    ">暂无任务</h3>
                    <p style="
                        color: #6b7280;
                        font-size: 14px;
                        margin: 0;
                    ">点击上方 "新建" 按钮创建第一个任务</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Modern compact task list
            for task in tasks:
                with st.container():
                    # Single row layout - more space for actions column
                    col_name, col_status, col_time, col_exp, col_actions = st.columns([3, 1.5, 1.5, 1.2, 2.0])

                    with col_name:
                        st.markdown(f"""
                            <div style="font-weight: 600; font-size: 14px; color: #1f2937;">{task['name']}</div>
                            <div style="font-size: 11px; color: #9ca3af; margin-top: 1px;">
                                {task.get('user', '')} {f"/ {task.get('model', '')}" if task.get('model') else ''}
                            </div>
                        """, unsafe_allow_html=True)

                    with col_status:
                        st.markdown(render_status_badge(task['status']), unsafe_allow_html=True)

                    with col_time:
                        st.markdown(f"""
                            <div style="font-size: 12px; color: #9ca3af;">
                                {task['created_at'][:16] if task['created_at'] else '-'}
                            </div>
                        """, unsafe_allow_html=True)

                    with col_exp:
                        if task.get('experiment_id'):
                            st.markdown(f"""
                                <div style="
                                    font-size: 11px;
                                    color: #6b7280;
                                    background: #f3f4f6;
                                    padding: 2px 8px;
                                    border-radius: 4px;
                                    text-align: center;
                                ">{task['experiment_id']}</div>
                            """, unsafe_allow_html=True)

                    with col_actions:
                        # Horizontal button layout with spacing
                        if st.session_state.get(f"confirm_delete_{task['id']}", False):
                            # Show confirm/cancel when delete is pending
                            c1, c2 = st.columns([1, 1])
                            with c1:
                                if st.button("✓", key=f"confirm_{task['id']}", type="primary", help="确认删除", use_container_width=True):
                                    response = requests.delete(f"{API_BASE}/api/workflow/tasks/{task['id']}")
                                    if response.status_code == 200:
                                        del st.session_state[f"confirm_delete_{task['id']}"]
                                        st.rerun()
                            with c2:
                                if st.button("✕", key=f"cancel_{task['id']}", help="取消", use_container_width=True):
                                    del st.session_state[f"confirm_delete_{task['id']}"]
                                    st.rerun()
                        else:
                            # Normal action buttons - horizontal with gap
                            bc1, bc2 = st.columns([1.5, 1])
                            with bc1:
                                if st.button("配置", key=f"config_{task['id']}", help="配置并运行任务", use_container_width=True):
                                    st.query_params["task_id"] = task['id']
                                    st.rerun()
                            with bc2:
                                if st.button("🗑", key=f"delete_{task['id']}", help="删除任务", use_container_width=True):
                                    st.session_state[f"confirm_delete_{task['id']}"] = True
                                    st.rerun()

                    # Divider
                    st.markdown('<div style="height: 1px; background-color: #f3f4f6; margin: 8px 0;"></div>', unsafe_allow_html=True)
    else:
        st.error("无法加载任务列表")
