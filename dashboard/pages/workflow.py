import streamlit as st
import requests
import os
import yaml
from datetime import datetime
import json
import time

API_BASE = "http://localhost:8001"

# Load CSS styles
def load_css():
    """Load modern flat design CSS styles."""
    css_content = """
    <style>
    /* Modern Flat Design System */
    :root {
        --primary: #3b82f6;
        --primary-hover: #2563eb;
        --primary-light: #dbeafe;
        --success: #10b981;
        --success-light: #d1fae5;
        --warning: #f59e0b;
        --warning-light: #fef3c7;
        --danger: #ef4444;
        --danger-light: #fee2e2;
        --gray-50: #f8fafc;
        --gray-100: #f1f5f9;
        --gray-200: #e2e8f0;
        --gray-300: #cbd5e1;
        --gray-400: #94a3b8;
        --gray-500: #64748b;
        --gray-600: #475569;
        --gray-700: #334155;
        --gray-800: #1e293b;
        --gray-900: #0f172a;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --radius-sm: 6px;
        --radius: 8px;
        --radius-md: 10px;
        --radius-lg: 12px;
    }

    /* Global Typography */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        background: var(--gray-50);
    }

    /* Section Header */
    .section-header {
        padding: 20px 0;
        border-bottom: 1px solid var(--gray-200);
        margin-bottom: 24px;
    }

    /* Stats Bar */
    .stats-bar {
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
        padding: 16px 20px;
        background: white;
        border-radius: var(--radius-md);
        border: 1px solid var(--gray-200);
        box-shadow: var(--shadow-sm);
    }

    .stat-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: var(--gray-50);
        border-radius: var(--radius);
        border: 1px solid var(--gray-200);
    }

    .stat-value {
        font-size: 20px;
        font-weight: 700;
        color: var(--gray-900);
    }

    .stat-label {
        font-size: 13px;
        color: var(--gray-500);
        font-weight: 500;
    }

    /* Task Card */
    .task-card {
        background: white;
        border-radius: var(--radius-md);
        border: 1px solid var(--gray-200);
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }

    .task-card:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--gray-300);
        transform: translateY(-1px);
    }

    .task-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 16px;
    }

    .task-title {
        font-size: 16px;
        font-weight: 600;
        color: var(--gray-900);
        margin: 0;
    }

    .task-meta {
        font-size: 13px;
        color: var(--gray-500);
        margin-top: 4px;
    }

    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    .status-badge.pending {
        background: var(--gray-100);
        color: var(--gray-600);
    }

    .status-badge.running {
        background: var(--primary-light);
        color: var(--primary);
        animation: pulse 2s infinite;
    }

    .status-badge.completed {
        background: var(--success-light);
        color: var(--success);
    }

    .status-badge.failed {
        background: var(--danger-light);
        color: var(--danger);
    }

    .status-badge.cancelled {
        background: var(--warning-light);
        color: var(--warning);
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* Progress Bar */
    .progress-container {
        margin: 16px 0;
    }

    .progress-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .progress-label {
        font-size: 13px;
        font-weight: 500;
        color: var(--gray-600);
    }

    .progress-value {
        font-size: 13px;
        font-weight: 600;
        color: var(--gray-900);
    }

    .progress-bar {
        height: 8px;
        background: var(--gray-200);
        border-radius: 4px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary) 0%, #60a5fa 100%);
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    .progress-fill.success {
        background: linear-gradient(90deg, var(--success) 0%, #34d399 100%);
    }

    .progress-fill.error {
        background: linear-gradient(90deg, var(--danger) 0%, #f87171 100%);
    }

    /* Step Indicator */
    .step-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 16px 0;
        padding: 16px;
        background: var(--gray-50);
        border-radius: var(--radius);
        border: 1px solid var(--gray-200);
    }

    .step-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        font-weight: 500;
        color: var(--gray-500);
    }

    .step-item.active {
        color: var(--primary);
    }

    .step-item.completed {
        color: var(--success);
    }

    .step-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--gray-300);
    }

    .step-item.active .step-dot {
        background: var(--primary);
        box-shadow: 0 0 0 3px var(--primary-light);
    }

    .step-item.completed .step-dot {
        background: var(--success);
    }

    /* Action Buttons */
    .action-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        padding: 8px 16px;
        border-radius: var(--radius);
        font-size: 13px;
        font-weight: 500;
        border: 1px solid transparent;
        cursor: pointer;
        transition: all 0.15s ease;
    }

    .action-btn:hover {
        transform: translateY(-1px);
    }

    .action-btn.primary {
        background: var(--primary);
        color: white;
    }

    .action-btn.primary:hover {
        background: var(--primary-hover);
        box-shadow: var(--shadow-md);
    }

    .action-btn.secondary {
        background: white;
        color: var(--gray-700);
        border-color: var(--gray-300);
    }

    .action-btn.secondary:hover {
        background: var(--gray-50);
        border-color: var(--gray-400);
    }

    .action-btn.danger {
        background: white;
        color: var(--danger);
        border-color: var(--danger-light);
    }

    .action-btn.danger:hover {
        background: var(--danger-light);
    }

    .action-btn.ghost {
        background: transparent;
        color: var(--gray-500);
    }

    .action-btn.ghost:hover {
        background: var(--gray-100);
        color: var(--gray-700);
    }

    /* Modal/Dialog */
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(15, 23, 42, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
        backdrop-filter: blur(4px);
    }

    .modal-content {
        background: white;
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-lg);
        max-width: 800px;
        width: 90%;
        max-height: 90vh;
        overflow: hidden;
        animation: modalSlideIn 0.2s ease;
    }

    @keyframes modalSlideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .modal-header {
        padding: 20px 24px;
        border-bottom: 1px solid var(--gray-200);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .modal-title {
        font-size: 18px;
        font-weight: 600;
        color: var(--gray-900);
        margin: 0;
    }

    .modal-body {
        padding: 24px;
        overflow-y: auto;
        max-height: calc(90vh - 140px);
    }

    .modal-footer {
        padding: 16px 24px;
        border-top: 1px solid var(--gray-200);
        display: flex;
        justify-content: flex-end;
        gap: 12px;
        background: var(--gray-50);
    }

    /* Form Styles */
    .form-group {
        margin-bottom: 20px;
    }

    .form-label {
        display: block;
        font-size: 13px;
        font-weight: 600;
        color: var(--gray-700);
        margin-bottom: 6px;
    }

    .form-hint {
        font-size: 12px;
        color: var(--gray-500);
        margin-top: 4px;
    }

    .form-section {
        margin-bottom: 24px;
        padding: 20px;
        background: var(--gray-50);
        border-radius: var(--radius);
        border: 1px solid var(--gray-200);
    }

    .form-section-title {
        font-size: 14px;
        font-weight: 600;
        color: var(--gray-800);
        margin: 0 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--gray-200);
    }

    /* Log Panel */
    .log-panel {
        background: var(--gray-900);
        border-radius: var(--radius);
        padding: 16px;
        font-family: 'SF Mono', Monaco, monospace;
        font-size: 12px;
        line-height: 1.6;
        color: #e2e8f0;
        max-height: 400px;
        overflow-y: auto;
    }

    .log-entry {
        padding: 2px 0;
        border-left: 2px solid transparent;
        padding-left: 8px;
        margin: 2px 0;
    }

    .log-entry.info {
        border-left-color: var(--primary);
    }

    .log-entry.success {
        border-left-color: var(--success);
    }

    .log-entry.error {
        border-left-color: var(--danger);
    }

    .log-entry.warning {
        border-left-color: var(--warning);
    }

    .log-timestamp {
        color: var(--gray-500);
        margin-right: 8px;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    </style>
    """
    st.markdown(css_content, unsafe_allow_html=True)

load_css()

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

# Page Header with Modern Design
st.markdown("""
    <div style="padding: 20px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 24px;">
        <h1 style="font-size: 24px; font-weight: 600; color: #0f172a; margin: 0;">
            全流程管理
        </h1>
        <p style="font-size: 14px; color: #64748b; margin: 4px 0 0 0;">
            管理模型训练、推理和数据传输任务
        </p>
    </div>
""", unsafe_allow_html=True)


def render_websocket_logs(container, task_id: int, api_base: str, auto_refresh: bool = True):
    """
    Render a WebSocket-based real-time log viewer.

    This component uses JavaScript WebSocket API to receive real-time logs
    from the backend workflow service.
    """
    ws_url = f"ws://localhost:8001/api/workflow/tasks/{task_id}/logs"

    # HTML/JS component for WebSocket
    html_code = f"""
    <div id="log-container-{task_id}" style="
        background: #1e1e1e;
        border-radius: 8px;
        padding: 16px;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 13px;
        color: #d4d4d4;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #3e3e3e;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.3);
    ">
        <div id="log-header-{task_id}" style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #3e3e3e;
        ">
            <span style="color: #94a3b8; font-weight: 600;">日志输出</span>
            <span id="ws-status-{task_id}" style="
                padding: 4px 10px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
                background: #3e3e3e;
                color: #808080;
            ">等待连接</span>
        </div>
        <div id="log-content-{task_id}" style="
            line-height: 1.6;
        ">
            <div style="color: #808080; font-style: italic;">等待日志数据...</div>
        </div>
    </div>

    <script>
    (function() {{
        const wsUrl = "{ws_url}";
        const logContainer = document.getElementById('log-content-{task_id}');
        const statusElement = document.getElementById('ws-status-{task_id}');
        const mainContainer = document.getElementById('log-container-{task_id}');

        let ws = null;
        let reconnectTimeout = null;
        let messageCount = 0;
        const MAX_MESSAGES = 500; // Prevent DOM overflow

        // Color schemes for different log types
        const COLORS = {{
            'log': '#d4d4d4',
            'progress': '#4ec9b0',
            'metric': '#dcdcaa',
            'error': '#f14c4c',
            'complete': '#4ec9b0',
            'status': '#569cd6'
        }};

        const LEVEL_COLORS = {{
            'INFO': '#d4d4d4',
            'WARNING': '#dcdcaa',
            'ERROR': '#f14c4c',
            'DEBUG': '#808080'
        }};

        function formatTime(timestamp) {{
            return timestamp || new Date().toLocaleTimeString('zh-CN');
        }}

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        function addLogMessage(data) {{
            // Limit message count
            if (messageCount > MAX_MESSAGES) {{
                logContainer.innerHTML = '';
                messageCount = 0;
            }}

            const msgType = data.type || 'log';
            const step = data.step || 'system';
            const timestamp = formatTime(data.timestamp);

            let logLine = '';
            let color = COLORS[msgType] || COLORS['log'];

            if (msgType === 'log') {{
                const level = data.data?.level || 'INFO';
                const levelColor = LEVEL_COLORS[level] || LEVEL_COLORS['INFO'];
                const message = escapeHtml(data.data?.message || '');
                logLine = `<div style="color: {{color}};">
                    <span style="color: #808080;">[${{timestamp}}]</span>
                    <span style="color: #569cd6;">[{{step}}]</span>
                    <span style="color: ${{levelColor}};">[{{level}}]</span>
                    {{message}}
                </div>`;
            }} else if (msgType === 'progress') {{
                const current = data.data?.current || 0;
                const total = data.data?.total || 0;
                const percent = data.data?.percent || 0;
                const message = data.data?.message || '';
                logLine = `<div style="color: {{color}};">
                    <span style="color: #64748b;">[${{timestamp}}]</span>
                    <span style="color: #60a5fa;">[{{step}}]</span>
                    <span style="color: #4ec9b0;">进度: ${{current}}/${{total}} (${{percent}}%)</span>
                    ${{message ? ' - ' + escapeHtml(message) : ''}}
                </div>`;
            }} else if (msgType === 'metric') {{
                const name = data.data?.name || '';
                const value = data.data?.value || '';
                const unit = data.data?.unit || '';
                logLine = `<div style="color: {{color}};">
                    <span style="color: #64748b;">[${{timestamp}}]</span>
                    <span style="color: #60a5fa;">[{{step}}]</span>
                    <span style="color: #dcdcaa;">${{name}}: ${{value}}${{unit}}</span>
                </div>`;
            }} else if (msgType === 'error') {{
                const message = escapeHtml(data.data?.message || '');
                logLine = `<div style="color: {{color}};">
                    <span style="color: #64748b;">[${{timestamp}}]</span>
                    <span style="color: #60a5fa;">[{{step}}]</span>
                    <span style="color: #f14c4c;">错误: {{message}}</span>
                </div>`;
            }} else if (msgType === 'complete') {{
                logLine = `<div style="color: {{color}};">
                    <span style="color: #64748b;">[${{timestamp}}]</span>
                    <span style="color: #60a5fa;">[{{step}}]</span>
                    <span style="color: #4ec9b0;">完成</span>
                </div>`;
            }} else if (msgType === 'status') {{
                logLine = `<div style="color: {{color}};">
                    <span style="color: #808080;">[${{timestamp}}]</span>
                    <span style="color: #569cd6;">[状态更新]</span>
                    状态: ${{data.data?.status || 'unknown'}},
                    当前步骤: ${{data.data?.current_step || 0}}
                </div>`;
            }}

            logContainer.innerHTML += logLine;
            messageCount++;

            // Auto-scroll to bottom
            mainContainer.scrollTop = mainContainer.scrollHeight;
        }}

        function connect() {{
            if (ws) {{
                ws.close();
            }}

            ws = new WebSocket(wsUrl);

            ws.onopen = function() {{
                console.log('WebSocket connected');
                statusElement.textContent = '已连接';
                statusElement.style.background = '#4ec9b0';
                statusElement.style.color = '#1e1e1e';
                clearTimeout(reconnectTimeout);
            }};

            ws.onmessage = function(event) {{
                try {{
                    const data = JSON.parse(event.data);
                    addLogMessage(data);
                }} catch (e) {{
                    console.error('Failed to parse message:', e);
                }}
            }};

            ws.onerror = function(error) {{
                console.error('WebSocket error:', error);
                statusElement.textContent = '连接错误';
                statusElement.style.background = '#f14c4c';
            }};

            ws.onclose = function() {{
                console.log('WebSocket disconnected');
                statusElement.textContent = '已断开';
                statusElement.style.background = '#3e3e3e';
                statusElement.style.color = '#808080';

                // Reconnect after 3 seconds
                reconnectTimeout = setTimeout(connect, 3000);
            }};
        }}

        // Start connection
        connect();

        // Cleanup on page unload
        window.addEventListener('beforeunload', function() {{
            if (ws) ws.close();
            if (reconnectTimeout) clearTimeout(reconnectTimeout);
        }});
    }})();
    </script>
    """

    container.html(html_code)


# ========== TASK DETAIL VIEW ==========
if task_id:
    # Fetch task details
    response = requests.get(f"{API_BASE}/api/workflow/tasks/{task_id}")

    if response.status_code != 200:
        st.markdown("""
            <div style="
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 400px;
                background: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
            ">
                <h3 style="color: #1e293b; margin: 0 0 8px 0;">任务不存在</h3>
                <p style="color: #64748b; margin: 0 0 16px 0;">请检查任务ID是否正确</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("返回任务列表", use_container_width=True):
            st.query_params.clear()
            st.rerun()
    else:
        task = response.json()

        # ========== HEADER ==========
        st.markdown(f"""
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 24px;
                padding: 20px 24px;
                background: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div>
                    <button onclick="window.location.href='?'" style="
                        background: none;
                        border: none;
                        color: #64748b;
                        font-size: 14px;
                        cursor: pointer;
                        padding: 8px 0;
                        display: flex;
                        align-items: center;
                        gap: 6px;
                        margin-bottom: 12px;
                    ">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M19 12H5M12 19l-7-7 7-7"/>
                        </svg>
                        返回列表
                    </button>
                    <h1 style="font-size: 24px; font-weight: 600; color: #0f172a; margin: 0 0 4px 0;">
                        {task.get('name', '未命名任务')}
                    </h1>
                    <div style="font-size: 13px; color: #64748b; margin-top: 4px;">
                        创建于 {task.get('created_at', '')[:16] if task.get('created_at') else '-'}
                    </div>
                </div>
                {render_status_badge(task.get('status', 'unknown'))}
            </div>
        """, unsafe_allow_html=True)

        # ========== CONFIGURATION FORM ==========
        st.markdown("""
            <div style="
                padding: 24px;
                background: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                margin-bottom: 24px;
            ">
                <h3 style="font-size: 18px; font-weight: 600; color: #0f172a; margin: 0 0 20px 0;">
                    任务配置
                </h3>
        """, unsafe_allow_html=True)


        # Configuration Form

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
                default_user_idx = USER_OPTIONS.index(task.get('user')) if task.get('user') in USER_OPTIONS else 0
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
                default_model_idx = models.index(task.get('model')) if task.get('model') in models else 0
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
                    default_exp_idx = available_expids.index(task.get('experiment_id')) if task.get('experiment_id') in available_expids else 0
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
                        value=task.get('experiment_id') or selected_model.split('/')[-1] + "_test",
                        key="detail_expid_input",
                        help="未找到预配置的实验ID，请手动输入"
                    )
            else:
                experiment_id = st.text_input(
                    "Experiment ID *",
                    value=task.get('experiment_id') or "",
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
                    value=task.get('sample_sql') or "",
                    height=140,
                    key="detail_sample_sql",
                    help="从HDFS导出样本数据的SQL语句",
                    label_visibility="visible"
                )

            with sql_col2:
                infer_sql = st.text_area(
                    "推理数据 SQL",
                    value=task.get('infer_sql') or "",
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
                    value=task.get('hdfs_path') or "/hdfs/data/",
                    key="detail_hdfs_path",
                    help="HDFS存储路径",
                    label_visibility="visible"
                )
            with path_col2:
                hive_table = st.text_input(
                    "Hive 表",
                    value=task.get('hive_table') or "hive.result",
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
                    "name": task.get('name'),
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

        st.markdown('</div>', unsafe_allow_html=True)

        # ========== PROGRESS SECTION ==========
        st.markdown("""
            <div style="
                padding: 24px;
                background: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                margin-bottom: 24px;
            ">
                <h3 style="font-size: 18px; font-weight: 600; color: #0f172a; margin: 0 0 20px 0;">
                    执行进度
                </h3>
        """, unsafe_allow_html=True)

        # Fetch progress
        progress_response = requests.get(f"{API_BASE}/api/workflow/tasks/{task_id}/progress")
        if progress_response.status_code == 200:
            progress = progress_response.json()

            # Progress bar
            current_step = progress.get('current_step', 0)
            total_steps = progress.get('total_steps', 5)
            progress_percent = int((current_step / total_steps) * 100) if total_steps > 0 else 0

            st.markdown(f"""
                <div style="margin-bottom: 24px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-size: 13px; color: #64748b; font-weight: 500;">整体进度</span>
                        <span style="font-size: 14px; font-weight: 600; color: #0f172a;">{current_step}/{total_steps} 步骤</span>
                    </div>
                    <div style="
                        width: 100%;
                        height: 8px;
                        background: #e2e8f0;
                        border-radius: 6px;
                        overflow: hidden;
                    ">
                        <div style="
                            width: {progress_percent}%;
                            height: 100%;
                            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
                            border-radius: 6px;
                            transition: width 0.3s ease;
                        "></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Step progress - modern design without emoji
            steps_data = progress.get('steps', [])
            if steps_data:
                st.markdown('<div style="display: flex; gap: 8px; flex-wrap: wrap;">', unsafe_allow_html=True)

                for step_data in steps_data:
                    step_status = step_data.get('status', 'pending')
                    step_name = step_data.get('name', '')
                    step_name_cn = get_step_name_chinese(step_name)

                    # Determine colors based on status
                    if step_status == 'completed':
                        bg_color = "#dcfce7"
                        text_color = "#166534"
                        border_color = "#86efac"
                        status_text = "完成"
                    elif step_status == 'running':
                        bg_color = "#dbeafe"
                        text_color = "#1e40af"
                        border_color = "#93c5fd"
                        status_text = "运行中"
                    elif step_status == 'failed':
                        bg_color = "#fee2e2"
                        text_color = "#991b1b"
                        border_color = "#fca5a5"
                        status_text = "失败"
                    else:
                        bg_color = "#f1f5f9"
                        text_color = "#475569"
                        border_color = "#cbd5e1"
                        status_text = "等待"

                    st.markdown(f"""
                        <div style="
                            flex: 1;
                            min-width: 140px;
                            padding: 12px 16px;
                            background: {bg_color};
                            border: 1px solid {border_color};
                            border-radius: 8px;
                        ">
                            <div style="font-size: 12px; color: {text_color}; font-weight: 600; margin-bottom: 4px;">
                                {step_name_cn}
                            </div>
                            <div style="font-size: 11px; color: #64748b;">
                                {status_text}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ========== LOGS SECTION ==========
        st.markdown("""
            <div style="
                padding: 24px;
                background: #1e293b;
                border-radius: 12px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                    <h3 style="font-size: 18px; font-weight: 600; color: #f8fafc; margin: 0;">
                        实时日志
                    </h3>
                    <div style="display: flex; gap: 12px; align-items: center;">
        """, unsafe_allow_html=True)

        # Auto-refresh toggle
        auto_refresh = st.checkbox("自动刷新", value=True, key=f"auto_refresh_{task_id}")
        if st.button("刷新", key=f"refresh_{task_id}"):
            st.rerun()

        st.markdown("</div></div>", unsafe_allow_html=True)

        # WebSocket Real-time Log Component
        ws_logs_placeholder = st.empty()

        # Render WebSocket log viewer
        render_websocket_logs(ws_logs_placeholder, task_id, API_BASE, auto_refresh)


# ========== TASK LIST VIEW ==========
else:
    # Fetch tasks for stats
    stats_response = requests.get(f"{API_BASE}/api/workflow/tasks")
    all_tasks = stats_response.json() if stats_response.status_code == 200 else []
    running_count = sum(1 for t in all_tasks if t.get('status') == 'running')
    pending_count = sum(1 for t in all_tasks if t.get('status') == 'pending')
    completed_count = sum(1 for t in all_tasks if t.get('status') == 'completed')
    failed_count = sum(1 for t in all_tasks if t.get('status') == 'failed')

    # Stats Bar
    st.markdown(f"""
        <div style="display: flex; gap: 12px; margin-bottom: 20px;">
            <div style="flex: 1; padding: 16px; background: white; border-radius: 10px; border: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                <div style="font-size: 24px; font-weight: 700; color: #3b82f6;">{running_count}</div>
                <div style="font-size: 12px; color: #64748b; font-weight: 500;">运行中</div>
            </div>
            <div style="flex: 1; padding: 16px; background: white; border-radius: 10px; border: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                <div style="font-size: 24px; font-weight: 700; color: #f59e0b;">{pending_count}</div>
                <div style="font-size: 12px; color: #64748b; font-weight: 500;">待处理</div>
            </div>
            <div style="flex: 1; padding: 16px; background: white; border-radius: 10px; border: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                <div style="font-size: 24px; font-weight: 700; color: #10b981;">{completed_count}</div>
                <div style="font-size: 12px; color: #64748b; font-weight: 500;">已完成</div>
            </div>
            <div style="flex: 1; padding: 16px; background: white; border-radius: 10px; border: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                <div style="font-size: 24px; font-weight: 700; color: #ef4444;">{failed_count}</div>
                <div style="font-size: 12px; color: #64748b; font-weight: 500;">失败</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Header with create button
    col_header, col_create = st.columns([5, 1])

    with col_header:
        st.markdown('<h2 style="font-size: 18px; font-weight: 600; color: #0f172a; margin: 0;">任务列表</h2>', unsafe_allow_html=True)

    with col_create:
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        if st.button("+ 新建任务", key="create_task_btn", type="primary", use_container_width=True):
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
                            <div style="font-weight: 600; font-size: 14px; color: #1f2937;">{task.get('name', '未命名任务')}</div>
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
                                ">{task.get('experiment_id')}</div>
                            """, unsafe_allow_html=True)

                    with col_actions:
                        # Horizontal button layout with spacing
                        action_col1, action_col2, action_col3 = st.columns([1, 1, 1])

                        if st.session_state.get(f"confirm_delete_{task['task_id']}", False):
                            # Show confirm/cancel when delete is pending
                            c1, c2, c3 = st.columns([1, 1, 1])
                            with c1:
                                if st.button("✓", key=f"confirm_{task['task_id']}", type="primary", help="确认删除", use_container_width=True):
                                    response = requests.delete(f"{API_BASE}/api/workflow/tasks/{task['task_id']}")
                                    if response.status_code == 200:
                                        del st.session_state[f"confirm_delete_{task['task_id']}"]
                                        st.rerun()
                            with c2:
                                if st.button("✕", key=f"cancel_{task['task_id']}", help="取消", use_container_width=True):
                                    del st.session_state[f"confirm_delete_{task['task_id']}"]
                                    st.rerun()
                            with c3:
                                st.write("")
                        else:
                            # Normal action buttons
                            task_status = task.get('status', '').lower()

                            # Config button - always available
                            with action_col1:
                                if st.button("配置", key=f"config_{task['task_id']}", help="配置并运行任务", use_container_width=True):
                                    st.query_params["task_id"] = task['task_id']
                                    st.rerun()

                            # Retry button - for failed or cancelled tasks
                            with action_col2:
                                if task_status in ['failed', 'cancelled']:
                                    if st.button("重试", key=f"retry_{task['task_id']}", help="重新执行任务", use_container_width=True):
                                        response = requests.post(f"{API_BASE}/api/workflow/tasks/{task['task_id']}/retry")
                                        if response.status_code == 200:
                                            st.success(f"任务 {task['task_id']} 已重新启动")
                                            st.rerun()
                                        else:
                                            st.error(f"重试失败: {response.text}")
                                elif task_status == 'running':
                                    # Cancel button for running tasks
                                    if st.button("取消", key=f"cancel_run_{task['task_id']}", help="取消运行中的任务", use_container_width=True):
                                        response = requests.post(f"{API_BASE}/api/workflow/tasks/{task['task_id']}/cancel")
                                        if response.status_code == 200:
                                            st.info(f"任务 {task['task_id']} 取消请求已发送")
                                            st.rerun()
                                        else:
                                            st.error(f"取消失败: {response.text}")
                                else:
                                    st.write("")

                            # Delete button
                            with action_col3:
                                if st.button("🗑", key=f"delete_{task['task_id']}", help="删除任务", use_container_width=True):
                                    st.session_state[f"confirm_delete_{task['task_id']}"] = True
                                    st.rerun()

                    # Divider
                    st.markdown('<div style="height: 1px; background-color: #f3f4f6; margin: 8px 0;"></div>', unsafe_allow_html=True)
    else:
        st.error("无法加载任务列表")
