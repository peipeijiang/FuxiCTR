# FuxiCTR Workflow 界面实现指南

## 概述

本指南描述如何将设计规范应用到现有的 `workflow.py` 文件中，实现现代化的扁平化界面设计。

## 文件结构

```
dashboard/
├── design-system/
│   ├── workflow-design-spec.md      # 设计规范文档
│   ├── workflow-layout.md           # 布局结构建议
│   ├── workflow-components.md       # 组件设计
│   ├── workflow-styles.css          # CSS 样式文件
│   └── workflow-implementation-guide.md  # 本文件
└── pages/
    └── workflow.py                  # 主界面文件（需要修改）
```

## 实施步骤

### 步骤 1: 引入 CSS 样式

在 `workflow.py` 的顶部添加 CSS 引入代码：

```python
import streamlit as st
import requests
import os
import yaml
from datetime import datetime
import json
import time

# 读取并注入 CSS 样式
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "..", "design-system", "workflow-styles.css")
    if os.path.exists(css_file):
        with open(css_file, 'r') as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

load_css()
```

### 步骤 2: 修改页面标题区域

**当前代码:**
```python
st.title("全流程管理")
```

**新设计:**
```python
# Page Header
st.markdown("""
    <div class="section-header" style="margin-bottom: 24px;">
        <h1 style="font-size: 24px; font-weight: 600; color: #0f172a; margin: 0;">
            全流程管理
        </h1>
    </div>
""", unsafe_allow_html=True)
```

### 步骤 3: 修改任务列表头部

**当前代码:**
```python
col_header, col_create = st.columns([5, 1])

with col_header:
    st.markdown('<h2 style="font-size: 18px; font-weight: 600; color: #1f2937; margin: 0;">任务列表</h2>', unsafe_allow_html=True)

with col_create:
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    if st.button("+ 新建", key="create_task_btn", type="primary", use_container_width=True):
        st.session_state["show_create_form"] = not st.session_state.get("show_create_form", False)
        st.rerun()
```

**新设计:**
```python
# Task List Header with Stats
tasks_response = requests.get(f"{API_BASE}/api/workflow/tasks")
if tasks_response.status_code == 200:
    all_tasks = tasks_response.json()
    running_count = sum(1 for t in all_tasks if t.get('status') == 'running')
    pending_count = sum(1 for t in all_tasks if t.get('status') == 'pending')
    completed_count = sum(1 for t in all_tasks if t.get('status') == 'completed')
    failed_count = sum(1 for t in all_tasks if t.get('status') == 'failed')
else:
    all_tasks = []
    running_count = pending_count = completed_count = failed_count = 0

# Stats Bar
st.markdown(f"""
    <div style="display: flex; gap: 16px; margin-bottom: 20px; padding: 16px 20px;
                background: white; border-radius: 10px; border: 1px solid #e2e8f0;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="width: 8px; height: 8px; background: #3b82f6; border-radius: 50%;"></span>
            <span style="font-size: 13px; color: #64748b;">运行中</span>
            <span style="font-size: 15px; font-weight: 600; color: #0f172a;">{running_count}</span>
        </div>
        <div style="width: 1px; background: #e2e8f0;"></div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="width: 8px; height: 8px; background: #f59e0b; border-radius: 50%;"></span>
            <span style="font-size: 13px; color: #64748b;">待处理</span>
            <span style="font-size: 15px; font-weight: 600; color: #0f172a;">{pending_count}</span>
        </div>
        <div style="width: 1px; background: #e2e8f0;"></div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="width: 8px; height: 8px; background: #10b981; border-radius: 50%;"></span>
            <span style="font-size: 13px; color: #64748b;">已完成</span>
            <span style="font-size: 15px; font-weight: 600; color: #0f172a;">{completed_count}</span>
        </div>
        <div style="width: 1px; background: #e2e8f0;"></div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="width: 8px; height: 8px; background: #ef4444; border-radius: 50%;"></span>
            <span style="font-size: 13px; color: #64748b;">失败</span>
            <span style="font-size: 15px; font-weight: 600; color: #0f172a;">{failed_count}</span>
        </div>
        <div style="flex: 1;"></div>
        <button onclick="window.parent.document.querySelector('button[kind=primary]').click()"
                style="padding: 8px 16px; background: linear-gradient(135deg, #3b82f6, #2563eb);
                       color: white; border: none; border-radius: 6px; font-size: 14px;
                       font-weight: 500; cursor: pointer;">+ 新建任务</button>
    </div>
""", unsafe_allow_html=True)

# Hidden button for functionality
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("+ 新建任务", key="create_task_btn", type="primary", use_container_width=True):
        st.session_state["show_create_form"] = not st.session_state.get("show_create_form", False)
        st.rerun()
```

### 步骤 4: 修改任务卡片渲染

**当前代码:**
```python
for task in tasks:
    with st.container():
        col_name, col_status, col_time, col_exp, col_actions = st.columns([3, 1.5, 1.5, 1.2, 2.0])
        # ... 列内容
```

**新设计:**
```python
def render_task_card(task):
    """Render a modern task card."""
    status = task.get('status', 'pending').lower()
    status_config = {
        'pending': {'icon': '⏳', 'class': 'pending', 'text': '待处理'},
        'running': {'icon': '🔄', 'class': 'running', 'text': '运行中'},
        'completed': {'icon': '✅', 'class': 'completed', 'text': '已完成'},
        'failed': {'icon': '❌', 'class': 'failed', 'text': '失败'},
        'cancelled': {'icon': '⭕', 'class': 'cancelled', 'text': '已取消'}
    }
    config = status_config.get(status, status_config['pending'])

    # Calculate progress
    progress = task.get('progress', 0)

    card_html = f"""
    <div class="task-card" style="margin-bottom: 12px;">
        <div class="task-card-status">
            <div class="status-icon {config['class']}">{config['icon']}</div>
            <span class="status-text">{config['text']}</span>
        </div>
        <div class="task-card-content">
            <div class="task-card-header">
                <h3 class="task-name">{task['name']}</h3>
            </div>
            <div class="task-card-meta">
                <span>{task.get('user', '')}</span>
                <span class="meta-separator">/</span>
                <span>{task.get('model', '')}</span>
            </div>
            {f'''
            <div class="task-progress">
                <div class="progress-bar" style="flex: 1;">
                    <div class="progress-fill" style="width: {progress}%"></div>
                </div>
                <span class="progress-text">{progress}%</span>
            </div>
            ''' if status == 'running' else ''}
            <div class="task-card-footer">
                <div class="task-meta-info">
                    <span>{task['created_at'][:16] if task.get('created_at') else '-'}</span>
                    {f'<span style="background: #f1f5f9; padding: 2px 8px; border-radius: 4px;">{task.get("experiment_id", "")}</span>' if task.get('experiment_id') else ''}
                </div>
            </div>
        </div>
    </div>
    """
    return card_html

# Render task list
for task in tasks:
    st.markdown(render_task_card(task), unsafe_allow_html=True)

    # Action buttons row
    cols = st.columns([1, 1, 1, 4])
    with cols[0]:
        if st.button("配置", key=f"config_{task['task_id']}", use_container_width=True):
            st.query_params["task_id"] = task['task_id']
            st.rerun()
    with cols[1]:
        task_status = task.get('status', '').lower()
        if task_status in ['failed', 'cancelled']:
            if st.button("重试", key=f"retry_{task['task_id']}", use_container_width=True):
                response = requests.post(f"{API_BASE}/api/workflow/tasks/{task['task_id']}/retry")
                if response.status_code == 200:
                    st.success(f"任务 {task['task_id']} 已重新启动")
                    st.rerun()
        elif task_status == 'running':
            if st.button("取消", key=f"cancel_run_{task['task_id']}", use_container_width=True):
                response = requests.post(f"{API_BASE}/api/workflow/tasks/{task['task_id']}/cancel")
                if response.status_code == 200:
                    st.info(f"任务 {task['task_id']} 取消请求已发送")
                    st.rerun()
    with cols[2]:
        if st.button("🗑", key=f"delete_{task['task_id']}", use_container_width=True):
            st.session_state[f"confirm_delete_{task['task_id']}"] = True
            st.rerun()
```

### 步骤 5: 修改任务详情页进度展示

**当前代码:**
```python
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Progress Metrics Section
st.subheader("执行进度")
```

**新设计:**
```python
# Progress Section - 放在最前面，突出显示
st.markdown("""
    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
            <h3 style="font-size: 16px; font-weight: 600; color: #0f172a; margin: 0;">执行进度</h3>
            <span class="badge badge-{status}">{status_text}</span>
        </div>
""", unsafe_allow_html=True)

# Progress bar
progress_response = requests.get(f"{API_BASE}/api/workflow/tasks/{task_id}/progress")
if progress_response.status_code == 200:
    progress = progress_response.json()
    current_step = progress.get('current_step', 0)
    total_steps = progress.get('total_steps', 5)
    progress_percent = int((current_step / total_steps) * 100) if total_steps > 0 else 0

    st.markdown(f"""
        <div style="margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-size: 13px; color: #64748b;">整体进度</span>
                <span style="font-size: 13px; font-weight: 600; color: #0f172a;">{current_step}/{total_steps} 步骤</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress_percent}%"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Step indicators
    steps_data = progress.get('steps', [])
    if steps_data:
        step_html = '<div class="step-progress" style="padding: 0;">'
        for i, step in enumerate(steps_data):
            step_status = step.get('status', 'pending')
            step_name = get_step_name_chinese(step.get('name', ''))

            if step_status == 'completed':
                icon = '✓'
                step_class = 'completed'
            elif step_status == 'running':
                icon = '🔄'
                step_class = 'running'
            elif step_status == 'failed':
                icon = '✗'
                step_class = 'failed'
            else:
                icon = str(i + 1)
                step_class = 'pending'

            step_html += f'''
                <div class="step-item {step_class}">
                    <div class="step-circle">{icon}</div>
                    <span class="step-label">{step_name}</span>
                    <span class="step-status">{step.get("started_at", "未开始")[:16] if step.get("started_at") else "未开始"}</span>
                </div>
            '''
            if i < len(steps_data) - 1:
                connector_class = 'completed' if step_status == 'completed' else 'active' if step_status == 'running' else ''
                step_html += f'<div class="step-connector {connector_class}"></div>'

        step_html += '</div>'
        st.markdown(step_html, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
```

### 步骤 6: 修改配置表单布局

**当前代码:**
```python
with st.form("task_config_form"):
    st.markdown("""
        <div style="margin-top: 0; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid #e5e7eb;">
            <span style="font-size: 15px; font-weight: 600; color: #1f2937;">模型配置</span>
        </div>
    """, unsafe_allow_html=True)
    # ... 表单字段
```

**新设计:**
```python
st.markdown("""
    <div class="form-section">
        <div class="form-section-header">
            <span class="form-section-icon">📋</span>
            <h3 class="form-section-title">模型配置</h3>
        </div>
""", unsafe_allow_html=True)

with st.form("task_config_form"):
    # Model Configuration
    col1, col2 = st.columns(2)
    with col1:
        default_user_idx = USER_OPTIONS.index(task['user']) if task['user'] in USER_OPTIONS else 0
        current_user = st.selectbox(
            "用户名 *",
            USER_OPTIONS,
            index=default_user_idx,
            key="detail_user",
            help="选择任务所属用户"
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
            help="选择要运行的模型"
        )

    # Experiment ID
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

    st.markdown('</div>', unsafe_allow_html=True)

    # SQL Configuration
    st.markdown("""
        <div class="form-section" style="margin-top: 24px;">
            <div class="form-section-header">
                <span class="form-section-icon">🗄️</span>
                <h3 class="form-section-title">SQL 配置</h3>
            </div>
    """, unsafe_allow_html=True)

    sql_col1, sql_col2 = st.columns(2)
    with sql_col1:
        sample_sql = st.text_area(
            "样本数据 SQL",
            value=task['sample_sql'] or "",
            height=140,
            key="detail_sample_sql",
            help="从HDFS导出样本数据的SQL语句"
        )
    with sql_col2:
        infer_sql = st.text_area(
            "推理数据 SQL",
            value=task['infer_sql'] or "",
            height=140,
            key="detail_infer_sql",
            help="从HDFS导出推理数据的SQL语句"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Path Configuration
    st.markdown("""
        <div class="form-section" style="margin-top: 24px;">
            <div class="form-section-header">
                <span class="form-section-icon">📁</span>
                <h3 class="form-section-title">路径配置</h3>
            </div>
    """, unsafe_allow_html=True)

    path_col1, path_col2 = st.columns(2)
    with path_col1:
        hdfs_path = st.text_input(
            "HDFS 路径",
            value=task['hdfs_path'] or "/hdfs/data/",
            key="detail_hdfs_path",
            help="HDFS存储路径"
        )
    with path_col2:
        hive_table = st.text_input(
            "Hive 表",
            value=task['hive_table'] or "hive.result",
            key="detail_hive_table",
            help="目标Hive表名"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Form Actions
    col_save, col_spacer, col_run = st.columns([1, 3, 1])
    with col_save:
        save_submitted = st.form_submit_button("💾 保存配置", use_container_width=True)
    with col_run:
        run_submitted = st.form_submit_button("▶ 保存并运行", type="primary", use_container_width=True)

    if save_submitted or run_submitted:
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
            exec_response = requests.post(f"{API_BASE}/api/workflow/tasks", json=payload)
            if exec_response.status_code == 200:
                st.success(f"任务已启动! Execution ID: {exec_response.json()['task_id']}")
                st.session_state["running_task_id"] = exec_response.json()['task_id']
            else:
                st.error(f"启动失败: {exec_response.text}")
        else:
            st.info("配置已保存（功能开发中）")
```

### 步骤 7: 修改状态徽章渲染函数

**当前代码:**
```python
def render_status_badge(status: str) -> str:
    status_styles = {
        "pending": "background-color: #f3f4f6; color: #4b5563; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;",
        "running": "background-color: #eff6ff; color: #2563eb; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;",
        # ...
    }
```

**新设计:**
```python
def render_status_badge(status: str) -> str:
    """Render modern status badge with CSS classes."""
    status_classes = {
        "pending": "badge-pending",
        "running": "badge-running",
        "completed": "badge-completed",
        "failed": "badge-failed",
        "cancelled": "badge-cancelled"
    }

    status_text = {
        "pending": "待处理",
        "running": "运行中",
        "completed": "已完成",
        "failed": "失败",
        "cancelled": "已取消"
    }

    badge_class = status_classes.get(status.lower(), "badge-pending")
    text = status_text.get(status.lower(), status.upper())

    return f'<span class="badge {badge_class}">{text}</span>'
```

## 完整修改后的 workflow.py 结构

```python
import streamlit as st
import requests
import os
import yaml
from datetime import datetime
import json
import time

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

# Load CSS
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "..", "design-system", "workflow-styles.css")
    if os.path.exists(css_file):
        with open(css_file, 'r') as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

load_css()

# ... (helper functions)

def render_status_badge(status: str) -> str:
    """Render modern status badge."""
    # ... implementation

def render_task_card(task):
    """Render modern task card."""
    # ... implementation

# Get query params
query_params = st.query_params
task_id = query_params.get("task_id", None)

# Page Header
st.markdown("""
    <div class="section-header" style="margin-bottom: 24px;">
        <h1 style="font-size: 24px; font-weight: 600; color: #0f172a; margin: 0;">
            全流程管理
        </h1>
    </div>
""", unsafe_allow_html=True)

# ========== TASK DETAIL VIEW ==========
if task_id:
    # ... task detail implementation with new design

# ========== TASK LIST VIEW ==========
else:
    # ... task list implementation with new design
```

## 注意事项

1. **CSS 优先级**: Streamlit 的默认样式可能会覆盖一些自定义样式，需要使用 `!important` 或更具体的选择器
2. **响应式设计**: 确保在移动设备上也能正常显示
3. **性能**: 大量任务列表时，考虑使用虚拟滚动或分页
4. **兼容性**: 测试不同浏览器的兼容性

## 后续优化建议

1. 添加暗黑模式支持
2. 实现任务卡片的拖拽排序
3. 添加更多动画效果
4. 实现实时通知系统
5. 添加键盘快捷键支持
