#!/bin/bash
# 一键配置 FuxiCTR systemd 服务
# 使用方法: sudo bash setup_systemd_services.sh

set -e

# 获取当前用户名
USERNAME=$(whoami)
echo "🔧 当前用户: $USERNAME"

# 检查是否为 root
if [ "$EUID" -ne 0 ]; then
    echo "❌ 请使用 sudo 运行此脚本"
    echo "   sudo bash $0"
    exit 1
fi

# 配置路径
FUXICTR_DIR="/opt/fuxictr"
VENV_DIR="/opt/fuxictr_venv"
CONFIG_FILE="$FUXICTR_DIR/fuxictr/workflow/config.yaml"

# 检查目录是否存在
if [ ! -d "$FUXICTR_DIR" ]; then
    echo "❌ FuxiCTR 目录不存在: $FUXICTR_DIR"
    exit 1
fi

echo "✅ FuxiCTR 目录: $FUXICTR_DIR"

# 创建 Workflow 服务
echo ""
echo "📝 创建 fuxictr-workflow 服务..."
cat > /etc/systemd/system/fuxictr-workflow.service <<EOF
[Unit]
Description=FuxiCTR Workflow Service
After=network.target

[Service]
Type=simple
User=$USERNAME
Group=$USERNAME
WorkingDirectory=$FUXICTR_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin"
Environment="WORKFLOW_CONFIG_PATH=$CONFIG_FILE"
ExecStart=$VENV_DIR/bin/python -m fuxictr.workflow.service
Restart=always
RestartSec=10
StandardOutput=append:/data/fuxictr/workflow_logs/service.log
StandardError=append:/data/fuxictr/workflow_logs/service.log

[Install]
WantedBy=multi-user.target
EOF
echo "✅ Created: /etc/systemd/system/fuxictr-workflow.service"

# 创建 Dashboard 服务
echo ""
echo "📝 创建 fuxictr-dashboard 服务..."
cat > /etc/systemd/system/fuxictr-dashboard.service <<EOF
[Unit]
Description=FuxiCTR Dashboard
After=network.target fuxictr-workflow.service

[Service]
Type=simple
User=$USERNAME
Group=$USERNAME
WorkingDirectory=$FUXICTR_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$VENV_DIR/bin/streamlit run dashboard/app.py \\
    --server.port 8501 \\
    --server.address 0.0.0.0 \\
    --browser.gatherUsageStats false
Restart=always
RestartSec=10
StandardOutput=append:/opt/fuxictr/dashboard/logs/streamlit.log
StandardError=append:/opt/fuxictr/dashboard/logs/streamlit.log

[Install]
WantedBy=multi-user.target
EOF
echo "✅ Created: /etc/systemd/system/fuxictr-dashboard.service"

# 重新加载 systemd
echo ""
echo "🔄 重新加载 systemd 配置..."
systemctl daemon-reload
echo "✅ systemd 配置已重新加载"

# 启用服务
echo ""
echo "🔌 启用服务（开机自启）..."
systemctl enable fuxictr-workflow
systemctl enable fuxictr-dashboard
echo "✅ 服务已启用"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "🎉 systemd 服务配置完成！"
echo ""
echo "📌 现在可以使用以下命令："
echo ""
echo "   启动服务："
echo "     sudo systemctl start fuxictr-workflow"
echo "     sudo systemctl start fuxictr-dashboard"
echo ""
echo "   查看状态："
echo "     sudo systemctl status fuxictr-workflow"
echo "     sudo systemctl status fuxictr-dashboard"
echo ""
echo "   立即启动："
echo "     sudo systemctl start fuxictr-workflow"
echo "     sudo systemctl start fuxictr-dashboard"
echo ""
echo "   访问地址："
echo "     http://$(hostname -I | awk '{print $1}'):8501"
echo "═══════════════════════════════════════════════════════"
