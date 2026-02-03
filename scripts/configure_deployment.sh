#!/bin/bash
# =========================================================================
# FuxiCTR 部署配置向导
# =========================================================================
#
# 用途：交互式配置部署路径，自动生成所有配置文件和目录结构
#
# 使用方法：sudo bash configure_deployment.sh
#
# =========================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取当前用户名
REAL_USER=${SUDO_USER:-$USER}
echo -e "${BLUE}🚀 FuxiCTR 部署配置向导${NC}"
echo ""
echo -e "当前用户: ${GREEN}$REAL_USER${NC}"
echo ""

# ============================================================================
# 选择部署场景
# ============================================================================

echo -e "${YELLOW}请选择部署场景：${NC}"
echo ""
echo "1) 标准部署 (/opt/fuxictr + /data/fuxictr)"
echo "   ├── 代码: /opt/fuxictr"
echo "   ├── 虚拟环境: /opt/fuxictr_venv"
echo "   └── 数据: /data/fuxictr"
echo ""
echo "2) 单分区部署 (~/fuxictr)"
echo "   ├── 代码: ~/fuxictr"
echo "   ├── 虚拟环境: ~/fuxictr_venv"
echo "   └── 数据: ~/fuxictr_data"
echo ""
echo "3) 自定义路径"
echo "   完全自定义部署路径"
echo ""
read -p "请输入选择 [1-3]: " choice

case $choice in
    1)
        FUXICTR_ROOT="/opt/fuxictr"
        FUXICTR_VENV="/opt/fuxictr_venv"
        FUXICTR_STORAGE_BASE="/data/fuxictr"
        ;;
    2)
        FUXICTR_ROOT="$HOME/fuxictr"
        FUXICTR_VENV="$HOME/fuxictr_venv"
        FUXICTR_STORAGE_BASE="$HOME/fuxictr_data"
        ;;
    3)
        echo ""
        echo -e "${YELLOW}自定义路径配置${NC}"
        echo ""
        read -p "请输入代码目录 [如 /opt/fuxictr]: " FUXICTR_ROOT
        read -p "请输入虚拟环境路径 [如 /opt/fuxictr_venv]: " FUXICTR_VENV
        read -p "请输入数据基础目录 [如 /data/fuxictr]: " FUXICTR_STORAGE_BASE
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

# ============================================================================
# Server 21 配置
# ============================================================================

echo ""
echo -e "${YELLOW}Server 21 配置（数据源服务器）${NC}"
echo ""
read -p "Server 21 主机名或IP [如 21.xxxxxx.com]: " SERVER_21_HOST
read -p "SSH 用户名 [默认: $REAL_USER]: " SERVER_21_USER
SERVER_21_USER=${SERVER_21_USER:-$REAL_USER}
SERVER_21_PORT="22"

# ============================================================================
# 创建环境变量文件
# ============================================================================

ENV_FILE="$FUXICTR_ROOT/fuxictr_env.sh"

echo ""
echo -e "${BLUE}📝 生成环境变量文件: $ENV_FILE${NC}"

cat > "$ENV_FILE" <<EOF
#!/bin/bash
# =========================================================================
# FuxiCTR 部署环境变量配置
# =========================================================================
# 生成时间: $(date '+%Y-%m-%d %H:%M:%S')
# 部署场景: $choice

# ============================================================================
# 基础路径配置
# ============================================================================

export FUXICTR_ROOT="$FUXICTR_ROOT"
export FUXICTR_VENV="$FUXICTR_VENV"

# ============================================================================
# 数据存储路径（Server 142 - 训练服务器）
# ============================================================================

export FUXICTR_STORAGE_BASE="$FUXICTR_STORAGE_BASE"

# Dashboard 数据路径
export FUXICTR_DATA_ROOT="\${FUXICTR_STORAGE_BASE}/data"
export FUXICTR_PROCESSED_ROOT="\${FUXICTR_STORAGE_BASE}/processed_data"

# Workflow 数据路径
export FUXICTR_WORKFLOW_DATASETS="\${FUXICTR_STORAGE_BASE}/workflow_datasets"
export FUXICTR_WORKFLOW_PROCESSED="\${FUXICTR_STORAGE_BASE}/workflow_processed"
export FUXICTR_WORKFLOW_MODELS="\${FUXICTR_STORAGE_BASE}/workflow_models"
export FUXICTR_WORKFLOW_LOGS="\${FUXICTR_STORAGE_BASE}/workflow_logs"

# ============================================================================
# 日志路径
# ============================================================================

export FUXICTR_DASHBOARD_LOG_DIR="\${FUXICTR_ROOT}/dashboard/logs"
export FUXICTR_DB_BACKUP_DIR="\${FUXICTR_STORAGE_BASE}/db_backup"

# ============================================================================
# 配置文件路径
# ============================================================================

export FUXICTR_CONFIG_PATH="\${FUXICTR_ROOT}/fuxictr/workflow/config.yaml"

# ============================================================================
# Server 21 配置（数据源服务器）
# ============================================================================

export FUXICTR_SERVER_21_HOST="$SERVER_21_HOST"
export FUXICTR_SERVER_21_USER="$SERVER_21_USER"
export FUXICTR_SERVER_21_PORT="$SERVER_21_PORT"
export FUXICTR_SERVER_21_KEY_PATH="~/.ssh/id_rsa"
export FUXICTR_SERVER_21_STAGING="/tmp/fuxictr_staging"

# ============================================================================
# 服务端口配置
# ============================================================================

export FUXICTR_WORKFLOW_PORT="8001"
export FUXICTR_DASHBOARD_PORT="8501"

# ============================================================================
# 显示环境变量信息（加载时显示）
# ============================================================================

if [ -n "\$FUXICTR_ENV_LOADED" ]; then
    return 0  # 避免重复加载
fi

echo "✅ FuxiCTR 环境变量已加载"
echo ""
echo "📂 配置路径："
echo "   代码目录:     \$FUXICTR_ROOT"
echo "   虚拟环境:     \$FUXICTR_VENV"
echo "   数据存储:     \$FUXICTR_STORAGE_BASE"
echo ""
echo "🔌 服务端口："
echo "   Workflow:     \$FUXICTR_WORKFLOW_PORT"
echo "   Dashboard:    \$FUXICTR_DASHBOARD_PORT"
echo ""

export FUXICTR_ENV_LOADED=1
EOF

chmod +x "$ENV_FILE"
echo -e "${GREEN}✅ 环境变量文件已生成${NC}"

# ============================================================================
# 创建目录结构
# ============================================================================

echo ""
echo -e "${BLUE}📁 创建目录结构${NC}"

dirs=(
    "$FUXICTR_ROOT/data"
    "$FUXICTR_ROOT/processed_data"
    "$FUXICTR_STORAGE_BASE/workflow_datasets"
    "$FUXICTR_STORAGE_BASE/workflow_processed"
    "$FUXICTR_STORAGE_BASE/workflow_models"
    "$FUXICR_STORAGE_BASE/workflow_logs"
    "$FUXICTR_ROOT/dashboard/logs"
    "$FUXICTR_STORAGE_BASE/db_backup"
)

for dir in "${dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "  ${GREEN}✓${NC} 创建: $dir"
    else
        echo -e "  ${YELLOW}○${NC} 已存在: $dir"
    fi
done

# 设置权限
sudo chown -R $REAL_USER:$REAL_USER "$FUXICTR_ROOT"
sudo chown -R $REAL_USER:$REAL_USER "$FUXICTR_STORAGE_BASE"
echo ""
echo -e "${GREEN}✅ 目录权限已设置${NC}"

# ============================================================================
# 更新配置文件
# ============================================================================

CONFIG_FILE="$FUXICTR_ROOT/fuxictr/workflow/config.yaml"

if [ -f "$CONFIG_FILE" ]; then
    echo ""
    echo -e "${BLUE}📝 更新配置文件: $CONFIG_FILE${NC}"

    # 使用 sed 替换配置文件中的路径
    sed -i.bak "s|/opt/fuxictr|$FUXICTR_ROOT|g" "$CONFIG_FILE"
    sed -i.bak "s|/data/fuxictr/|$FUXICTR_STORAGE_BASE/|g" "$CONFIG_FILE"
    echo -e "${GREEN}✅ 配置文件已更新${NC}"
    echo -e "  原文件备份: ${CONFIG_FILE}.bak"
fi

# ============================================================================
# 更新 .bashrc
# ============================================================================

BASHRC="$HOME/.bashrc"
SOURCE_LINE="source $ENV_FILE"

if ! grep -q "$SOURCE_LINE" "$BASHRC" 2>/dev/null; then
    echo ""
    echo -e "${BLUE}📝 更新 ~/.bashrc${NC}"
    echo "" >> "$BASHRC"
    echo "# FuxiCTR 环境变量" >> "$BASHRC"
    echo "$SOURCE_LINE" >> "$BASHRC"
    echo -e "${GREEN}✅ 已添加环境变量加载命令到 ~/.bashrc${NC}"
    echo -e "${YELLOW}⚠️  请执行 'source ~/.bashrc' 使其生效${NC}"
else
    echo ""
    echo -e "${GREEN}✅ ~/.bashrc 已包含环境变量配置${NC}"
fi

# ============================================================================
# 生成 systemd 服务文件
# ============================================================================

echo ""
echo -e "${BLUE}📝 生成 systemd 服务文件${NC}"

# Workflow 服务
WORKFLOW_SERVICE="/etc/systemd/system/fuxictr-workflow.service"

cat > "$WORKFLOW_SERVICE" <<EOF
[Unit]
Description=FuxiCTR Workflow Service
After=network.target

[Service]
Type=simple
User=$REAL_USER
Group=$REAL_USER
WorkingDirectory=$FUXICTR_ROOT
Environment="PATH=$FUXICTR_VENV/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=$ENV_FILE
ExecStart=$FUXICTR_VENV/bin/python -m fuxictr.workflow.service
Restart=always
RestartSec=10
StandardOutput=append:$FUXICTR_STORAGE_BASE/workflow_logs/service.log
StandardError=append:$FUXICTR_STORAGE_BASE/workflow_logs/service.log

[Install]
WantedBy=multi-user.target
EOF

# Dashboard 服务
DASHBOARD_SERVICE="/etc/systemd/system/fuxictr-dashboard.service"

cat > "$DASHBOARD_SERVICE" <<EOF
[Unit]
Description=FuxiCTR Dashboard
After=network.target fuxictr-workflow.service

[Service]
Type=simple
User=$REAL_USER
Group=$REAL_USER
WorkingDirectory=$FUXICTR_ROOT
Environment="PATH=$FUXICTR_VENV/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=$ENV_FILE
ExecStart=$FUXICTR_VENV/bin/streamlit run dashboard/app.py \\
    --server.port=\${FUXICTR_DASHBOARD_PORT} \\
    --server.address 0.0.0.0 \\
    --browser.gatherUsageStats false
Restart=always
RestartSec=10
StandardOutput=append:$FUXICTR_ROOT/dashboard/logs/streamlit.log
StandardError=append:$FUXICTR_ROOT/dashboard/logs/streamlit.log

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}✅ systemd 服务文件已生成${NC}"
echo "  Workflow: $WORKFLOW_SERVICE"
echo "  Dashboard: $DASHBOARD_SERVICE"

# ============================================================================
# 重新加载 systemd
# ============================================================================

echo ""
echo -e "${BLUE}🔄 重新加载 systemd 配置${NC}"
systemctl daemon-reload
echo -e "${GREEN}✅ systemd 配置已重新加载${NC}"

# ============================================================================
# 完成
# ============================================================================

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🎉 部署配置完成！${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}📋 配置信息：${NC}"
echo "   代码目录:     $FUXICTR_ROOT"
echo "   虚拟环境:     $FUXICTR_VENV"
echo "   数据存储:     $FUXICTR_STORAGE_BASE"
echo ""
echo -e "${BLUE}🔌 Server 21（数据源）：${NC}"
echo "   主机:         $SERVER_21_HOST"
echo "   用户:         $SERVER_21_USER"
echo ""
echo -e "${BLUE}📌 下一步操作：${NC}"
echo ""
echo "1. 激活环境变量："
echo "   source ~/.bashrc"
echo ""
echo "2. 启动服务："
echo "   sudo systemctl start fuxictr-workflow"
echo "   sudo systemctl start fuxictr-dashboard"
echo ""
echo "3. 查看服务状态："
echo "   sudo systemctl status fuxictr-workflow"
echo "   sudo systemctl status fuxictr-dashboard"
echo ""
echo "4. 访问 Dashboard："
echo "   http://$(hostname -I | awk '{print $1}'):\${FUXICTR_DASHBOARD_PORT:-8501}"
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
