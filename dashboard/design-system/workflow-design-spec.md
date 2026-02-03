# FuxiCTR Workflow 设计规范文档

## 1. 设计概述

### 1.1 设计理念
- **现代扁平化**: 简洁、干净、无多余装饰
- **清晰层次**: 重要信息突出，次要信息弱化
- **高效操作**: 减少用户操作路径，提高效率
- **视觉引导**: 通过色彩和排版引导用户视线

### 1.2 设计原则
1. **一致性**: 统一的设计语言贯穿整个界面
2. **可读性**: 清晰的信息层级和足够的对比度
3. **反馈性**: 操作有即时反馈，状态清晰可见
4. **容错性**: 关键操作有确认机制，防止误操作

---

## 2. 配色方案

### 2.1 主色调
```css
--color-primary-50: #eff6ff;   /* 最浅 - 背景高亮 */
--color-primary-100: #dbeafe;  /* 浅 - hover背景 */
--color-primary-200: #bfdbfe;  /* 较浅 - 边框 */
--color-primary-300: #93c5fd;  /* 浅中 - 禁用状态 */
--color-primary-400: #60a5fa;  /* 中 - 次要元素 */
--color-primary-500: #3b82f6;  /* 主色 - 按钮、链接 */
--color-primary-600: #2563eb;  /* 深 - hover状态 */
--color-primary-700: #1d4ed8;  /* 较深 - active状态 */
--color-primary-800: #1e40af;  /* 深 - 文字 */
--color-primary-900: #1e3a8a;  /* 最深 - 标题 */
```

### 2.2 辅助色
```css
/* 成功 - 绿色系 */
--color-success-50: #ecfdf5;
--color-success-100: #d1fae5;
--color-success-500: #10b981;
--color-success-600: #059669;
--color-success-700: #047857;

/* 警告 - 橙色系 */
--color-warning-50: #fffbeb;
--color-warning-100: #fef3c7;
--color-warning-500: #f59e0b;
--color-warning-600: #d97706;
--color-warning-700: #b45309;

/* 错误 - 红色系 */
--color-danger-50: #fef2f2;
--color-danger-100: #fee2e2;
--color-danger-500: #ef4444;
--color-danger-600: #dc2626;
--color-danger-700: #b91c1c;

/* 信息 - 蓝色系 */
--color-info-50: #eff6ff;
--color-info-100: #dbeafe;
--color-info-500: #3b82f6;
--color-info-600: #2563eb;
--color-info-700: #1d4ed8;
```

### 2.3 中性色
```css
/* 背景色 */
--color-bg-primary: #ffffff;       /* 主背景 */
--color-bg-secondary: #f8fafc;     /* 次级背景 */
--color-bg-tertiary: #f1f5f9;      /* 三级背景 */
--color-bg-hover: #f8fafc;         /* hover背景 */
--color-bg-active: #f1f5f9;        /* active背景 */

/* 边框色 */
--color-border-light: #e2e8f0;     /* 浅色边框 */
--color-border-default: #cbd5e1;   /* 默认边框 */
--color-border-strong: #94a3b8;    /* 强调边框 */

/* 文字色 */
--color-text-primary: #0f172a;     /* 主要文字 - 标题 */
--color-text-secondary: #475569;   /* 次要文字 - 正文 */
--color-text-tertiary: #64748b;    /* 三级文字 - 辅助 */
--color-text-placeholder: #94a3b8; /* 占位符 */
--color-text-disabled: #cbd5e1;    /* 禁用状态 */
--color-text-inverse: #ffffff;     /* 反色文字 */
```

### 2.4 状态色映射
| 状态 | 背景色 | 文字色 | 边框色 | 用途 |
|------|--------|--------|--------|------|
| pending | --color-bg-secondary | --color-text-tertiary | --color-border-light | 待处理 |
| running | --color-primary-50 | --color-primary-600 | --color-primary-200 | 运行中 |
| completed | --color-success-50 | --color-success-600 | --color-success-100 | 已完成 |
| failed | --color-danger-50 | --color-danger-600 | --color-danger-100 | 失败 |
| cancelled | --color-bg-tertiary | --color-text-tertiary | --color-border-light | 已取消 |

---

## 3. 字体规范

### 3.1 字体栈
```css
--font-family-base: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans SC', sans-serif;
--font-family-mono: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
```

### 3.2 字体大小
```css
--font-size-xs: 11px;      /* 标签、徽章 */
--font-size-sm: 12px;      /* 辅助文字、时间戳 */
--font-size-base: 14px;    /* 正文 */
--font-size-md: 15px;      /* 稍大正文 */
--font-size-lg: 16px;      /* 小标题 */
--font-size-xl: 18px;      /* 卡片标题 */
--font-size-2xl: 20px;     /* 页面副标题 */
--font-size-3xl: 24px;     /* 页面标题 */
```

### 3.3 字体粗细
```css
--font-weight-normal: 400;
--font-weight-medium: 500;
--font-weight-semibold: 600;
--font-weight-bold: 700;
```

### 3.4 行高
```css
--line-height-tight: 1.25;   /* 标题 */
--line-height-normal: 1.5;   /* 正文 */
--line-height-relaxed: 1.75; /* 宽松排版 */
```

### 3.5 字体使用规范
| 元素 | 大小 | 粗细 | 行高 | 颜色 |
|------|------|------|------|------|
| 页面标题 | 24px | 600 | 1.25 | --color-text-primary |
| 页面副标题 | 18px | 600 | 1.25 | --color-text-primary |
| 卡片标题 | 16px | 600 | 1.25 | --color-text-primary |
| 正文 | 14px | 400 | 1.5 | --color-text-secondary |
| 辅助文字 | 12px | 400 | 1.5 | --color-text-tertiary |
| 标签/徽章 | 11px | 600 | 1 | 根据状态变化 |

---

## 4. 间距系统

### 4.1 基础间距单位
```css
--space-unit: 4px;
```

### 4.2 间距刻度
```css
--space-1: 4px;     /* 极紧凑 */
--space-2: 8px;     /* 紧凑 */
--space-3: 12px;    /* 默认 */
--space-4: 16px;    /* 舒适 */
--space-5: 20px;    /* 宽松 */
--space-6: 24px;    /* 区块间距 */
--space-8: 32px;    /* 大区块 */
--space-10: 40px;   /* 页面间距 */
--space-12: 48px;   /* 超大间距 */
```

### 4.3 组件间距规范
| 场景 | 间距值 | 说明 |
|------|--------|------|
| 卡片内边距 | 16px-20px | 统一卡片内部空间 |
| 表单字段间距 | 16px | 字段之间 |
| 按钮内边距 | 8px 16px | 标准按钮 |
| 小按钮内边距 | 6px 12px | 图标按钮 |
| 列表项间距 | 12px | 列表项之间 |
| 区块间距 | 24px | 主要内容区块 |
| 页面边距 | 24px-32px | 页面四周 |

---

## 5. 圆角规范

```css
--radius-sm: 4px;    /* 小元素：标签、徽章 */
--radius-md: 6px;    /* 按钮、输入框 */
--radius-lg: 8px;    /* 卡片、弹窗 */
--radius-xl: 12px;   /* 大卡片、模态框 */
--radius-2xl: 16px;  /* 特殊强调卡片 */
--radius-full: 9999px; /* 圆形、胶囊 */
```

---

## 6. 阴影规范

```css
/* 基础阴影 */
--shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);

/* 默认阴影 - 卡片 */
--shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);

/* 大阴影 - 弹窗、下拉 */
--shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);

/* 强调阴影 - 悬浮卡片 */
--shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);

/* 彩色阴影 - 主色强调 */
--shadow-primary: 0 4px 14px 0 rgba(59, 130, 246, 0.39);

/* 危险阴影 */
--shadow-danger: 0 4px 14px 0 rgba(239, 68, 68, 0.39);
```

---

## 7. 按钮样式

### 7.1 Primary 按钮
```css
.btn-primary {
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  color: white;
  border: none;
  border-radius: 6px;
  padding: 8px 16px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 1px 3px rgba(37, 99, 235, 0.2);
}

.btn-primary:hover {
  background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
  transform: translateY(-1px);
}

.btn-primary:active {
  transform: translateY(0);
  box-shadow: 0 1px 2px rgba(37, 99, 235, 0.2);
}

.btn-primary:disabled {
  background: #93c5fd;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}
```

### 7.2 Secondary 按钮
```css
.btn-secondary {
  background: white;
  color: #374151;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  padding: 8px 16px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-secondary:hover {
  background: #f9fafb;
  border-color: #d1d5db;
  color: #111827;
}

.btn-secondary:active {
  background: #f3f4f6;
}
```

### 7.3 Danger 按钮
```css
.btn-danger {
  background: #ef4444;
  color: white;
  border: none;
  border-radius: 6px;
  padding: 8px 16px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-danger:hover {
  background: #dc2626;
  box-shadow: 0 4px 12px rgba(239, 68, 68, 0.35);
}
```

### 7.4 Ghost 按钮
```css
.btn-ghost {
  background: transparent;
  color: #6b7280;
  border: none;
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-ghost:hover {
  background: #f3f4f6;
  color: #374151;
}
```

### 7.5 Icon 按钮
```css
.btn-icon {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  color: #6b7280;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-icon:hover {
  background: #f9fafb;
  border-color: #d1d5db;
  color: #374151;
}

.btn-icon.danger:hover {
  background: #fef2f2;
  border-color: #fecaca;
  color: #dc2626;
}
```

---

## 8. 卡片样式

### 8.1 标准卡片
```css
.card {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}
```

### 8.2 任务卡片
```css
.task-card {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 16px 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  transition: all 0.2s ease;
  cursor: pointer;
}

.task-card:hover {
  border-color: #cbd5e1;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
  transform: translateY(-1px);
}

.task-card.selected {
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}
```

---

## 9. 表单输入框样式

### 9.1 标准输入框
```css
.input {
  width: 100%;
  padding: 10px 14px;
  font-size: 14px;
  line-height: 1.5;
  color: #0f172a;
  background: white;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  transition: all 0.2s ease;
}

.input:hover {
  border-color: #94a3b8;
}

.input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.input::placeholder {
  color: #94a3b8;
}

.input:disabled {
  background: #f1f5f9;
  color: #94a3b8;
  cursor: not-allowed;
}
```

### 9.2 文本域
```css
.textarea {
  width: 100%;
  padding: 12px 14px;
  font-size: 14px;
  line-height: 1.6;
  color: #0f172a;
  background: white;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  resize: vertical;
  min-height: 100px;
  transition: all 0.2s ease;
}

.textarea:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}
```

### 9.3 选择框
```css
.select {
  width: 100%;
  padding: 10px 36px 10px 14px;
  font-size: 14px;
  color: #0f172a;
  background: white url("data:image/svg+xml,...") no-repeat right 12px center;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  appearance: none;
  cursor: pointer;
  transition: all 0.2s ease;
}

.select:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}
```

---

## 10. 进度条样式

### 10.1 整体进度条
```css
.progress-container {
  width: 100%;
  background: #e2e8f0;
  border-radius: 9999px;
  height: 8px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%);
  border-radius: 9999px;
  transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
}

/* 进度条动画效果 */
.progress-bar::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(255, 255, 255, 0.3) 50%,
    transparent 100%
  );
  animation: shimmer 2s infinite;
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}
```

### 10.2 步骤指示器
```css
.step-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
}

.step-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.step-circle {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 600;
  transition: all 0.3s ease;
}

.step-circle.pending {
  background: #f1f5f9;
  color: #64748b;
  border: 2px solid #e2e8f0;
}

.step-circle.running {
  background: #eff6ff;
  color: #2563eb;
  border: 2px solid #3b82f6;
  animation: pulse 2s infinite;
}

.step-circle.completed {
  background: #ecfdf5;
  color: #059669;
  border: 2px solid #10b981;
}

.step-circle.failed {
  background: #fef2f2;
  color: #dc2626;
  border: 2px solid #ef4444;
}

.step-label {
  font-size: 12px;
  font-weight: 500;
  color: #475569;
}

.step-connector {
  flex: 1;
  height: 2px;
  background: #e2e8f0;
  margin: 0 8px;
  position: relative;
  top: -14px;
}

.step-connector.completed {
  background: #10b981;
}
```

---

## 11. 标签/徽章样式

```css
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.02em;
  line-height: 1;
}

.badge-pending {
  background: #f1f5f9;
  color: #64748b;
}

.badge-running {
  background: #eff6ff;
  color: #2563eb;
}

.badge-running::before {
  content: '';
  width: 6px;
  height: 6px;
  background: #2563eb;
  border-radius: 50%;
  animation: pulse-dot 2s infinite;
}

.badge-completed {
  background: #ecfdf5;
  color: #059669;
}

.badge-failed {
  background: #fef2f2;
  color: #dc2626;
}

.badge-cancelled {
  background: #f1f5f9;
  color: #64748b;
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.8); }
}
```

---

## 12. 弹窗/对话框样式

```css
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.5);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  animation: fadeIn 0.2s ease;
}

.modal {
  background: white;
  border-radius: 12px;
  width: 100%;
  max-width: 600px;
  max-height: 90vh;
  overflow: hidden;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
  animation: slideUp 0.3s ease;
}

.modal-header {
  padding: 20px 24px;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.modal-title {
  font-size: 18px;
  font-weight: 600;
  color: #0f172a;
}

.modal-close {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: none;
  border-radius: 6px;
  color: #64748b;
  cursor: pointer;
  transition: all 0.2s ease;
}

.modal-close:hover {
  background: #f1f5f9;
  color: #374151;
}

.modal-body {
  padding: 24px;
  overflow-y: auto;
  max-height: calc(90vh - 140px);
}

.modal-footer {
  padding: 16px 24px;
  border-top: 1px solid #e2e8f0;
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(20px) scale(0.98);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}
```

---

## 13. 动画效果

### 13.1 过渡效果
```css
--transition-fast: 150ms ease;
--transition-normal: 200ms ease;
--transition-slow: 300ms ease;
--transition-bounce: 300ms cubic-bezier(0.34, 1.56, 0.64, 1);
```

### 13.2 关键帧动画
```css
/* 淡入 */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

/* 向上滑入 */
@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 脉冲 */
@keyframes pulse {
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4);
  }
  50% {
    box-shadow: 0 0 0 8px rgba(59, 130, 246, 0);
  }
}

/* 旋转 */
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* 弹跳 */
@keyframes bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-4px); }
}

/* 抖动 - 用于错误提示 */
@keyframes shake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-4px); }
  75% { transform: translateX(4px); }
}
```

---

## 14. 响应式断点

```css
--breakpoint-sm: 640px;   /* 手机横屏 */
--breakpoint-md: 768px;   /* 平板 */
--breakpoint-lg: 1024px;  /* 小桌面 */
--breakpoint-xl: 1280px;  /* 大桌面 */
--breakpoint-2xl: 1536px; /* 超大屏 */
```

---

## 15. Z-Index 层级

```css
--z-dropdown: 100;
--z-sticky: 200;
--z-modal: 300;
--z-popover: 400;
--z-tooltip: 500;
--z-toast: 600;
```
