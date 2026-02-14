#!/usr/bin/env python
"""
业界级多任务特征筛选完整流程
Industry-Standard Multi-Task Feature Selection Pipeline

参考来源:
- MIT Press (2025): Multitask Learning 1997-2024: Regularization and Optimization
- Cambridge (2024): Multitask feature selection with LASSO
- Springer (2025): Deep multi-task learning review
- 业界实践: 阿里、字节、腾讯推荐系统

流程概述:
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: 数据质量检查 (Data Quality Check)                      │
│  Stage 2: 数据泄露检测 (Data Leakage Detection)                   │
│  Stage 4: 多任务特异性分析 (Multi-Task Specific Analysis)        │
│  Stage 5: Wrapper/Embedded方法 (Model-Based Selection) │  Stage 3: 基础特征筛选 (Filter Methods)                          │
         │
│  Stage 6: 特征稳定性验证 (Stability Validation)                  │
│  Stage 7: 业务逻辑审查 (Domain Review)                           │
└─────────────────────────────────────────────────────────────────┘

使用方法:
    python multi_task_feature_selection_pipeline.py --stage all
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MultiTaskFeatureSelectionPipeline:
    """多任务特征筛选流水线"""

    def __init__(self, data_path: str, label_cols: List[str], output_dir: str):
        self.data_path = data_path
        self.label_cols = label_cols
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载数据
        print(f"Loading data from: {data_path}")
        self.df = pd.read_parquet(data_path)
        print(f"Data shape: {self.df.shape}")

        # 初始化结果存储
        self.results = {
            'stage1_data_quality': {},
            'stage2_leakage': {},
            'stage3_filter': {},
            'stage4_multitask': {},
            'stage5_model': {},
            'stage6_stability': {},
            'stage7_domain': {},
            'final_features': []
        }

    # ============================================================
    # Stage 1: 数据质量检查
    # ============================================================
    def stage1_data_quality_check(self) -> Dict:
        """
        阶段1: 数据质量检查

        检查项:
        1. 缺失值率 > 50% → 移除
        2. 常数特征 (unique=1) → 移除
        3. 高基数特征 (>10000) → 需要编码
        4. 唯一值特征 (unique=样本数) → 移除

        业界标准: 参考阿里、字节推荐系统特征工程规范
        """
        print("\n" + "="*80)
        print("STAGE 1: DATA QUALITY CHECK")
        print("="*80)

        remove_features = set()
        warning_features = {}

        # 新增：按筛选原因分组
        removed_by_reason = {
            'high_missing': [],       # 高缺失率
            'constant_features': [],  # 常数特征
            'zero_variance': [],      # 零方差
        }

        # 特征列表
        feature_cols = [col for col in self.df.columns if col not in self.label_cols]

        # 1. 缺失值检查
        print("\n[1/4] Missing Value Check...")
        for col in feature_cols:
            missing_rate = self.df[col].isna().sum() / len(self.df)

            if missing_rate > 0.5:
                remove_features.add(col)
                removed_by_reason['high_missing'].append({
                    'feature': col,
                    'missing_rate': missing_rate
                })
                print(f"  ❌ REMOVE: {col} (missing={missing_rate:.1%})")
            elif missing_rate > 0.3:
                warning_features[col] = warning_features.get(col, []) + ['high_missing']
                print(f"  ⚠️  WARNING: {col} (missing={missing_rate:.1%})")

        # 2. 常数特征检查
        print("\n[2/4] Constant Feature Check...")
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if col in self.label_cols:
                continue
            unique_count = self.df[col].nunique()

            if unique_count <= 1:
                remove_features.add(col)
                removed_by_reason['constant_features'].append({
                    'feature': col,
                    'unique_count': unique_count
                })
                print(f"  ❌ REMOVE: {col} (unique={unique_count})")
            elif unique_count == len(self.df):
                # ID特征，需要确认
                warning_features[col] = warning_features.get(col, []) + ['id_like']
                print(f"  ⚠️  WARNING: {col} (unique={unique_count}, possibly ID)")

        # 3. 高基数特征检查
        print("\n[3/4] High Cardinality Check...")
        for col in self.df.columns:
            if col in self.label_cols:
                continue
            # 跳过序列类型列（list/array）
            if self.df[col].dtype == 'object':
                # 检查是否是序列类型
                sample = self.df[col].dropna().iloc[0] if len(self.df[col].dropna()) > 0 else None
                if isinstance(sample, (list, np.ndarray)):
                    print(f"  ℹ️  {col}: sequence feature, skipping cardinality check")
                    continue

            try:
                cardinality = self.df[col].nunique()
            except TypeError:
                # 无法计算基数的列
                continue

            if cardinality > 10000:
                warning_features[col] = warning_features.get(col, []) + ['high_cardinality']
                print(f"  ⚠️  WARNING: {col} (cardinality={cardinality})")

        # 4. 零方差特征
        print("\n[4/4] Zero Variance Check...")
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if col in self.label_cols:
                continue
            if self.df[col].std() == 0:
                remove_features.add(col)
                removed_by_reason['zero_variance'].append({
                    'feature': col
                })
                print(f"  ❌ REMOVE: {col} (zero variance)")

        self.results['stage1_data_quality'] = {
            'remove': list(remove_features),
            'warning': warning_features,
            'removed_by_reason': removed_by_reason
        }

        print(f"\n✓ Stage 1 Complete: {len(remove_features)} features to remove")

        return {
            'remove': list(remove_features),
            'warning': warning_features,
            'removed_by_reason': removed_by_reason
        }

    # ============================================================
    # 辅助方法: 特征类型检测
    # ============================================================
    def _is_sequence_feature(self, col: str) -> bool:
        """检测是否为序列特征（包含list或ndarray）"""
        if self.df[col].dtype != 'object':
            return False
        sample = self.df[col].dropna().iloc[0] if len(self.df[col].dropna()) > 0 else None
        return isinstance(sample, (list, np.ndarray))

    def _get_feature_cardinality(self, col: str) -> int:
        """获取特征基数（唯一值数量）"""
        return self.df[col].nunique()

    def _is_categorical_feature(self, col: str) -> bool:
        """检测是否为类别特征（非数值、非序列）"""
        if self._is_sequence_feature(col):
            return False
        if pd.api.types.is_numeric_dtype(self.df[col]):
            return False
        return True

    def _calculate_woe_iv_categorical(self, feature: pd.Series, label: pd.Series) -> float:
        """
        计算类别特征的WoE和IV（用于二分类）

        参考文献:
        - Siddiqi, N. (2006). "Credit Risk Scorecards"
        - Industry standard in financial risk control

        IV阈值标准:
        - IV < 0.02:  无预测能力，移除
        - 0.02-0.1:  弱预测能力
        - 0.1-0.3:   中等预测能力
        - 0.3-0.5:   强预测能力
        - IV > 0.5:  可疑（可能数据泄露）

        Args:
            feature: 类别特征数据
            label: 标签数据 (0/1)

        Returns:
            IV值
        """
        try:
            # 合并数据，移除NaN
            df_temp = pd.DataFrame({'feature': feature, 'label': label}).dropna()

            if len(df_temp) < 100:  # 样本太少
                return 0.0

            # 统计总的好坏样本数
            total_good = (df_temp['label'] == 1).sum()
            total_bad = (df_temp['label'] == 0).sum()

            if total_good == 0 or total_bad == 0:
                return 0.0

            iv_sum = 0

            # 对每个类别计算 WoE 和 IV
            for category in df_temp['feature'].unique():
                df_cat = df_temp[df_temp['feature'] == category]
                good = df_cat['label'].sum()
                bad = df_cat.shape[0] - good

                # 跳过空类别
                if good == 0 or bad == 0:
                    continue

                # 计算分布
                dist_good = good / total_good
                dist_bad = bad / total_bad

                # 平滑处理，避免log(0)
                dist_good = max(dist_good, 0.0001)
                dist_bad = max(dist_bad, 0.0001)

                # 计算 WoE
                woe = np.log(dist_good / dist_bad)

                # 计算 IV
                iv = (dist_good - dist_bad) * woe
                iv_sum += iv

            return float(iv_sum)

        except Exception as e:
            return 0.0

    # ============================================================
    # Stage 2: 数据泄露检测
    # ============================================================
    def stage2_leakage_detection(self, threshold: float = 0.8) -> Dict:
        """
        阶段2: 数据泄露检测

        方法:
        1. 基于命名规则: reportmodel_*, *_rate, *_converate 等明显泄露
        2. 基于相关性: 与任一label相关性 > threshold
        3. 基于业务逻辑: 特征计算依赖未来信息

        参考: Cambridge (2024) - Feature selection with LASSO
        """
        print("\n" + "="*80)
        print("STAGE 2: DATA LEAKAGE DETECTION")
        print("="*80)

        leakage_features = set()
        suspicious_features = set()

        feature_cols = [col for col in self.df.columns if col not in self.label_cols]

        # 1. 基于命名规则
        print("\n[1/3] Rule-Based Detection (Naming Patterns)...")
        leakage_patterns = [
            'reportmodel_',
            '_regisconverate_',
            '_applyconverate_',
            '_creditconverate_',
            '_rate_cnt',
            '_roi_cnt',
            '_clickrate_cnt',
            '_pricerate_cnt',
        ]

        for col in feature_cols:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in leakage_patterns):
                leakage_features.add(col)
                print(f"  ⚠️  LEAKAGE: {col} (matched naming pattern)")

        # 2. 基于相关性分析
        print("\n[2/3] Correlation-Based Detection...")

        # 数值特征相关性
        numeric_features = [col for col in feature_cols
                          if pd.api.types.is_numeric_dtype(self.df[col])]

        from scipy.stats import pointbiserialr

        for col in numeric_features:
            if col in leakage_features:
                continue

            feature_data = self.df[col]
            max_abs_corr = 0

            for label in self.label_cols:
                label_data = self.df[label]

                # 过滤NaN
                valid_mask = ~(feature_data.isna() | label_data.isna())
                if valid_mask.sum() < 1000:
                    continue

                try:
                    corr, _ = pointbiserialr(
                        label_data[valid_mask].values,
                        feature_data[valid_mask].values
                    )

                    if abs(corr) > max_abs_corr:
                        max_abs_corr = abs(corr)
                except:
                    pass

            if max_abs_corr > threshold:
                leakage_features.add(col)
                print(f"  🚨 LEAKAGE: {col} (max_corr={max_abs_corr:.4f})")
            elif max_abs_corr > 0.6:
                suspicious_features.add(col)
                print(f"  ⚠️  SUSPICIOUS: {col} (max_corr={max_abs_corr:.4f})")

        # 3. 特殊特征检查
        print("\n[3/3] Special Feature Check...")

        # mymodel_dk_* 特征 - 可能是模型相关
        mymodel_features = [col for col in numeric_features if col.startswith('mymodel_dk_')]
        if mymodel_features:
            print(f"  ℹ️  Found {len(mymodel_features)} mymodel_dk_* features")
            print(f"     These need domain expert review")

        self.results['stage2_leakage'] = {
            'leakage': list(leakage_features),
            'suspicious': list(suspicious_features)
        }

        print(f"\n✓ Stage 2 Complete: {len(leakage_features)} leakage features")

        return {'leakage': list(leakage_features), 'suspicious': list(suspicious_features)}

    def _calculate_iv(self, feature: pd.Series, label: pd.Series, n_bins: int = 10) -> float:
        """
        计算特征的信息值 (Information Value)

        参考文献:
        - Siddiqi, N. (2006). "Credit Risk Scorecards"
        - Industry standard in financial risk control

        IV阈值标准:
        - IV < 0.02:  无预测能力，移除
        - 0.02-0.1:  弱预测能力
        - 0.1-0.3:   中等预测能力
        - 0.3-0.5:   强预测能力
        - IV > 0.5:  可疑（可能数据泄露），提取为业务规则

        Args:
            feature: 特征数据
            label: 标签数据 (0/1)
            n_bins: 分箱数量 (默认10)

        Returns:
            IV值
        """
        try:
            # 合并数据，移除NaN
            df_temp = pd.DataFrame({'feature': feature, 'label': label}).dropna()

            if len(df_temp) < 100:  # 样本太少
                return 0.0

            # 分箱 (quantile-based)
            df_temp['bin'] = pd.qcut(df_temp['feature'], q=n_bins, duplicates='drop')

            # 统计每个箱的好坏样本数
            stats = df_temp.groupby('bin').agg({
                'label': ['count', 'sum']
            }).reset_index()

            stats.columns = ['bin', 'total', 'bad']
            stats['good'] = stats['total'] - stats['bad']

            # 计算总的好坏样本数
            total_good = stats['good'].sum()
            total_bad = stats['bad'].sum()

            if total_good == 0 or total_bad == 0:
                return 0.0

            # 计算每个箱的分布
            stats['dist_good'] = stats['good'] / total_good
            stats['dist_bad'] = stats['bad'] / total_bad

            # 平滑处理，避免log(0)
            smoothing = 0.5
            stats['dist_good'] = stats['dist_good'].replace(0, smoothing / total_good)
            stats['dist_bad'] = stats['dist_bad'].replace(0, smoothing / total_bad)

            # 计算WOE (Weight of Evidence)
            stats['woe'] = np.log(stats['dist_good'] / stats['dist_bad'])

            # 计算IV
            stats['iv'] = (stats['dist_good'] - stats['dist_bad']) * stats['woe']

            iv_value = stats['iv'].sum()

            return float(iv_value)

        except Exception as e:
            return 0.0

    # ============================================================
    # Stage 3: 基础特征筛选 (Filter Methods)
    # ============================================================
    def stage3_filter_methods(self, features: List[str]) -> Dict:
        """
        阶段3: Filter方法特征筛选（支持数值特征和类别特征）

        方法:
        【数值特征】:
        1. 方差阈值 (Variance Threshold) - 移除低方差特征
        2. 相关性冗余 (Correlation Redundancy) - 移除高相关冗余特征
        3. 单特征性能 (Univariate Performance) - 计算每个特征的预测能力
        4. 信息值 (Information Value) - 风控行业标准方法 ⭐

        【类别特征】:
        5. WoE + IV (Weight of Evidence + Information Value) - 风控行业标准 ⭐

        参考文献:
        - Variance Threshold: Saeys et al. (2007), Bioinformatics (PMC)
        - Correlation Redundancy: Guyon & Elisseeff (2003), JMLR
        - IV (Information Value): Siddiqi (2006), "Credit Risk Scorecards", Wiley
        """
        print("\n" + "="*80)
        print("STAGE 3: FILTER METHODS (Numeric + Categorical)")
        print("="*80)

        if features:
            candidate_features = [f for f in features if f in self.df.columns]
        else:
            candidate_features = [col for col in self.df.columns if col not in self.label_cols]

        # 1. 特征分类
        sequence_features = []
        categorical_features = []
        numeric_features = []

        for col in candidate_features:
            if self._is_sequence_feature(col):
                sequence_features.append(col)
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                numeric_features.append(col)
            else:
                categorical_features.append(col)

        print(f"\n[Feature Type Distribution]")
        print(f"  Numeric: {len(numeric_features)}")
        print(f"  Categorical: {len(categorical_features)}")
        print(f"  Sequence (skipped): {len(sequence_features)}")

        remove_features = set()
        feature_iv_scores = {}
        feature_iv_scores_by_task = {}
        feature_scores_by_task = {}
        # Multi-task aggregation: prefer features that are both useful and stable across tasks.
        iv_multitask_alpha = 0.3
        iv_threshold = 0.02

        # 新增：按筛选原因分组
        removed_by_reason = {
            'variance_threshold': [],      # 方差阈值
            'correlation_redundancy': [],  # 相关性冗余
            'low_univariate_auc': [],      # 低单变量AUC
            'low_iv_numeric': [],          # 数值特征低IV
            'low_iv_categorical': [],      # 类别特征低IV
            'correlation_pairs': []        # 相关性特征对
        }

        # ============================================================
        # 数值特征筛选
        # ============================================================
        print("\n" + "="*80)
        print("[NUMERIC FEATURES FILTERING]")
        print("="*80)

        if numeric_features:
            # 1. 方差阈值
            print("\n[1/4] Variance Threshold...")
            variance_threshold = 0.01  # 业界标准值

            for col in numeric_features:
                var = self.df[col].var()
                if var < variance_threshold:
                    remove_features.add(col)
                    removed_by_reason['variance_threshold'].append({
                        'feature': col,
                        'variance': var
                    })
                    print(f"  ❌ REMOVE: {col} (variance={var:.6f})")

            # 2. 冗余特征检测 (高相关性特征对)
            print("\n[2/4] Redundancy Check (Correlation > 0.95)...")

            # 计算特征间相关性矩阵
            valid_numeric = [f for f in numeric_features if f not in remove_features]
            if len(valid_numeric) > 0:
                corr_matrix = self.df[valid_numeric].corr().abs()

                # 找出高相关特征对
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if corr_val > 0.95:
                            feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
                            high_corr_pairs.append((feat1, feat2, corr_val))

                # 保留相关性较高的特征，移除另一个
                redundant = set()
                for feat1, feat2, corr in high_corr_pairs:
                    if feat1 not in redundant and feat2 not in redundant:
                        # 移除相关性较低的那个（与label的相关性）
                        # 这里简化处理：移除feat2
                        redundant.add(feat2)
                        removed_by_reason['correlation_redundancy'].append({
                            'feature': feat2,
                            'corr_with': feat1,
                            'correlation': corr
                        })
                        removed_by_reason['correlation_pairs'].append({
                            'feat1': feat1,
                            'feat2': feat2,
                            'corr': corr
                        })
                        print(f"  ⚠️  REDUNDANT: {feat2} (corr={corr:.3f} with {feat1})")

                remove_features.update(redundant)
                valid_numeric = [f for f in valid_numeric if f not in redundant]

            # 3. 单特征性能评估
            print("\n[3/4] Univariate Performance...")

            from sklearn.metrics import roc_auc_score

            feature_scores = {}
            for col in valid_numeric:
                if col in remove_features:
                    continue

                try:
                    feature_data = self.df[col].fillna(0)  # 简单填充

                    scores = []
                    task_scores = {}
                    for label in self.label_cols:
                        label_data = self.df[label]

                        # 二分类AUC
                        valid_mask = ~(feature_data.isna() | label_data.isna())
                        if valid_mask.sum() > 100:
                            try:
                                auc = roc_auc_score(
                                    label_data[valid_mask],
                                    feature_data[valid_mask]
                                )
                                task_score = abs(auc - 0.5) * 2  # 归一化到[0,1]
                                scores.append(task_score)
                                task_scores[label] = task_score
                            except:
                                pass

                    if scores:
                        avg_score = np.mean(scores)
                        feature_scores[col] = avg_score
                        feature_scores_by_task[col] = task_scores

                except Exception as e:
                    pass

            # 移除低分特征
            low_score_threshold = 0.05  # 业界标准
            low_score_features = [f for f, s in feature_scores.items()
                                  if s < low_score_threshold]

            for f in low_score_features:
                removed_by_reason['low_univariate_auc'].append({
                    'feature': f,
                    'auc_score': feature_scores[f]
                })

            print(f"  ℹ️  {len(low_score_features)} features with low univariate score (< {low_score_threshold})")
            remove_features.update(low_score_features)

            # 4. 信息值 (Information Value) - 风控行业标准方法
            print("\n[4/4] Information Value (IV) - Risk Control Standard...")

            iv_threshold = 0.02  # 风控行业最低标准

            for col in valid_numeric:
                if col in remove_features:
                    continue

                try:
                    # 多任务IV：分别计算后做聚合（mean - alpha * std）
                    iv_by_task = {}
                    for label in self.label_cols:
                        label_data = self.df[label]
                        iv_by_task[label] = self._calculate_iv(self.df[col], label_data, n_bins=10)

                    iv_values = list(iv_by_task.values())
                    iv_score = float(np.mean(iv_values) - iv_multitask_alpha * np.std(iv_values))
                    feature_iv_scores[col] = iv_score
                    feature_iv_scores_by_task[col] = iv_by_task

                    # 输出IV分析
                    if iv_score < 0.02:
                        remove_features.add(col)
                        removed_by_reason['low_iv_numeric'].append({
                            'feature': col,
                            'iv': iv_score,
                            'iv_by_task': iv_by_task
                        })
                        print(f"  ❌ REMOVE: {col} (IV={iv_score:.4f} - No predictive power)")
                    elif iv_score < 0.1:
                        print(f"  ⚠️  WEAK: {col} (IV={iv_score:.4f} - Weak predictor)")
                    elif iv_score < 0.3:
                        print(f"  ✓ MEDIUM: {col} (IV={iv_score:.4f} - Medium predictor)")
                    elif iv_score < 0.5:
                        print(f"  ✓✓ STRONG: {col} (IV={iv_score:.4f} - Strong predictor)")
                    else:
                        print(f"  ⚠️  SUSPICIOUS: {col} (IV={iv_score:.4f} - Possible data leakage)")

                except Exception as e:
                    pass

            print(f"\n  ℹ️  {len(removed_by_reason['low_iv_numeric'])} numeric features with IV < {iv_threshold}")
        else:
            print("\n  ℹ️  No numeric features to filter")
            feature_scores = {}

        # ============================================================
        # 类别特征筛选
        # ============================================================
        print("\n" + "="*80)
        print("[CATEGORICAL FEATURES FILTERING]")
        print("="*80)

        if categorical_features:
            print("\n[1/1] WoE + IV (Weight of Evidence) - Risk Control Standard...")
            print(f"  Processing {len(categorical_features)} categorical features...")

            iv_threshold = 0.02  # 风控行业最低标准

            for col in categorical_features:
                try:
                    # 多任务IV：分别计算后做聚合（mean - alpha * std）
                    iv_by_task = {}
                    for label in self.label_cols:
                        label_data = self.df[label]
                        iv_by_task[label] = self._calculate_woe_iv_categorical(self.df[col], label_data)

                    iv_values = list(iv_by_task.values())
                    iv_score = float(np.mean(iv_values) - iv_multitask_alpha * np.std(iv_values))
                    feature_iv_scores[col] = iv_score
                    feature_iv_scores_by_task[col] = iv_by_task

                    # 输出IV分析
                    if iv_score < 0.02:
                        remove_features.add(col)
                        removed_by_reason['low_iv_categorical'].append({
                            'feature': col,
                            'iv': iv_score,
                            'iv_by_task': iv_by_task
                        })
                        print(f"  ❌ REMOVE: {col} (IV={iv_score:.4f} - No predictive power)")
                    elif iv_score < 0.1:
                        print(f"  ⚠️  WEAK: {col} (IV={iv_score:.4f} - Weak predictor)")
                    elif iv_score < 0.3:
                        print(f"  ✓ MEDIUM: {col} (IV={iv_score:.4f} - Medium predictor)")
                    elif iv_score < 0.5:
                        print(f"  ✓✓ STRONG: {col} (IV={iv_score:.4f} - Strong predictor)")
                    else:
                        print(f"  ⚠️  SUSPICIOUS: {col} (IV={iv_score:.4f} - Possible data leakage)")

                except Exception as e:
                    print(f"  ⚠️  ERROR: {col} - {e}")

            # 移除低IV特征
            low_iv_categorical = [f for f in categorical_features
                                if f in feature_iv_scores and feature_iv_scores[f] < iv_threshold]
            print(f"\n  ℹ️  {len(low_iv_categorical)} categorical features with IV < {iv_threshold}")

        else:
            print("\n  ℹ️  No categorical features to filter")

        # ============================================================
        # 汇总输出
        # ============================================================
        print("\n" + "="*80)
        print("[IV DISTRIBUTION SUMMARY]")
        print("="*80)

        if feature_iv_scores:
            print(f"  Weak (<0.1): {sum(1 for iv in feature_iv_scores.values() if iv < 0.1)}")
            print(f"  Medium (0.1-0.3): {sum(1 for iv in feature_iv_scores.values() if 0.1 <= iv < 0.3)}")
            print(f"  Strong (0.3-0.5): {sum(1 for iv in feature_iv_scores.values() if 0.3 <= iv < 0.5)}")
            print(f"  Suspicious (>0.5): {sum(1 for iv in feature_iv_scores.values() if iv >= 0.5)}")

        self.results['stage3_filter'] = {
            'remove': list(remove_features),
            'feature_scores': feature_scores,
            'feature_scores_by_task': feature_scores_by_task,
            'iv_scores': feature_iv_scores,
            'iv_scores_by_task': feature_iv_scores_by_task,
            'removed_by_reason': removed_by_reason,
            'iv_threshold': iv_threshold,
            'iv_multitask_alpha': iv_multitask_alpha,
            'numeric_count': len(numeric_features),
            'categorical_count': len(categorical_features),
            'sequence_count': len(sequence_features)
        }

        print(f"\n✓ Stage 3 Complete: {len(remove_features)} features to remove")

        return {
            'remove': list(remove_features),
            'scores': feature_scores,
            'iv_scores': feature_iv_scores,
            'scores_by_task': feature_scores_by_task,
            'iv_scores_by_task': feature_iv_scores_by_task,
            'removed_by_reason': removed_by_reason
        }

    # ============================================================
    # Stage 4: 多任务特异性分析
    # ============================================================
    def stage4_multitask_analysis(self, features: List[str]) -> Dict:
        """
        阶段4: 多任务特异性分析

        分析内容:
        1. 任务共享特征 (Task-Shared Features) - 对所有任务都有用
        2. 任务特异特征 (Task-Specific Features) - 只对某个任务有用
        3. 任务冲突特征 (Task-Conflicting Features) - 对不同任务有相反作用

        参考: Springer (2025) - Deep multi-task learning review
        """
        print("\n" + "="*80)
        print("STAGE 4: MULTI-TASK SPECIFIC ANALYSIS")
        print("="*80)

        if features:
            candidate_features = [f for f in features if f in self.df.columns]
        else:
            candidate_features = [col for col in self.df.columns if col not in self.label_cols]

        from sklearn.metrics import roc_auc_score

        feature_task_importance = {}

        # 计算每个特征对每个任务的重要性
        for col in candidate_features:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue

            feature_data = self.df[col].fillna(0)
            importances = {}

            for label in self.label_cols:
                label_data = self.df[label]

                valid_mask = ~(feature_data.isna() | label_data.isna())
                if valid_mask.sum() > 100:
                    try:
                        auc = roc_auc_score(label_data[valid_mask], feature_data[valid_mask])
                        importances[label] = abs(auc - 0.5) * 2
                    except:
                        importances[label] = 0

            feature_task_importance[col] = importances

        # 分类特征
        task_specific = {}
        task_shared = []
        task_conflicting = []

        for feat, imps in feature_task_importance.items():
            if len(imps) < len(self.label_cols):
                continue

            values = list(imps.values())
            max_val = max(values)
            min_val = min(values)

            # 任务特异: 某个任务的重要性显著高于其他任务
            if max_val > 2 * np.mean(values) and max_val > 0.3:
                dominant_task = max(imps, key=imps.get)
                task_specific[feat] = {
                    'dominant_task': dominant_task,
                    'importance': imps
                }

            # 任务冲突: 不同任务的重要性符号相反（需要原始相关性符号）
            # 这里简化处理

            # 任务共享: 对所有任务都有中等以上重要性
            elif min_val > 0.1:
                task_shared.append(feat)

        # 输出结果
        print(f"\n[Summary]")
        print(f"  Task-Specific Features: {len(task_specific)}")
        print(f"  Task-Shared Features: {len(task_shared)}")

        # Top task-specific features
        print(f"\n[Top 10 Task-Specific Features]")
        for feat, info in sorted(task_specific.items(),
                                key=lambda x: x[1]['importance'][x[1]['dominant_task']],
                                reverse=True)[:10]:
            print(f"  {feat}")
            for task, imp in info['importance'].items():
                print(f"    - {task}: {imp:.4f}")

        self.results['stage4_multitask'] = {
            'task_specific': task_specific,
            'task_shared': task_shared,
            'task_conflicting': task_conflicting
        }

        print(f"\n✓ Stage 4 Complete")

        return {
            'task_specific': task_specific,
            'task_shared': task_shared,
            'task_conflicting': task_conflicting
        }

    # ============================================================
    # Stage 5: 模型方法 (Embedded Methods)
    # ============================================================
    def stage5_model_based_selection(self, features: List[str], top_k: int = 100, use_categorical: bool = True) -> Dict:
        """
        阶段5: 基于模型的特征筛选（支持数值特征和类别特征）

        方法:
        1. LightGBM特征重要性 - 支持数值和类别特征
        2. Label Encoding for 类别特征
        3. Top-K特征选择

        参考: Cambridge (2024) - LASSO for multitask feature selection
        """
        print("\n" + "="*80)
        print("STAGE 5: MODEL-BASED SELECTION (Numeric + Categorical)")
        print("="*80)

        if features:
            candidate_features = [f for f in features if f in self.df.columns]
        else:
            candidate_features = [col for col in self.df.columns if col not in self.label_cols]

        # 特征分类
        numeric_features = []
        categorical_features = []
        sequence_features = []

        for col in candidate_features:
            if self._is_sequence_feature(col):
                sequence_features.append(col)
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                numeric_features.append(col)
            else:
                categorical_features.append(col)

        print(f"\n[Feature Type Distribution]")
        print(f"  Numeric: {len(numeric_features)}")
        print(f"  Categorical: {len(categorical_features)}")
        print(f"  Sequence (skipped): {len(sequence_features)}")

        print("\n[1/3] Preparing data for LightGBM...")

        try:
            import lightgbm as lgb
            from sklearn.preprocessing import LabelEncoder
            aggregation_beta = 0.2

            # 准备数值特征
            X_numeric = self.df[numeric_features].fillna(0) if numeric_features else pd.DataFrame()

            # 准备类别特征（Label Encoding）
            X_categorical = pd.DataFrame(index=self.df.index)
            le_dict = {}

            for col in categorical_features:
                le = LabelEncoder()
                # 处理NaN值：填充为'UNKNOWN'后再编码
                X_categorical[col] = le.fit_transform(
                    self.df[col].fillna('UNKNOWN').astype(str)
                )
                le_dict[col] = le
                print(f"  Encoded {col}: {len(le.classes_)} unique values")

            # 合并特征
            X_list = []
            if not X_numeric.empty:
                X_list.append(X_numeric)
            if not X_categorical.empty:
                X_list.append(X_categorical)

            if not X_list:
                print("  ⚠️  No features available for model-based selection")
                return {'top_features': [], 'importance': {}, 'categorical_features': []}

            X = pd.concat(X_list, axis=1)

            print(f"\n[2/3] Training LightGBM with {X.shape[1]} features...")

            params = {
                'objective': 'binary',
                'verbose': -1,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'num_threads': 1,
                'force_col_wise': True,
                'deterministic': True,
                'min_data_per_group': 1,
                'cat_l2': 10,
                'cat_smooth': 10
            }

            per_task_importance = {}
            for label in self.label_cols:
                y = self.df[label].fillna(0)
                if y.nunique(dropna=True) < 2:
                    print(f"  ⚠️  Skip task {label}: label has < 2 classes")
                    continue

                train_data = lgb.Dataset(
                    X,
                    label=y,
                    categorical_feature=(categorical_features if (use_categorical and categorical_features) else 'auto')
                )
                model = lgb.train(params, train_data)
                importance = model.feature_importance(importance_type='gain')
                per_task_importance[label] = dict(zip(X.columns, importance))

            if not per_task_importance:
                raise ValueError("no valid task labels for stage5 model training")

            # 聚合任务重要性: mean - beta * std，偏向稳定共享特征
            feature_importance = {}
            for feature in X.columns:
                task_vals = [
                    task_imp.get(feature, 0.0)
                    for task_imp in per_task_importance.values()
                ]
                mean_imp = float(np.mean(task_vals))
                std_imp = float(np.std(task_vals))
                feature_importance[feature] = mean_imp - aggregation_beta * std_imp

            # 排序并选择top-k
            sorted_features = sorted(feature_importance.items(),
                                    key=lambda x: x[1], reverse=True)

            top_features = [f for f, _ in sorted_features[:top_k]]

            print(f"\n[3/3] Top 10 Features by LightGBM Importance]")
            for i, (feat, imp) in enumerate(sorted_features[:10], 1):
                feat_type = "🔢" if feat in numeric_features else "📝"
                print(f"  {i:2}. {feat_type} {feat}: {imp}")

            # 统计top_k中的特征类型分布
            numeric_in_top = sum(1 for f in top_features if f in numeric_features)
            categorical_in_top = sum(1 for f in top_features if f in categorical_features)

            print(f"\n[Top-K Feature Distribution]")
            print(f"  Numeric: {numeric_in_top}/{len(numeric_features)} available")
            print(f"  Categorical: {categorical_in_top}/{len(categorical_features)} available")

            self.results['stage5_model'] = {
                'top_features': top_features,
                'feature_importance': feature_importance,
                'per_task_importance': per_task_importance,
                'use_categorical': use_categorical,
                'importance_aggregation': f'mean - {aggregation_beta} * std',
                'numeric_features': numeric_features,
                'categorical_features': categorical_features,
                'encoding': 'label_encoding'
            }

            print(f"\n✓ Stage 5 Complete: Selected top {len(top_features)} features")

            return {
                'top_features': top_features,
                'importance': feature_importance,
                'per_task_importance': per_task_importance,
                'use_categorical': use_categorical,
                'categorical_features': categorical_features
            }

        except ImportError:
            print("  ⚠️  LightGBM not installed, skipping model-based selection")
            # 返回简单的选择（数值特征优先）
            all_features = numeric_features + categorical_features
            return {
                'top_features': all_features[:top_k],
                'importance': {},
                'categorical_features': categorical_features
            }
        except Exception as e:
            print(f"  ⚠️  Error in Stage 5: {e}")
            # 返回简单的选择
            all_features = numeric_features + categorical_features
            return {
                'top_features': all_features[:top_k],
                'importance': {},
                'categorical_features': categorical_features
            }

    # ============================================================
    # Stage 6: 特征稳定性验证
    # ============================================================
    def stage6_stability_validation(self, features: List[str]) -> Dict:
        """
        阶段6: 特征稳定性验证

        检查内容:
        1. 时间稳定性 - 不同时间段特征分布是否稳定
        2. 样本稳定性 - 不同采样下特征表现是否一致

        业界实践: 参考阿里推荐系统特征稳定性监控
        """
        print("\n" + "="*80)
        print("STAGE 6: STABILITY VALIDATION")
        print("="*80)

        # 简化实现: 使用bootstrap采样验证稳定性
        print("\n[1/1] Bootstrap Stability Check...")

        if features:
            candidate_features = [f for f in features if f in self.df.columns]
        else:
            candidate_features = [col for col in self.df.columns if col not in self.label_cols]

        numeric_features = [col for col in candidate_features
                          if pd.api.types.is_numeric_dtype(self.df[col])]

        n_bootstrap = 5
        sample_size = min(100000, len(self.df) // 2)

        feature_stability = {}

        for col in numeric_features:
            aucs = []

            for i in range(n_bootstrap):
                # 采样
                sample_df = self.df.sample(n=sample_size, replace=True)

                feature_data = sample_df[col].fillna(0)
                label_data = sample_df[self.label_cols[0]]

                valid_mask = ~(feature_data.isna() | label_data.isna())
                if valid_mask.sum() > 100:
                    try:
                        from sklearn.metrics import roc_auc_score
                        auc = roc_auc_score(label_data[valid_mask], feature_data[valid_mask])
                        aucs.append(abs(auc - 0.5) * 2)
                    except:
                        pass

            if aucs:
                stability = np.std(aucs)  # 标准差越小越稳定
                feature_stability[col] = {
                    'mean_auc': np.mean(aucs),
                    'std_auc': stability
                }

        # 找出不稳定特征
        unstable = [f for f, s in feature_stability.items() if s['std_auc'] > 0.1]

        print(f"\n[Unstable Features (std > 0.1)]")
        for feat in unstable[:10]:
            print(f"  ⚠️  {feat}: std={feature_stability[feat]['std_auc']:.4f}")

        self.results['stage6_stability'] = {
            'stability': feature_stability,
            'unstable': unstable
        }

        print(f"\n✓ Stage 6 Complete: {len(unstable)} unstable features")

        return {'stability': feature_stability, 'unstable': unstable}

    # ============================================================
    # Stage 7: 业务逻辑审查
    # ============================================================
    def stage7_domain_review(self, features: List[str]) -> Dict:
        """
        阶段7: 业务逻辑审查

        检查项:
        1. 特征计算逻辑是否合理
        2. 特征是否包含未来信息
        3. 特征是否可在线上实时计算

        业界实践: 必须由业务专家review
        """
        print("\n" + "="*80)
        print("STAGE 7: DOMAIN REVIEW")
        print("="*80)

        print("\n⚠️  This stage requires manual domain expert review!")

        # 生成需要人工审查的特征列表
        review_features = []

        if features:
            candidate_features = [f for f in features if f in self.df.columns]
        else:
            candidate_features = [col for col in self.df.columns if col not in self.label_cols]

        # 特征分类
        model_features = [f for f in candidate_features if 'model' in f.lower()]
        rate_features = [f for f in candidate_features if 'rate' in f.lower()]
        tag_features = [f for f in candidate_features if f.endswith('_tag')]

        print(f"\n[Features Requiring Review]")
        print(f"  Model-related: {len(model_features)}")
        print(f"  Rate-related: {len(rate_features)}")
        print(f"  Tag features: {len(tag_features)}")

        # 保存待审查列表
        review_file = self.output_dir / "features_for_domain_review.csv"
        pd.DataFrame({'feature': candidate_features}).to_csv(review_file, index=False)

        print(f"\n  Saved review list to: {review_file}")

        self.results['stage7_domain'] = {
            'review_required': len(candidate_features),
            'categories': {
                'model': len(model_features),
                'rate': len(rate_features),
                'tag': len(tag_features)
            }
        }

        return {'review_required': True}

    # ============================================================
    # 完整流程执行
    # ============================================================
    def run_full_pipeline(self) -> Dict:
        """执行完整的特征筛选流程"""

        print("\n" + "="*80)
        print("MULTI-TASK FEATURE SELECTION PIPELINE")
        print("="*80)

        # Stage 1: 数据质量检查
        stage1_result = self.stage1_data_quality_check()
        remove_set1 = set(stage1_result['remove'])

        # Stage 2: 数据泄露检测 - SKIPPED per user request
        # These features may be useful in production if they can be obtained at inference time
        print("\n" + "="*80)
        print("STAGE 2: DATA LEAKAGE DETECTION - SKIPPED")
        print("="*80)
        print("\nℹ️  Stage 2 skipped per user request.")
        print("    If using model-output features (reportmodel_*), ensure they are")
        print("    obtainable at inference time.")

        stage2_result = {'leakage': []}
        remove_set2 = set()

        # 剩余特征
        remaining_features = [col for col in self.df.columns
                             if col not in self.label_cols
                             and col not in remove_set1
                             and col not in remove_set2]

        # Stage 3: Filter方法
        stage3_result = self.stage3_filter_methods(remaining_features)
        remove_set3 = set(stage3_result['remove'])

        # 更新剩余特征
        remaining_features = [f for f in remaining_features if f not in remove_set3]

        # Stage 4: 多任务分析
        stage4_result = self.stage4_multitask_analysis(remaining_features)

        # Stage 5: 模型方法
        stage5_result = self.stage5_model_based_selection(remaining_features, top_k=150)

        # Stage 6: 稳定性验证
        # stage6_result = self.stage6_stability_validation(stage5_result['top_features'])

        # Stage 7: 业务审查
        # stage7_result = self.stage7_domain_review(stage5_result['top_features'])

        # 汇总最终特征列表
        final_features = stage5_result['top_features']

        # 保存结果
        self.results['final_features'] = final_features

        output_file = self.output_dir / "feature_selection_results.json"
        with open(output_file, 'w') as f:
            # 转换numpy类型
            def convert(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, list):
                    return [convert(x) for x in obj]
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                return obj

            json.dump(convert(self.results), f, indent=2)

        # 保存最终特征列表
        final_features_file = self.output_dir / "final_features.txt"
        with open(final_features_file, 'w') as f:
            for feat in final_features:
                f.write(f"{feat}\n")

        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"\nFinal feature count: {len(final_features)}")
        print(f"Results saved to: {output_file}")
        print(f"Feature list saved to: {final_features_file}")

        return self.results


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Task Feature Selection Pipeline')
    parser.add_argument('--data', type=str,
                       default="/mnt/home/gxwang9/fuxictr/data/all_seeds_1v5_rh_0206/train.parquet",
                       help='Path to training data')
    parser.add_argument('--labels', type=str, nargs='+',
                       default=['label_register', 'label_apply', 'label_credit'],
                       help='Label columns')
    parser.add_argument('--output', type=str,
                       default="/mnt/home/gxwang9/fuxictr/analysis/feature_selection_output",
                       help='Output directory')
    parser.add_argument('--stage', type=str, default='all',
                       choices=['all', '1', '2', '3', '4', '5', '6', '7'],
                       help='Which stage to run (default: all)')

    args = parser.parse_args()

    # 创建pipeline
    pipeline = MultiTaskFeatureSelectionPipeline(
        data_path=args.data,
        label_cols=args.labels,
        output_dir=args.output
    )

    # 执行
    if args.stage == 'all':
        results = pipeline.run_full_pipeline()
    else:
        # 运行单个stage
        stage_methods = {
            '1': pipeline.stage1_data_quality_check,
            '2': lambda: pipeline.stage2_leakage_detection(threshold=0.8),
            '3': lambda: pipeline.stage3_filter_methods(None),
            '4': lambda: pipeline.stage4_multitask_analysis(None),
            '5': lambda: pipeline.stage5_model_based_selection(None, top_k=150),
            '6': lambda: pipeline.stage6_stability_validation(None),
            '7': lambda: pipeline.stage7_domain_review(None),
        }
        results = stage_methods[args.stage]()

    print("\n✓ Pipeline execution complete!")
