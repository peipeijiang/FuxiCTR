# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
import os, sys
import logging
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.torch_utils import get_device, get_optimizer, get_loss, get_regularizer
from tqdm import tqdm
from collections import defaultdict, deque


class AutomaticWeightedLoss(nn.Module):
    """Automatically weighted multi-task loss using uncertainty weighting.
    
    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    https://arxiv.org/abs/1705.07115
    
    Params:
        num: int, the number of losses
    
    Examples:
        loss1 = F.binary_cross_entropy(pred1, target1)
        loss2 = F.binary_cross_entropy(pred2, target2)
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)  # log variance parameters
    
    def forward(self, *losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.params[i])
            loss_sum += 0.5 * precision * loss + 0.5 * self.params[i]
        return loss_sum


class GradNorm(nn.Module):
    """Gradient Normalization for adaptive loss balancing in multi-task learning.
    
    Reference: "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
    (ICML 2018) http://proceedings.mlr.press/v80/chen18a/chen18a.pdf
    
    Args:
        num_tasks: int, number of tasks
        alpha: float, the strength of the restoring force (default=1.5)
        
    The method dynamically adjusts task weights by:
    1. Computing gradient norms for each task
    2. Balancing gradients based on relative training rates
    3. Using an asymmetry parameter alpha to control adaptation speed
    """
    def __init__(self, num_tasks=2, alpha=1.5):
        super(GradNorm, self).__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        # Initialize loss scale parameters
        self.loss_scale = nn.Parameter(torch.ones(num_tasks, requires_grad=True))
        # Buffer to store initial losses for computing relative training rates
        self.register_buffer('initial_losses', torch.zeros(num_tasks))
        self.register_buffer('has_initial_losses', torch.tensor(False))
        
    def get_loss_weights(self):
        """Get current loss weights (normalized)."""
        return F.softmax(self.loss_scale, dim=-1) * self.num_tasks


class MultiTaskModel(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="MultiTaskModel",
                 task=["binary_classification"],
                 num_tasks=1,
                 loss_weight='EQ',
                 gpu=-1,
                 monitor="AUC",
                 save_best_only=True,
                 monitor_mode="max",
                 early_stop_patience=2,
                 eval_steps=None,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 reduce_lr_on_plateau=True,
                 **kwargs):
        super(MultiTaskModel, self).__init__(feature_map=feature_map,
                                             model_id=model_id,
                                             task="binary_classification",
                                             gpu=gpu,
                                             loss_weight=loss_weight,
                                             monitor=monitor,
                                             save_best_only=save_best_only,
                                             monitor_mode=monitor_mode,
                                             early_stop_patience=early_stop_patience,
                                             eval_steps=eval_steps,
                                             embedding_regularizer=embedding_regularizer,
                                             net_regularizer=net_regularizer,
                                             reduce_lr_on_plateau=reduce_lr_on_plateau,
                                             **kwargs)
        self.label_col = kwargs.get("label_col", [])
        self.device = get_device(gpu)
        self.num_tasks = num_tasks
        self.loss_weight = loss_weight
        
        # Initialize loss weighting method
        self.awl = None
        self.gradnorm = None
        self.manual_weights = None
        
        # Parse loss_weight parameter
        if self.loss_weight == 'UW':  # UW = Uncertainty Weighting
            self.awl = AutomaticWeightedLoss(num_tasks)
            self.awl.to(self.device)
            self._log(f'Using Uncertainty Weighting for {num_tasks} tasks')
        elif self.loss_weight == 'GN':  # GN = GradNorm
            gradnorm_alpha = kwargs.get('gradnorm_alpha', 1.5)
            self.gradnorm = GradNorm(num_tasks, alpha=gradnorm_alpha)
            self.gradnorm.to(self.device)
            self._log(f'Using GradNorm (alpha={gradnorm_alpha}) for {num_tasks} tasks')
        elif isinstance(self.loss_weight, (list, tuple)):
            # Manual weights: e.g., [0.3, 0.7] or [1, 2]
            assert len(self.loss_weight) == num_tasks, \
                f"Manual weights length ({len(self.loss_weight)}) must equal num_tasks ({num_tasks})"
            self.manual_weights = torch.tensor(self.loss_weight, dtype=torch.float32, device=self.device)
            self._log(f'Using manual loss weights: {self.loss_weight}')
        elif self.loss_weight != 'EQ':
            self._log(f'Warning: Unknown loss_weight "{self.loss_weight}", using Equal Weighting (EQ)')
            self.loss_weight = 'EQ'
        
        if isinstance(task, list):
            if len(task) == 1 and num_tasks > 1:
                self.task_list = task * num_tasks
                self.output_activation = nn.ModuleList([self.get_output_activation(str(task[0])) for _ in range(num_tasks)])
            else:
                assert len(task) == num_tasks, "the number of tasks must equal the length of \"task\""
                self.task_list = task
                self.output_activation = nn.ModuleList([self.get_output_activation(str(t)) for t in task])
        else:
            self.task_list = [task] * num_tasks
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(task) for _ in range(num_tasks)]
            )

    def compile(self, optimizer, loss, lr):
        # Collect parameters for optimizer
        special_params = set()
        if self.awl is not None:
            special_params.update(id(p) for p in self.awl.parameters())
        if self.gradnorm is not None:
            special_params.update(id(p) for p in self.gradnorm.parameters())

        params_list = [{'params': [p for p in self.parameters() if id(p) not in special_params]}]
        
        # Add AWL/GradNorm parameters with no weight decay
        if self.awl is not None:
            params_list.append({'params': self.awl.parameters(), 'weight_decay': 0})
            self._log('Added Uncertainty Weighting parameters to optimizer (weight_decay=0)')
        elif self.gradnorm is not None:
            params_list.append({'params': self.gradnorm.parameters(), 'weight_decay': 0})
            self._log('Added GradNorm parameters to optimizer (weight_decay=0)')
        
        self.optimizer = get_optimizer(optimizer, params_list, lr)

        if isinstance(loss, list):
            self.loss_fn = [get_loss(l, task=t) for l, t in zip(loss, self.task_list)]
        else:
            self.loss_fn = [get_loss(loss, task=t) for t in self.task_list]

    def get_labels(self, inputs):
        """ Override get_labels() to use multiple labels """
        labels = self.feature_map.labels
        y = [inputs[labels[i]].to(self.device).float().view(-1, 1)
             for i in range(len(labels))]
        return y

    def regularization_loss(self):
        reg_loss = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            for _, module in self.named_modules():
                for p_name, param in module.named_parameters():
                    if param.requires_grad:
                        if p_name in ["weight", "bias"]:
                            if type(module) == nn.Embedding:
                                if self._embedding_regularizer:
                                    for emb_p, emb_lambda in emb_reg:
                                        reg_loss += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
                            else:
                                if self._net_regularizer:
                                    for net_p, net_lambda in net_reg:
                                        reg_loss += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_loss

    def add_loss(self, return_dict, y_true):
        labels = self.feature_map.labels
        loss_list = []
        for i in range(len(labels)):
            y_pred = return_dict["{}_pred".format(labels[i])]
            y_target = y_true[i]
            
            # Mask out labels with value -1
            mask = y_target != -1
            if mask.all():
                loss = self.loss_fn[i](y_pred, y_target, reduction='mean')
            else:
                y_pred_valid = y_pred[mask]
                y_target_valid = y_target[mask]
                if len(y_target_valid) > 0:
                    loss = self.loss_fn[i](y_pred_valid, y_target_valid, reduction='mean')
                else:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_list.append(loss)

        if self.loss_weight == 'GN':
            # For GradNorm, return individual losses
            return loss_list
        elif self.loss_weight == 'UW':
            # Use Uncertainty Weighting
            loss = self.awl(*loss_list)
        elif self.manual_weights is not None:
            # Use manual weights
            weighted_losses = [w * l for w, l in zip(self.manual_weights, loss_list)]
            loss = torch.sum(torch.stack(weighted_losses))
        elif self.loss_weight == 'EQ':
            # Default: All losses are weighted equally
            loss = torch.sum(torch.stack(loss_list))
        else:
            loss = loss_list
        return loss

    def compute_loss(self, return_dict, y_true):
        loss = self.add_loss(return_dict, y_true)
        if isinstance(loss, list):
            # For GradNorm, losses are returned as list
            loss = torch.sum(torch.stack(loss))
        loss = loss + self.regularization_loss()
        return loss
    
    def train_step(self, batch_data):
        """Override train_step to support GradNorm."""
        if self.loss_weight == 'GN' and self.gradnorm is not None:
            return self._train_step_gradnorm(batch_data)
        else:
            # Use default train_step from BaseModel
            return super().train_step(batch_data)
    
    def _train_step_gradnorm(self, batch_data):
        """Custom training step for GradNorm."""
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        
        # Get individual task losses
        loss_list = self.add_loss(return_dict, y_true)
        # Clone losses to avoid computation graph conflicts
        # We need two copies: one for GradNorm gradient computation,
        # and one for final parameter update
        loss_list_for_gradnorm = [loss.clone() for loss in loss_list]
        
        # Store initial losses in the first iteration
        if not self.gradnorm.has_initial_losses:
            with torch.no_grad():
                self.gradnorm.initial_losses = torch.tensor(
                    [loss.item() for loss in loss_list], 
                    device=self.device
                )
                self.gradnorm.has_initial_losses = torch.tensor(True)
        
        # Get current loss weights
        loss_weights = self.gradnorm.get_loss_weights()
        
        # Weighted sum of losses
        weighted_loss = sum(w * l for w, l in zip(loss_weights, loss_list))
        reg_loss = self.regularization_loss()
        main_loss = weighted_loss
        total_loss = main_loss + reg_loss
        
        # Get the last shared layer for gradient computation
        # We need to identify a shared representation layer
        last_shared_layer = self._get_last_shared_layer()

        if last_shared_layer is not None and self._epoch_index >= 1:
            # Compute gradients for each task using cloned losses
            # This avoids interfering with the final total_loss backward pass
            task_gradients = []
            for i, loss in enumerate(loss_list_for_gradnorm):
                # Retain graph for all but the last task
                retain = (i < len(loss_list_for_gradnorm) - 1)
                if last_shared_layer.weight.grad is not None:
                    last_shared_layer.weight.grad.zero_()
                
                loss.backward(retain_graph=True)
                
                if last_shared_layer.weight.grad is not None:
                    task_gradients.append(last_shared_layer.weight.grad.clone())
                else:
                    task_gradients.append(torch.zeros_like(last_shared_layer.weight))
            
            # Clear gradients before GradNorm update
            self.optimizer.zero_grad()
            
            # Stack gradients: [num_tasks, ...]
            if len(task_gradients) > 0:
                # Compute gradient norms for each task
                # Detach loss_weights to avoid creating a new computation graph
                grad_norms = torch.stack([
                    torch.norm(loss_weights[i].detach() * grad, p=2)
                    for i, grad in enumerate(task_gradients)
                ])
                
                # Average gradient norm
                mean_grad_norm = grad_norms.mean()
                
                # Compute relative inverse training rates
                with torch.no_grad():
                    loss_ratios = torch.tensor([
                        loss_list[i].item() / (self.gradnorm.initial_losses[i].item() + 1e-8)
                        for i in range(len(loss_list))
                    ], device=self.device)
                    inverse_train_rates = loss_ratios / (loss_ratios.mean() + 1e-8)
                
                # GradNorm loss: balance gradient norms
                target_grad_norms = mean_grad_norm * (inverse_train_rates ** self.gradnorm.alpha)
                gradnorm_loss = torch.abs(grad_norms - target_grad_norms.detach()).sum()
                
                # Backward pass for GradNorm
                gradnorm_loss.backward()
        
        # Final backward pass with updated weights
        self.optimizer.zero_grad()
        total_loss.backward()
        self._sync_gradients()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        
        return total_loss, main_loss, reg_loss, grad_norm
    
    def _get_last_shared_layer(self):
        """Get the last shared layer for GradNorm gradient computation.
        Override this method in specific models to identify the shared representation layer.
        """
        # Try to find common layer names in multi-task models
        for name, module in self.named_modules():
            if any(keyword in name.lower() for keyword in ['shared', 'bottom', 'embedding', 'dnn']):
                if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                    return module
        
        # Fallback: return the first Linear layer
        for module in self.modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                return module
        
        return None
    
    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred_all = defaultdict(list)
            y_true_all = defaultdict(list)
            labels = self.feature_map.labels
            group_id = []
            if self._verbose > 0 and self._is_master:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                batch_y_true = self.get_labels(batch_data)
                for i in range(len(labels)):
                    pred = return_dict["{}_pred".format(labels[i])]
                    if self.task_list[i] == "binary_classification_logits":
                        pred = torch.sigmoid(pred)
                    y_pred_all[labels[i]].extend(pred.data.cpu().numpy().reshape(-1))
                    y_true_all[labels[i]].extend(batch_y_true[i].data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
            group_id = np.array(group_id) if len(group_id) > 0 else None
            group_id = self._gather_numpy(group_id) if group_id is not None else None
            all_val_logs = {}
            mean_val_logs = defaultdict(list)

            for i in range(len(labels)):
                y_pred = np.array(y_pred_all[labels[i]], np.float64)
                y_true = np.array(y_true_all[labels[i]], np.float64)
                y_pred = self._gather_numpy(y_pred)
                y_true = self._gather_numpy(y_true)

                # Mask out labels with value not 0 or 1
                mask = (y_true == 0) | (y_true == 1)
                y_true = y_true[mask]
                y_pred = y_pred[mask]
                group_id_i = group_id[mask] if group_id is not None else None
                
                threshold = 0.5
                if self.label_col:
                    for col in self.label_col:
                        if col["name"] == labels[i]:
                            threshold = col.get("threshold", 0.5)
                            break

                if self._is_master:
                    if metrics is not None:
                        val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id_i, threshold)
                    else:
                        val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id_i, threshold)
                    self._log('[Task: {}][Metrics] '.format(labels[i]) + ' - '.join(
                        '{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
                    for k, v in val_logs.items():
                        all_val_logs['{}_{}'.format(labels[i], k)] = v
                        mean_val_logs[k].append(v)
            if self._is_master:
                for k, v in mean_val_logs.items():
                    mean_val_logs[k] = np.mean(v)
                all_val_logs.update(mean_val_logs)
            all_val_logs = self._broadcast_logs(all_val_logs)
            return all_val_logs

    def predict(self, data_generator, gather_outputs=True):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred_all = defaultdict(list)
            labels = self.feature_map.labels
            if self._verbose > 0 and self._is_master:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                for i in range(len(labels)):
                    pred = return_dict["{}_pred".format(labels[i])]
                    if self.task_list[i] == "binary_classification_logits":
                        pred = torch.sigmoid(pred)
                    y_pred_all[labels[i]].extend(pred.data.cpu().numpy().reshape(-1))
            for i in range(len(labels)):
                y_pred = np.array(y_pred_all[labels[i]], np.float64)
                if gather_outputs:
                    y_pred = self._gather_numpy(y_pred)
                y_pred_all[labels[i]] = y_pred
        return y_pred_all
