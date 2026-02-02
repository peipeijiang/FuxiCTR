# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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


import torch.nn as nn
import numpy as np
import torch
import torch.distributed as dist
import os, sys
import logging
import math
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    logging.warning("TensorBoard not found. Visualization will be disabled.")
    SummaryWriter = None
from fuxictr.pytorch.layers import FeatureEmbeddingDict
from fuxictr.metrics import evaluate_metrics
from fuxictr.pytorch.torch_utils import get_device, get_optimizer, get_loss, get_regularizer, distributed_barrier
from fuxictr.utils import Monitor, not_in_whitelist
from tqdm import tqdm


class BaseModel(nn.Module):
    def __init__(self, 
                 feature_map, 
                 model_id="BaseModel", 
                 task="binary_classification", 
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
        super(BaseModel, self).__init__()
        self.device = get_device(gpu)
        self._distributed = bool(kwargs.get("distributed", False) and dist.is_available() and dist.is_initialized())
        self._distributed_rank = kwargs.get("distributed_rank", 0)
        self._distributed_world_size = max(1, kwargs.get("distributed_world_size", 1))
        self._is_master = (not self._distributed) or self._distributed_rank == 0
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._early_stop_patience = early_stop_patience
        self._eval_steps = eval_steps # None default, that is evaluating every epoch
        self._save_best_only = save_best_only
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._verbose = kwargs["verbose"]
        self.feature_map = feature_map
        self.output_activation = self.get_output_activation(task)
        self.task = task
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)

        # 为每个实验创建独立文件夹（v2.0 架构）
        # 新结构: model_root/{dataset_id}/{exp_id}/{exp_id}.model
        # 旧结构: model_root/{dataset_id}/{exp_id}.model (向后兼容)
        self.exp_dir = os.path.join(self.model_dir, self.model_id)
        os.makedirs(self.exp_dir, exist_ok=True)

        # 模型文件在实验文件夹内
        self.checkpoint = os.path.abspath(os.path.join(self.exp_dir, self.model_id + ".model"))

        # Epoch checkpoints 目录（可选）
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            try:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            except:
                pass  # 目录可能已存在或权限问题

        self.validation_metrics = kwargs["metrics"]
        self._nan_debug = bool(int(os.environ.get("FUXICTR_DEBUG_NAN", "1")))

        # TensorBoard 日志也在实验文件夹内
        if SummaryWriter and self._is_master:
            tb_log_dir = os.path.join(self.exp_dir, "tensorboard")
            self.writer = SummaryWriter(log_dir=tb_log_dir)
        else:
            self.writer = None

    def _nan_guard(self, name, value):
        """Detect non-finite values and raise with simple stats when debug flag is on."""
        if not self._nan_debug:
            return
        if value is None:
            return
        if torch.is_tensor(value):
            if value.numel() == 0:
                raise RuntimeError(f"[NaNGuard] {name} is empty. shape={tuple(value.shape)}")
            finite = torch.isfinite(value)
            if finite.all():
                return
            stats = {
                "shape": tuple(value.shape),
                "numel": value.numel(),
                "finite_count": int(finite.sum().item()),
                "nan_count": int(torch.isnan(value).sum().item()),
                "inf_count": int(torch.isinf(value).sum().item()),
            }
            if finite.any():
                stats["min"] = value[finite].min().item()
                stats["max"] = value[finite].max().item()
                stats["mean"] = value[finite].mean().item()
            raise RuntimeError(f"[NaNGuard] {name} has non-finite values. finite_stats={stats}")
        if isinstance(value, (list, tuple)):
            for idx, v in enumerate(value):
                self._nan_guard(f"{name}[{idx}]", v)
            return
        if isinstance(value, (float, int)) and not math.isfinite(value):
            raise RuntimeError(f"[NaNGuard] {name} is non-finite: {value}")

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        self.loss_fn = get_loss(loss, task=self.task)

    def regularization_loss(self):
        reg_term = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            emb_params = set()
            for m_name, module in self.named_modules():
                if type(module) == FeatureEmbeddingDict:
                    for p_name, param in module.named_parameters():
                        if param.requires_grad:
                            emb_params.add(".".join([m_name, p_name]))
                            for emb_p, emb_lambda in emb_reg:
                                reg_term += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if name not in emb_params:
                        for net_p, net_lambda in net_reg:
                            reg_term += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_term

    def add_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        return loss

    def compute_loss(self, return_dict, y_true):
        loss = self.add_loss(return_dict, y_true) + self.regularization_loss()
        return loss

    def reset_parameters(self):
        def default_reset_params(m):
            # initialize nn.Linear/nn.Conv1d layers by default
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        def custom_reset_params(m):
            # initialize layers with customized init_weights()
            if hasattr(m, 'init_weights'):
                m.init_weights()
        self.apply(default_reset_params)
        self.apply(custom_reset_params)

    def get_inputs(self, inputs, feature_source=None):
        X_dict = dict()
        for feature in inputs.keys():
            if feature in self.feature_map.labels:
                continue
            spec = self.feature_map.features[feature]
            if spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(spec["source"], feature_source):
                continue
            X_dict[feature] = inputs[feature].to(self.device)
        return X_dict

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        y = inputs[labels[0]].to(self.device)
        return y.float().view(-1, 1)
                
    def get_group_id(self, inputs):
        return inputs[self.feature_map.group_id]

    def model_to_device(self):
        self.to(device=self.device)

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            reduced_lr = max(param_group["lr"] * factor, min_lr)
            param_group["lr"] = reduced_lr
        return reduced_lr
           
    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._train_gen = data_generator  # 保存训练数据生成器用于评估
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.inf if self._monitor_mode == "min" else -np.inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch

        self._log("Start training: {} batches/epoch".format(self._steps_per_epoch))
        self._log("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self.train_epoch(data_generator)
            if self._stop_training:
                break
            else:
                # 每个 epoch 结束后评估训练集和验证集指标
                self._evaluate_epoch_end()
                self._log("************ Epoch={} end ************".format(self._epoch_index + 1))
        self._log("Training finished.")
        self._distributed_barrier()
        self._log("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)
        self._distributed_barrier()

    def checkpoint_and_earlystop(self, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
           (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            self._log("Monitor({})={:.6f} STOP!".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.lr_decay()
                self._log("Reduce learning rate on plateau: {:.6f}".format(current_lr))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                self._log("Save best model: monitor({})={:.6f}"\
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps >= self._early_stop_patience:
            self._stop_training = True
            self._log("********* Epoch={} early stop *********".format(self._epoch_index + 1))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)
        if self._distributed:
            self._distributed_barrier()

    def eval_step(self):
        self._log('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())
        self.checkpoint_and_earlystop(val_logs)
        self._sync_training_state()
        self.train()

    def _evaluate_epoch_end(self):
        """在每个 epoch 结束后评估训练集和验证集指标"""
        self._log('Evaluation @epoch {} end:'.format(self._epoch_index + 1))

        # 评估训练集指标
        train_logs = self.evaluate(self._train_gen, metrics=self._monitor.get_metrics())
        self._log("Train metrics: " + print_to_list(train_logs))

        # 评估验证集指标
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())
        self._log("Validation metrics: " + print_to_list(val_logs))

        # 使用验证集指标进行 checkpoint 和 early stopping
        self.checkpoint_and_earlystop(val_logs)
        self._sync_training_state()

        # 记录到 TensorBoard
        if self.writer:
            for metric_name, metric_value in train_logs.items():
                self.writer.add_scalar(f'epoch_train/{metric_name}', metric_value, self._epoch_index)
            for metric_name, metric_value in val_logs.items():
                self.writer.add_scalar(f'epoch_val/{metric_name}', metric_value, self._epoch_index)

        self.train()  # 恢复训练模式

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)

        # Catch empty batches early to provide actionable diagnostics
        if y_true is None or (torch.is_tensor(y_true) and y_true.numel() == 0):
            raise RuntimeError("[NaNGuard] Empty batch encountered: y_true has no elements. Check data loader / glob pattern.")
        
        # Calculate loss components for logging
        main_loss = self.add_loss(return_dict, y_true)
        reg_loss = self.regularization_loss()
        loss = main_loss + reg_loss

        # If loss is non-finite, log key stats before guard raises
        if torch.is_tensor(main_loss) and not torch.isfinite(main_loss):
            with torch.no_grad():
                y_pred = return_dict.get("y_pred")
                pred_stats = None
                if y_pred is not None and torch.is_tensor(y_pred) and y_pred.numel() > 0:
                    finite = torch.isfinite(y_pred)
                    pred_stats = {
                        "shape": tuple(y_pred.shape),
                        "numel": y_pred.numel(),
                        "finite": int(finite.sum().item()),
                        "nan": int(torch.isnan(y_pred).sum().item()),
                        "inf": int(torch.isinf(y_pred).sum().item()),
                    }
                    if finite.any():
                        pred_stats.update({
                            "min": y_pred[finite].min().item(),
                            "max": y_pred[finite].max().item(),
                            "mean": y_pred[finite].mean().item(),
                        })
                label_stats = None
                if torch.is_tensor(y_true) and y_true.numel() > 0:
                    finite = torch.isfinite(y_true)
                    label_stats = {
                        "shape": tuple(y_true.shape),
                        "numel": y_true.numel(),
                        "finite": int(finite.sum().item()),
                        "nan": int(torch.isnan(y_true).sum().item()),
                        "inf": int(torch.isinf(y_true).sum().item()),
                    }
                    if finite.any():
                        label_stats.update({
                            "min": y_true[finite].min().item(),
                            "max": y_true[finite].max().item(),
                            "mean": y_true[finite].mean().item(),
                        })
                self._log(f"[NaNGuard] main_loss non-finite. y_pred_stats={pred_stats} y_true_stats={label_stats}")

        self._nan_guard("y_pred", return_dict.get("y_pred"))
        self._nan_guard("y_true", y_true)
        self._nan_guard("main_loss", main_loss)
        self._nan_guard("reg_loss", reg_loss)
        self._nan_guard("loss", loss)
        
        loss.backward()
        self._sync_gradients()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss, main_loss, reg_loss, grad_norm

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0
        train_main_loss = 0
        train_reg_loss = 0
        train_grad_norm = 0
        
        self.train()
        if self._verbose == 0 or not self._is_master:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout, leave=False) # Fix newline issue
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss, main_loss, reg_loss, grad_norm = self.train_step(batch_data)
            
            if (self._verbose > 0 and self._is_master):
                batch_iterator.set_postfix(loss=loss.item(), main=main_loss.item(), reg=reg_loss.item(), grad=grad_norm.item())

            train_loss += loss.item()
            train_main_loss += main_loss.item()
            train_reg_loss += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
            train_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            if self._total_steps % self._eval_steps == 0:
                avg_loss = train_loss / self._eval_steps
                avg_main_loss = train_main_loss / self._eval_steps
                avg_reg_loss = train_reg_loss / self._eval_steps
                avg_grad_norm = train_grad_norm / self._eval_steps
                
                self._log("Train loss: {:.6f}".format(avg_loss))
                if self.writer:
                    self.writer.add_scalar('train/loss', avg_loss, self._total_steps)
                    self.writer.add_scalar('train/main_loss', avg_main_loss, self._total_steps)
                    self.writer.add_scalar('train/reg_loss', avg_reg_loss, self._total_steps)
                    self.writer.add_scalar('train/grad_norm', avg_grad_norm, self._total_steps)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self._total_steps)
                
                train_loss = 0
                train_main_loss = 0
                train_reg_loss = 0
                train_grad_norm = 0
                
                self.eval_step()
            if self._stop_training:
                break

    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            group_id = []
            if self._verbose > 0 and self._is_master:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                pred = return_dict["y_pred"]
                if self.task == "binary_classification_logits":
                    pred = torch.sigmoid(pred)
                y_pred.extend(pred.data.cpu().numpy().reshape(-1))
                y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None
            y_pred = self._gather_numpy(y_pred)
            y_true = self._gather_numpy(y_true)
            group_id = self._gather_numpy(group_id) if group_id is not None else None
            if self._is_master:
                if metrics is not None:
                    val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
                else:
                    val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
                log_str = '[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items())
                self._log(log_str)
                for handler in logging.root.handlers:
                    handler.flush()
                if self.writer:
                    for k, v in val_logs.items():
                        self.writer.add_scalar(f'val_{k}', v, self._total_steps)
            else:
                val_logs = {}
            val_logs = self._broadcast_logs(val_logs)
            return val_logs

    def predict(self, data_generator, gather_outputs=True):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            if self._verbose > 0 and self._is_master:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                pred = return_dict["y_pred"]
                if self.task == "binary_classification_logits":
                    pred = torch.sigmoid(pred)
                elif self._batch_index == 0 and self._is_master:
                     self._log("[Predict] Task={}, Sigmoid NOT applied. Pred range: {} to {}".format(self.task, pred.min().item(), pred.max().item()))
                y_pred.extend(pred.data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            if gather_outputs:
                y_pred = self._gather_numpy(y_pred)
            return y_pred

    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None, threshold=0.5):
        return evaluate_metrics(y_true, y_pred, metrics, group_id, threshold)

    def save_weights(self, checkpoint):
        if self._distributed and not self._is_master:
            return
        torch.save(self.state_dict(), checkpoint)
    
    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)

    def get_output_activation(self, task):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "binary_classification_logits":
            return nn.Identity()
        elif task == "regression":
            return nn.Identity()
        else:
            raise NotImplementedError("task={} is not supported.".format(task))

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters(): 
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        self._log("Total number of parameters: {}.".format(total_params))

    def _log(self, message):
        if self._is_master:
            logging.info(message)

    def _distributed_barrier(self):
        if self._distributed and dist.is_initialized():
            device = self.device if self.device.type == "cuda" else None
            distributed_barrier(device=device)

    def _broadcast_object(self, obj):
        if not self._distributed or not dist.is_initialized():
            return obj
        object_list = [obj] if self._is_master else [None]
        dist.broadcast_object_list(object_list, src=0)
        return object_list[0]

    def _broadcast_logs(self, logs):
        if not isinstance(logs, dict):
            return logs
        return self._broadcast_object(logs)

    def _sync_training_state(self):
        if not self._distributed or not dist.is_initialized():
            return
        state = {
            "stop": self._stop_training,
            "best_metric": self._best_metric,
            "stopping_steps": self._stopping_steps
        }
        synced_state = self._broadcast_object(state)
        if not self._is_master:
            self._stop_training = synced_state["stop"]
            self._best_metric = synced_state["best_metric"]
            self._stopping_steps = synced_state["stopping_steps"]

    def _sync_gradients(self):
        if not self._distributed or not dist.is_initialized():
            return
        for param in self.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(self._distributed_world_size)

    def _gather_tensor(self, tensor):
        if (not self._distributed) or (not dist.is_initialized()):
            return tensor
        if tensor is None:
            return None
        local_length = torch.tensor([tensor.shape[0]], device=self.device, dtype=torch.long)
        length_list = [torch.zeros_like(local_length) for _ in range(self._distributed_world_size)]
        dist.all_gather(length_list, local_length)
        max_len = int(torch.max(torch.stack(length_list)).item())
        pad_shape = (max_len,) + tuple(tensor.shape[1:])
        padded = torch.zeros(pad_shape, dtype=tensor.dtype, device=self.device)
        if tensor.shape[0] > 0:
            padded[:tensor.shape[0]] = tensor
        gather_list = [torch.zeros_like(padded) for _ in range(self._distributed_world_size)]
        dist.all_gather(gather_list, padded)
        lengths = [int(l.item()) for l in length_list]
        trimmed = [g[:l] for g, l in zip(gather_list, lengths) if l > 0]
        if len(trimmed) == 0:
            return torch.zeros((0,) + tuple(tensor.shape[1:]), dtype=tensor.dtype, device=self.device)
        return torch.cat(trimmed, dim=0)

    def _gather_numpy(self, array):
        if array is None:
            return None
        if (not self._distributed) or (not dist.is_initialized()):
            return array
        tensor = torch.from_numpy(array).to(self.device)
        gathered = self._gather_tensor(tensor)
        return gathered.data.cpu().numpy()

