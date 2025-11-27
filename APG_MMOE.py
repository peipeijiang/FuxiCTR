import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union
from .basemodel import BaseModel
from .basemodel import FeatureEmbedding, FeatureEmbeddingDict
from .APG import APG_MLP


class APG_Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, hidden_activations, 
                 dropout_rates, batch_norm, hypernet_config, condition_dim, 
                 condition_mode, rank_k, overparam_p, generate_bias):
        super(APG_Expert, self).__init__()
        self.expert = APG_MLP(input_dim=input_dim,
                             output_dim=output_dim,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=dropout_rates,
                             batch_norm=batch_norm,
                             hypernet_config=hypernet_config,
                             condition_dim=condition_dim,
                             condition_mode=condition_mode,
                             rank_k=rank_k,
                             overparam_p=overparam_p,
                             generate_bias=generate_bias)
    
    def forward(self, x, condition_z=None):
        return self.expert(x, condition_z)


class APG_MMOE(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="APG_MMOE", 
                 gpu=-1,
                 device: Optional[str] = None,
                 num_tasks=3,
                 num_experts=4,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 expert_hidden_units=[64, 64],
                 tower_hidden_units=[32, 32],
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None,
                 hypernet_config={},
                 condition_features=[],
                 condition_mode="group-wise",  # 默认使用条件化权重，以避免忽略传入的condition_features
                 new_condition_emb=True,
                 rank_k=32,
                 overparam_p=1024,
                 generate_bias=True,
                 condition_participate_bottom: bool = True,
                 **kwargs):
        super(APG_MMOE, self).__init__(feature_map, 
                                       model_id=model_id, 
                                       gpu=gpu,
                                       device=device,
                                       embedding_regularizer=embedding_regularizer, 
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.condition_mode = condition_mode
        self.condition_features = condition_features
        self.condition_emb_layer = None
        self.condition_participate_bottom = condition_participate_bottom
        
        if condition_mode == "self-wise":
            condition_dim = None
        else:
            assert len(condition_features) > 0
            condition_dim = len(condition_features) * embedding_dim
            if new_condition_emb:
                self.condition_emb_layer = FeatureEmbedding(
                    feature_map, embedding_dim,
                    required_feature_columns=condition_features)
        
        total_feats = (len(self.embedding_layer.embeddings) + len(self.embedding_layer.sequence_embeddings))
        cond_count = 0
        for f in self.condition_features:
            if f in self.embedding_layer.embeddings or f in self.embedding_layer.sequence_embeddings:
                cond_count += 1
        effective_feats = total_feats if self.condition_participate_bottom else (total_feats - cond_count)
        input_dim = effective_feats * embedding_dim
        
        # Expert networks
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert = APG_Expert(input_dim=input_dim,
                               output_dim=expert_hidden_units[-1],
                               hidden_units=expert_hidden_units,
                               hidden_activations=hidden_activations,
                               dropout_rates=net_dropout,
                               batch_norm=batch_norm,
                               hypernet_config=hypernet_config,
                               condition_dim=condition_dim,
                               condition_mode=condition_mode,
                               rank_k=rank_k,
                               overparam_p=overparam_p,
                               generate_bias=generate_bias)
            self.experts.append(expert)
        
        # Gating networks for each task
        self.gating_networks = nn.ModuleList()
        for _ in range(num_tasks):
            gate = nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=1)
            )
            self.gating_networks.append(gate)
        
        # Task-specific towers
        self.towers = nn.ModuleList()
        for _ in range(num_tasks):
            tower = APG_MLP(input_dim=expert_hidden_units[-1],
                           output_dim=1,
                           hidden_units=tower_hidden_units,
                           hidden_activations=hidden_activations,
                           output_activation=None,
                           dropout_rates=net_dropout,
                           batch_norm=batch_norm,
                           hypernet_config=hypernet_config,
                           condition_dim=condition_dim,
                           condition_mode=condition_mode,
                           rank_k=rank_k,
                           overparam_p=overparam_p,
                           generate_bias=generate_bias)
            self.towers.append(tower)
        
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        condition_z = self.get_condition_z(X, feature_emb_dict)
        if self.condition_participate_bottom:
            feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)
        else:
            non_condition = [k for k in feature_emb_dict.keys() if k not in self.condition_features]
            feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, feature_list=non_condition, flatten_emb=True)
        if feature_emb.shape[1] == 0:
            debug_dims = {k: v["dim"] for k, v in self.feature_map.features.items()}
            raise RuntimeError(f"Feature embedding dimension is 0. 没有任何可用的嵌入特征。\n"
                               f"现有feature_map dims: {debug_dims}\n"
                               f"当前batch包含特征: {list(feature_emb_dict.keys())}")
        
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(feature_emb, condition_z)
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        task_outputs = []
        for gate, tower in zip(self.gating_networks, self.towers):
            gate_scores = gate(feature_emb)
            gate_scores = gate_scores.unsqueeze(2)
            weighted_expert_output = torch.sum(expert_outputs * gate_scores, dim=1)
            task_output = tower(weighted_expert_output, condition_z)
            task_outputs.append(task_output)
        y_pred = torch.cat(task_outputs, dim=-1)
        y_pred = torch.sigmoid(y_pred)
        return {"y_pred": y_pred}
    
    def get_condition_z(self, X, feature_emb_dict):
        condition_z = None
        if self.condition_mode != "self-wise":
            if self.condition_emb_layer is not None:
                condition_z = self.condition_emb_layer(X, flatten_emb=True)
            else:
                condition_z = self.embedding_layer.dict2tensor(feature_emb_dict, 
                                                               feature_list=self.condition_features,
                                                               flatten_emb=True)
        return condition_z

