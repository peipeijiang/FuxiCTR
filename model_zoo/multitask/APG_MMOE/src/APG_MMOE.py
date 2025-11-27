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
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block, FeatureEmbedding
from fuxictr.pytorch.torch_utils import get_activation
from model_zoo.APG.src.APG import APG_MLP


class APG_MMOE(MultiTaskModel):
    def __init__(self, 
                 feature_map, 
                 model_id="APG_MMOE", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 num_experts=4,
                 expert_hidden_units=[64, 64],
                 gate_hidden_units=[64],
                 tower_hidden_units=[32, 32],
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None,
                 hypernet_config={},
                 condition_features=[],
                 condition_mode="group-wise",
                 new_condition_emb=False,
                 rank_k=32,
                 overparam_p=1024,
                 generate_bias=True,
                 condition_participate_bottom=True,
                 **kwargs):
        super(APG_MMOE, self).__init__(feature_map, 
                                       model_id=model_id, 
                                       gpu=gpu, 
                                       embedding_regularizer=embedding_regularizer, 
                                       net_regularizer=net_regularizer,
                                       **kwargs)
        
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.condition_mode = condition_mode
        self.condition_features = condition_features
        self.condition_participate_bottom = condition_participate_bottom
        self.condition_emb_layer = None
        
        if condition_mode == "self-wise":
            condition_dim = None
        else:
            assert len(condition_features) > 0
            condition_dim = len(condition_features) * embedding_dim
            if new_condition_emb:
                self.condition_emb_layer = FeatureEmbedding(
                    feature_map, embedding_dim,
                    required_feature_columns=condition_features)

        self.input_dim = feature_map.sum_emb_out_dim()
        if not condition_participate_bottom:
             for feature in condition_features:
                 if feature in feature_map.features:
                     self.input_dim -= feature_map.features[feature]['emb_output_dim']
        
        self.experts = nn.ModuleList([
            APG_MLP(input_dim=self.input_dim,
                    hidden_units=expert_hidden_units,
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
            for _ in range(num_experts)
        ])

        self.gate = nn.ModuleList([
            MLP_Block(input_dim=self.input_dim,
                      hidden_units=gate_hidden_units,
                      output_dim=num_experts,
                      hidden_activations=hidden_activations,
                      output_activation=None,
                      dropout_rates=net_dropout,
                      batch_norm=batch_norm) 
            for _ in range(self.num_tasks)
        ])
        self.gate_activation = get_activation('softmax')

        self.tower = nn.ModuleList([
            APG_MLP(input_dim=expert_hidden_units[-1],
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
            for _ in range(self.num_tasks)
        ])
        
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
            
        experts_output = torch.stack([self.experts[i](feature_emb, condition_z) for i in range(len(self.experts))],
                                     dim=1)  # (?, num_experts, dim)
        
        mmoe_output = []
        for i in range(self.num_tasks):
            gate_output = self.gate[i](feature_emb)
            if self.gate_activation is not None:
                gate_output = self.gate_activation(gate_output)  # (?, num_experts)
            mmoe_output.append(torch.sum(torch.multiply(gate_output.unsqueeze(-1), experts_output), dim=1))
            
        tower_output = [self.tower[i](mmoe_output[i], condition_z) for i in range(self.num_tasks)]
        y_pred = [self.output_activation[i](tower_output[i]) for i in range(self.num_tasks)]
        
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict

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
