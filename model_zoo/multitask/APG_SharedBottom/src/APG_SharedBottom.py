import torch
from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, FeatureEmbedding
from model_zoo.APG.src.APG import APG_MLP


class APG_SharedBottom(MultiTaskModel):
    def __init__(self, 
                 feature_map, 
                 model_id="APG_SharedBottom", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 num_tasks=3,
                 bottom_hidden_units=[64, 64, 64],
                 tower_hidden_units=[32, 32],
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None,
                 hypernet_config={},
                 condition_features=[],
                 condition_mode="group-wise",
                 new_condition_emb=True,
                 rank_k=32,
                 overparam_p=1024,
                 generate_bias=True,
                 condition_participate_bottom=True,
                 **kwargs):
        super(APG_SharedBottom, self).__init__(feature_map, 
                                               model_id=model_id, 
                                               gpu=gpu, 
                                               embedding_regularizer=embedding_regularizer, 
                                               net_regularizer=net_regularizer,
                                               num_tasks=num_tasks,
                                               **kwargs)
        
        self.num_tasks = num_tasks
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

        input_dim = feature_map.sum_emb_out_dim()
        if not self.condition_participate_bottom:
             for feature in self.condition_features:
                 if feature in feature_map.features:
                     input_dim -= feature_map.features[feature]['emb_output_dim']
        
        self.shared_bottom = APG_MLP(input_dim=input_dim,
                                     output_dim=bottom_hidden_units[-1],
                                     hidden_units=bottom_hidden_units,
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
        
        self.towers = nn.ModuleList()
        for _ in range(num_tasks):
            tower = APG_MLP(input_dim=bottom_hidden_units[-1],
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
        shared_output = self.shared_bottom(feature_emb, condition_z)
        task_outputs = []
        for i, tower in enumerate(self.towers):
            task_output = tower(shared_output, condition_z)
            task_outputs.append(task_output)
        y_pred = [torch.sigmoid(out) for out in task_outputs]
        
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
