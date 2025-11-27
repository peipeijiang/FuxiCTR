import torch
from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, FeatureEmbedding
from model_zoo.APG.src.APG import APG_MLP


class APG_AITM(MultiTaskModel):
    def __init__(self, 
                 feature_map, 
                 model_id="APG_AITM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 num_tasks=3,
                 bottom_hidden_units=[64, 64],
                 tower_hidden_units=[32, 32],
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None,
                 hypernet_config={},
                 condition_features=['product'],
                 condition_mode="group-wise",
                 new_condition_emb=True,
                 condition_participate_bottom=True,
                 rank_k=32,
                 overparam_p=1024,
                 generate_bias=True,
                 **kwargs):
        super(APG_AITM, self).__init__(feature_map, 
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
        if self.condition_mode == "self-wise":
            condition_dim = None
        else:
            assert len(self.condition_features) > 0
            condition_dim = len(self.condition_features) * embedding_dim
            if new_condition_emb:
                self.condition_emb_layer = FeatureEmbedding(
                    feature_map, embedding_dim,
                    required_feature_columns=self.condition_features)
        
        input_dim = feature_map.sum_emb_out_dim()
        if not self.condition_participate_bottom:
             for feature in self.condition_features:
                 if feature in feature_map.features:
                     input_dim -= feature_map.features[feature]['emb_output_dim']

        self.bottom = nn.ModuleList([
            APG_MLP(
                input_dim=input_dim,
                output_dim=bottom_hidden_units[-1],
                hidden_units=bottom_hidden_units,
                hidden_activations=hidden_activations,
                output_activation=None,
                dropout_rates=net_dropout,
                batch_norm=batch_norm,
                hypernet_config=hypernet_config,
                condition_dim=condition_dim,
                condition_mode=self.condition_mode,
                rank_k=rank_k,
                overparam_p=overparam_p,
                generate_bias=generate_bias
            ) for _ in range(num_tasks)
        ])
        self.tower = nn.ModuleList([
            APG_MLP(
                input_dim=bottom_hidden_units[-1],
                output_dim=1,
                hidden_units=tower_hidden_units,
                hidden_activations=hidden_activations,
                output_activation=None,
                dropout_rates=net_dropout,
                batch_norm=batch_norm,
                hypernet_config=hypernet_config,
                condition_dim=condition_dim,
                condition_mode=self.condition_mode,
                rank_k=rank_k,
                overparam_p=overparam_p,
                generate_bias=generate_bias
            ) for _ in range(num_tasks)
        ])
        self.hidden_dim = bottom_hidden_units[-1]
        self.g = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(num_tasks - 1)])
        self.h1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.h2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.h3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        condition_z = None
        if self.condition_mode != "self-wise":
            if self.condition_emb_layer is not None:
                condition_z = self.condition_emb_layer(X, flatten_emb=True)
            else:
                condition_z = self.embedding_layer.dict2tensor(feature_emb_dict,
                                                               feature_list=self.condition_features,
                                                               flatten_emb=True)
        if self.condition_participate_bottom:
            feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)
        else:
            non_condition = [k for k in feature_emb_dict.keys() if k not in self.condition_features]
            feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, feature_list=non_condition, flatten_emb=True)
        fea = [self.bottom[i](feature_emb, condition_z) for i in range(self.num_tasks)]
        for i in range(1, self.num_tasks):
            p = self.g[i - 1](fea[i - 1]).unsqueeze(1)
            q = fea[i].unsqueeze(1)
            x = torch.cat([p, q], dim=1)
            V = self.h1(x)
            K = self.h2(x)
            Q = self.h3(x)
            att = torch.sum(K * Q, 2, True) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32, device=V.device))
            w = torch.nn.functional.softmax(att, dim=1)
            fea[i] = torch.sum(w * V, 1)
        outs = [self.tower[i](fea[i], condition_z) for i in range(self.num_tasks)]
        y_pred = [torch.sigmoid(out) for out in outs]
        
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict
