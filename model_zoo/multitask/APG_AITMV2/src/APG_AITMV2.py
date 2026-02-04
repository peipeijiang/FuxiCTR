import math
import torch
from torch import nn

from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation

from model_zoo.APG.src.APG import APG_MLP


class APG_CGC_Layer(nn.Module):
    def __init__(
        self,
        num_shared_experts,
        num_specific_experts,
        num_tasks,
        input_dim,
        expert_hidden_units,
        gate_hidden_units,
        hidden_activations,
        net_dropout,
        batch_norm,
        hypernet_config,
        condition_dim,
        condition_mode,
        rank_k,
        overparam_p,
        generate_bias,
    ):
        super().__init__()
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.num_tasks = num_tasks

        self.shared_experts = nn.ModuleList(
            [
                APG_MLP(
                    input_dim=input_dim,
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
                    generate_bias=generate_bias,
                )
                for _ in range(self.num_shared_experts)
            ]
        )

        self.specific_experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        APG_MLP(
                            input_dim=input_dim,
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
                            generate_bias=generate_bias,
                        )
                        for _ in range(self.num_specific_experts)
                    ]
                )
                for _ in range(num_tasks)
            ]
        )

        # Gate networks (DNN) output mixture weights over experts
        self.gate = nn.ModuleList(
            [
                MLP_Block(
                    input_dim=input_dim,
                    output_dim=(num_specific_experts + num_shared_experts) if i < num_tasks else num_shared_experts,
                    hidden_units=gate_hidden_units,
                    hidden_activations=hidden_activations,
                    output_activation=None,
                    dropout_rates=net_dropout,
                    batch_norm=batch_norm,
                )
                for i in range(num_tasks + 1)
            ]
        )
        self._gate_softmax_dim = -1

    def forward(self, x_list, condition_z=None):
        """CGC forward.

        Args:
            x_list: list of tensors, length == num_tasks + 1
                First num_tasks are per-task inputs; last is shared input.
            condition_z: optional condition tensor for APG experts.

        Returns:
            outputs: list of tensors, length == num_tasks + 1
        """
        specific_expert_outputs = []
        shared_expert_outputs = []

        # task-specific experts
        for task_id in range(self.num_tasks):
            task_outputs = []
            for expert in self.specific_experts[task_id].children():
                task_outputs.append(expert(x_list[task_id], condition_z))
            specific_expert_outputs.append(task_outputs)

        # shared experts
        shared_input = x_list[-1]
        for expert_id in range(self.num_shared_experts):
            shared_expert_outputs.append(self.shared_experts[expert_id](shared_input, condition_z))

        outputs = []
        for gate_id in range(self.num_tasks + 1):
            if gate_id < self.num_tasks:
                expert_stack = torch.stack(specific_expert_outputs[gate_id] + shared_expert_outputs, dim=1)
                gate_logits = self.gate[gate_id](x_list[gate_id])
                gate_weights = torch.softmax(gate_logits, dim=self._gate_softmax_dim)
                outputs.append(torch.sum(gate_weights.unsqueeze(-1) * expert_stack, dim=1))
            else:
                expert_stack = torch.stack(shared_expert_outputs, dim=1)
                gate_logits = self.gate[gate_id](x_list[-1])
                gate_weights = torch.softmax(gate_logits, dim=self._gate_softmax_dim)
                outputs.append(torch.sum(gate_weights.unsqueeze(-1) * expert_stack, dim=1))
        return outputs


class CascadedTransfer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        transfer_type="gated_residual",
        gate_hidden_units=(64,),
        hidden_activations="ReLU",
        net_dropout=0.0,
        batch_norm=False,
        use_prev_logit=True,
        detach_prev_rep=True,
        detach_prev_logit=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transfer_type = transfer_type
        self.use_prev_logit = use_prev_logit
        self.detach_prev_rep = detach_prev_rep
        self.detach_prev_logit = detach_prev_logit

        assert transfer_type in ["gated_residual", "attn"], f"Invalid transfer_type={transfer_type}"

        if transfer_type == "gated_residual":
            gate_in_dim = hidden_dim * 2 + (1 if use_prev_logit else 0)
            self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
            self.msg_act = nn.ReLU()
            self.gate_net = MLP_Block(
                input_dim=gate_in_dim,
                hidden_units=list(gate_hidden_units),
                output_dim=hidden_dim,
                hidden_activations=hidden_activations,
                output_activation=None,
                dropout_rates=net_dropout,
                batch_norm=batch_norm,
            )
            self.gate_act = nn.Sigmoid()
            self.ln = nn.LayerNorm(hidden_dim)
        else:
            # Keep a lightweight 2-token attention as a baseline
            self.prev_proj = nn.Linear(hidden_dim, hidden_dim)
            self.h1 = nn.Linear(hidden_dim, hidden_dim)
            self.h2 = nn.Linear(hidden_dim, hidden_dim)
            self.h3 = nn.Linear(hidden_dim, hidden_dim)
            self.scale = 1.0 / math.sqrt(hidden_dim)

    def forward(self, cur_h, prev_h, prev_logit=None):
        if self.detach_prev_rep:
            prev_h = prev_h.detach()

        if self.transfer_type == "gated_residual":
            msg = self.msg_act(self.msg_proj(prev_h))
            if self.use_prev_logit:
                if prev_logit is None:
                    raise ValueError("prev_logit is required when use_prev_logit=True")
                if self.detach_prev_logit:
                    prev_logit = prev_logit.detach()
                gate_inp = torch.cat([cur_h, msg, prev_logit], dim=-1)
            else:
                gate_inp = torch.cat([cur_h, msg], dim=-1)
            gate = self.gate_act(self.gate_net(gate_inp))
            out = self.ln(cur_h + gate * msg)
            return out

        # attn mode
        p = self.prev_proj(prev_h).unsqueeze(1)
        q = cur_h.unsqueeze(1)
        x = torch.cat([p, q], dim=1)  # (B, 2, D)
        V = self.h1(x)
        K = self.h2(x)
        Q = self.h3(x)
        att = torch.sum(K * Q, dim=2, keepdim=True) * self.scale
        w = torch.softmax(att, dim=1)
        return torch.sum(w * V, dim=1)


class APG_AITMV2(MultiTaskModel):
    def __init__(
        self,
        feature_map,
        model_id="APG_AITMV2",
        gpu=-1,
        learning_rate=1e-3,
        embedding_dim=10,
        num_tasks=3,
        num_layers=1,
        num_shared_experts=2,
        num_specific_experts=2,
        expert_hidden_units=(64, 64),
        gate_hidden_units=(64,),
        tower_type="dnn",
        tower_hidden_units=(32, 32),
        hidden_activations="ReLU",
        net_dropout=0,
        batch_norm=False,
        embedding_regularizer=None,
        net_regularizer=None,
        hypernet_config=None,
        condition_features=("product",),
        condition_mode="group-wise",
        new_condition_emb=True,
        condition_participate_bottom=True,
        rank_k=32,
        overparam_p=1024,
        generate_bias=True,
        # transfer
        transfer_type="gated_residual",
        transfer_gate_hidden_units=(64,),
        use_prev_logit=True,
        detach_prev_rep=True,
        detach_prev_logit=True,
        **kwargs,
    ):
        super().__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            num_tasks=num_tasks,
            **kwargs,
        )

        if hypernet_config is None:
            hypernet_config = {}

        self.num_tasks = num_tasks
        self.num_layers = num_layers
        self.tower_type = tower_type

        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)

        self.condition_mode = condition_mode
        self.condition_features = list(condition_features) if condition_features is not None else []
        self.condition_participate_bottom = condition_participate_bottom
        self.condition_emb_layer = None

        if self.condition_mode == "self-wise":
            condition_dim = None
        else:
            assert len(self.condition_features) > 0
            condition_dim = len(self.condition_features) * embedding_dim
            if new_condition_emb:
                self.condition_emb_layer = FeatureEmbedding(
                    feature_map, embedding_dim, required_feature_columns=self.condition_features
                )

        input_dim = feature_map.sum_emb_out_dim()
        if not self.condition_participate_bottom:
            for feature in self.condition_features:
                if feature in feature_map.features:
                    input_dim -= feature_map.features[feature]["emb_output_dim"]

        self.cgc_layers = nn.ModuleList(
            [
                APG_CGC_Layer(
                    num_shared_experts=num_shared_experts,
                    num_specific_experts=num_specific_experts,
                    num_tasks=num_tasks,
                    input_dim=input_dim if layer_id == 0 else int(expert_hidden_units[-1]),
                    expert_hidden_units=list(expert_hidden_units),
                    gate_hidden_units=list(gate_hidden_units),
                    hidden_activations=hidden_activations,
                    net_dropout=net_dropout,
                    batch_norm=batch_norm,
                    hypernet_config=hypernet_config,
                    condition_dim=condition_dim,
                    condition_mode=condition_mode,
                    rank_k=rank_k,
                    overparam_p=overparam_p,
                    generate_bias=generate_bias,
                )
                for layer_id in range(num_layers)
            ]
        )

        hidden_dim = int(expert_hidden_units[-1])
        self.transfer = CascadedTransfer(
            hidden_dim=hidden_dim,
            transfer_type=transfer_type,
            gate_hidden_units=transfer_gate_hidden_units,
            hidden_activations=hidden_activations,
            net_dropout=net_dropout,
            batch_norm=batch_norm,
            use_prev_logit=use_prev_logit,
            detach_prev_rep=detach_prev_rep,
            detach_prev_logit=detach_prev_logit,
        )

        if tower_type == "apg":
            self.tower = nn.ModuleList(
                [
                    APG_MLP(
                        input_dim=hidden_dim,
                        output_dim=1,
                        hidden_units=list(tower_hidden_units),
                        hidden_activations=hidden_activations,
                        output_activation=None,
                        dropout_rates=net_dropout,
                        batch_norm=batch_norm,
                        hypernet_config=hypernet_config,
                        condition_dim=condition_dim,
                        condition_mode=condition_mode,
                        rank_k=rank_k,
                        overparam_p=overparam_p,
                        generate_bias=generate_bias,
                    )
                    for _ in range(num_tasks)
                ]
            )
        elif tower_type == "dnn":
            self.tower = nn.ModuleList(
                [
                    MLP_Block(
                        input_dim=hidden_dim,
                        output_dim=1,
                        hidden_units=list(tower_hidden_units),
                        hidden_activations=hidden_activations,
                        output_activation=None,
                        dropout_rates=net_dropout,
                        batch_norm=batch_norm,
                    )
                    for _ in range(num_tasks)
                ]
            )
        else:
            raise ValueError(f"Invalid tower_type={tower_type}, expected 'dnn' or 'apg'.")

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def get_condition_z(self, X, feature_emb_dict):
        condition_z = None
        if self.condition_mode != "self-wise":
            if self.condition_emb_layer is not None:
                condition_z = self.condition_emb_layer(X, flatten_emb=True)
            else:
                condition_z = self.embedding_layer.dict2tensor(
                    feature_emb_dict, feature_list=self.condition_features, flatten_emb=True
                )
        return condition_z

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        condition_z = self.get_condition_z(X, feature_emb_dict)

        if self.condition_participate_bottom:
            feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)
        else:
            non_condition = [k for k in feature_emb_dict.keys() if k not in self.condition_features]
            feature_emb = self.embedding_layer.dict2tensor(
                feature_emb_dict, feature_list=non_condition, flatten_emb=True
            )

        cgc_inputs = [feature_emb for _ in range(self.num_tasks + 1)]
        for layer in self.cgc_layers:
            cgc_inputs = layer(cgc_inputs, condition_z)

        task_h = cgc_inputs[: self.num_tasks]

        tower_logits = []
        y_pred = []
        prev_h = None
        prev_logit = None

        for task_id in range(self.num_tasks):
            h = task_h[task_id]
            if task_id > 0:
                h = self.transfer(h, prev_h, prev_logit)

            if self.tower_type == "apg":
                logit = self.tower[task_id](h, condition_z)
            else:
                logit = self.tower[task_id](h)

            pred = self.output_activation[task_id](logit)

            tower_logits.append(logit)
            y_pred.append(pred)
            prev_h = h
            prev_logit = logit

        return_dict = {}
        labels = self.feature_map.labels
        for task_id in range(self.num_tasks):
            return_dict[f"{labels[task_id]}_pred"] = y_pred[task_id]
        return return_dict
