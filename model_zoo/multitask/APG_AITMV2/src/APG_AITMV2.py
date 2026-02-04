import math
import torch
from torch import nn

from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation

from model_zoo.APG.src.APG import APG_MLP


# ============================================================
# APG_AITMV2: PLE/CGC Bottom + 单向级联 Transfer + Tower
#
# 设计动机（简述）：
# - Bottom: 使用 PLE/CGC 结构在“共享专家”和“任务专属专家”之间做可控共享；
# - Experts: 使用 APG_MLP（超网络）让专家参数随 condition_z（如 product 分组）自适应生成；
# - Transfer: 在任务之间按顺序做单向信息传递（漏斗/级联多任务常用），可通过 detach 断开梯度回流；
# - Tower: 每个任务一个 tower 输出 logit，再过 task 对应的 output_activation。
# ============================================================


class APG_CGC_Layer(nn.Module):
    """APG 版本的 CGC（PLE Bottom 的核心层）。

    CGC（Customized Gate Control）层包含：
    - 多个共享 experts：为所有任务提供公共表示；
    - 每个任务多个专属 experts：刻画该任务的特有模式；
    - 多个 gate 网络：
        - 每个任务一个 gate：在【本任务专属 experts + 共享 experts】上做加权融合；
        - shared 分支一个 gate：只在【共享 experts】上做加权融合，形成“纯共享表征”。

    这里 experts 使用 APG_MLP：其参数可由 condition_z（条件向量）动态生成。
    """

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

        # 共享 experts：所有任务共用。输入为 shared 分支的表示（x_list[-1]）。
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

        # 任务专属 experts：每个任务一组。
        # specific_experts[task_id][expert_id] 只服务于该任务 gate 的融合。
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

        # Gate networks（DNN）输出 experts 的混合权重。
        # - 前 num_tasks 个 gate：输出维度 = num_specific_experts + num_shared_experts
        # - 最后 1 个 shared gate：输出维度 = num_shared_experts
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

        # softmax 的维度，-1 表示对最后一维（expert 维度）做归一化。
        self._gate_softmax_dim = -1

    def forward(self, x_list, condition_z=None):
        """CGC 前向。

        Args:
            x_list: list[tensor]，长度 == num_tasks + 1。
                - x_list[0..num_tasks-1]：各任务分支输入（shape 通常为 (B, D)）
                - x_list[-1]：shared 分支输入（shape 通常为 (B, D)）
            condition_z: APG 的条件向量（可选），用于动态生成 expert 参数。

        Returns:
            outputs: list[tensor]，长度 == num_tasks + 1。
                - outputs[0..num_tasks-1]：各任务分支输出表示
                - outputs[-1]：shared 分支输出表示
        """
        specific_expert_outputs = []
        shared_expert_outputs = []

        # 1) 计算每个任务的专属 experts 输出：specific_expert_outputs[task_id] 是一个 list，长度为 num_specific_experts。
        for task_id in range(self.num_tasks):
            task_outputs = []
            for expert in self.specific_experts[task_id].children():
                task_outputs.append(expert(x_list[task_id], condition_z))
            specific_expert_outputs.append(task_outputs)

        # 2) 计算共享 experts 输出：对 shared 分支输入 x_list[-1] 过 num_shared_experts 个 expert。
        shared_input = x_list[-1]
        for expert_id in range(self.num_shared_experts):
            shared_expert_outputs.append(self.shared_experts[expert_id](shared_input, condition_z))

        # 3) 对每个 gate 分支做融合：
        # - 任务 gate：融合【本任务专属 experts + 共享 experts】
        # - shared gate：只融合【共享 experts】
        outputs = []
        for gate_id in range(self.num_tasks + 1):
            if gate_id < self.num_tasks:
                # 任务分支：把该任务的 specific experts 与 shared experts 堆叠成 (B, E, D)
                expert_stack = torch.stack(specific_expert_outputs[gate_id] + shared_expert_outputs, dim=1)
                # gate logits: (B, E)，softmax 后得到每个 expert 的权重
                gate_logits = self.gate[gate_id](x_list[gate_id])
                gate_weights = torch.softmax(gate_logits, dim=self._gate_softmax_dim)
                # 加权求和得到融合后的表示：(B, D)
                outputs.append(torch.sum(gate_weights.unsqueeze(-1) * expert_stack, dim=1))
            else:
                # shared 分支：只对共享 experts 做融合，形成“尽量不含任务私有信息”的共享表示
                expert_stack = torch.stack(shared_expert_outputs, dim=1)
                gate_logits = self.gate[gate_id](x_list[-1])
                gate_weights = torch.softmax(gate_logits, dim=self._gate_softmax_dim)
                outputs.append(torch.sum(gate_weights.unsqueeze(-1) * expert_stack, dim=1))
        return outputs


class CascadedTransfer(nn.Module):
    """单向级联（漏斗式）任务信息传递模块。

    用途：
    - 让 task_{t} 的表示 cur_h 在进入 tower 前，融合一部分来自 task_{t-1} 的信息（prev_h / prev_logit）。
    - 通过 detach_prev_rep / detach_prev_logit 控制是否切断梯度回流，保证“单向依赖”的训练稳定性。

    支持两种 transfer 形式：
    - gated_residual：门控残差（推荐默认）
    - attn：轻量 2-token attention（作为对照/基线）
    """

    def __init__(
        self,
        hidden_dim,
        transfer_type="gated_residual",
        gate_hidden_units=(64,),
        gate_net_type="dnn",
        hidden_activations="ReLU",
        net_dropout=0.0,
        batch_norm=False,
        use_prev_logit=True,
        detach_prev_rep=True,
        detach_prev_logit=True,
        # 以下参数仅在 gate_net_type="apg" 时生效
        hypernet_config=None,
        condition_dim=None,
        condition_mode="group-wise",
        rank_k=32,
        overparam_p=1024,
        generate_bias=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transfer_type = transfer_type
        self.use_prev_logit = use_prev_logit
        self.detach_prev_rep = detach_prev_rep
        self.detach_prev_logit = detach_prev_logit
        self.gate_net_type = gate_net_type

        assert transfer_type in ["gated_residual", "attn"], f"Invalid transfer_type={transfer_type}"

        if transfer_type == "gated_residual":
            # gated_residual：
            # - 将 prev_h 经过投影得到 msg
            # - gate 网络根据 [cur_h, msg, (optional) prev_logit] 输出逐维 gate
            # - 输出 out = LayerNorm(cur_h + gate * msg)
            gate_in_dim = hidden_dim * 2 + (1 if use_prev_logit else 0)
            self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
            self.msg_act = nn.ReLU()
            assert gate_net_type in ["dnn", "apg"], f"Invalid gate_net_type={gate_net_type}"
            if gate_net_type == "apg":
                if hypernet_config is None:
                    hypernet_config = {}
                self.gate_net = APG_MLP(
                    input_dim=gate_in_dim,
                    hidden_units=list(gate_hidden_units),
                    output_dim=hidden_dim,
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
            else:
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
            # attn：仅用 2 个 token（prev 和 cur）做注意力融合，作为轻量基线
            self.prev_proj = nn.Linear(hidden_dim, hidden_dim)
            self.h1 = nn.Linear(hidden_dim, hidden_dim)
            self.h2 = nn.Linear(hidden_dim, hidden_dim)
            self.h3 = nn.Linear(hidden_dim, hidden_dim)
            self.scale = 1.0 / math.sqrt(hidden_dim)

    def forward(self, cur_h, prev_h, prev_logit=None, condition_z=None):
        # 是否对上游任务的表示做 detach：
        # - True：梯度不会回流到 prev_h 对应的计算图（更符合“单向依赖”假设，也更稳）
        # - False：允许下游任务反向影响上游任务（可能更强但更容易互相干扰）
        if self.detach_prev_rep:
            prev_h = prev_h.detach()

        if self.transfer_type == "gated_residual":
            # 1) 先把 prev_h 投影成 msg（可理解为“待传递的信息”）
            msg = self.msg_act(self.msg_proj(prev_h))
            if self.use_prev_logit:
                if prev_logit is None:
                    raise ValueError("prev_logit is required when use_prev_logit=True")
                # 是否 detach 上游任务的 logit（同理用于切断梯度）
                if self.detach_prev_logit:
                    prev_logit = prev_logit.detach()
                # 拼接门控输入：cur_h、msg、prev_logit
                gate_inp = torch.cat([cur_h, msg, prev_logit], dim=-1)
            else:
                # 只使用 cur_h 与 msg
                gate_inp = torch.cat([cur_h, msg], dim=-1)
            # 2) gate 输出逐维权重（0~1），控制 msg 注入比例
            if self.gate_net_type == "apg":
                gate_logits = self.gate_net(gate_inp, condition_z)
            else:
                gate_logits = self.gate_net(gate_inp)
            gate = self.gate_act(gate_logits)
            # 3) 残差融合 + LayerNorm
            out = self.ln(cur_h + gate * msg)
            return out

        # attn 模式：prev/cur 两个 token 做 self-attention 融合
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
    """APG_AITMV2 主模型。

    结构：
    - Embedding：FeatureEmbeddingDict 输出各特征 embedding
    - Condition：可选抽取 condition_features 形成 condition_z（用于 APG 超网络）
    - Bottom：num_layers 层 APG_CGC_Layer（PLE/CGC）
    - Transfer：按 task_id 顺序做单向级联信息传递
    - Tower：每任务一个 tower 输出 logit，并使用 MultiTaskModel 中的 output_activation 得到预测
    """

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
        transfer_gate_net_type="dnn",
        use_prev_logit=True,
        detach_prev_rep=True,
        detach_prev_logit=True,
        **kwargs,
    ):
        # 初始化 MultiTaskModel：包含 device、loss/metric、output_activation、regularizer 等通用逻辑
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

    # ========= Embedding 层 =========
    # FeatureEmbeddingDict 返回一个 dict：{feature_name: embedding_tensor}
    # 后续可以用 dict2tensor(flatten_emb=True) 把所有 embedding 拼成 (B, D) 向量。
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)

    # ========= Condition（用于 APG 超网络） =========
    # condition_mode:
    # - self-wise：不需要 condition_z（APG 退化为固定参数网络/或内部自适应）
    # - group-wise / mix-wise：需要从 condition_features 拼出 condition_z
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
                # 可选：为 condition_features 单独建一套 embedding（避免与主 embedding_layer 参数绑定）
                self.condition_emb_layer = FeatureEmbedding(
                    feature_map, embedding_dim, required_feature_columns=self.condition_features
                )

        # ========= Bottom 输入维度 =========
        # feature_map.sum_emb_out_dim() 是所有特征 embedding 拼接后的维度。
        # 若 condition_participate_bottom=False，则从 bottom 输入中移除 condition_features 的 embedding。
        input_dim = feature_map.sum_emb_out_dim()
        if not self.condition_participate_bottom:
            for feature in self.condition_features:
                if feature not in feature_map.features:
                    continue
                feature_spec = feature_map.features[feature]
                # 与 FeatureMap.sum_emb_out_dim 的口径保持一致：meta 特征不参与 embedding 拼接
                if feature_spec.get("type") == "meta":
                    continue
                emb_out_dim = feature_spec.get(
                    "emb_output_dim",
                    feature_spec.get("embedding_dim", getattr(feature_map, "default_emb_dim", None) or embedding_dim),
                )
                if emb_out_dim is None:
                    raise KeyError(
                        f"Cannot infer embedding output dim for condition feature={feature}. "
                        "Please set emb_output_dim or embedding_dim in feature_map/feature_specs."
                    )
                input_dim -= int(emb_out_dim)

        # ========= CGC / PLE Bottom（可堆叠多层） =========
        # 第一层输入维度为 input_dim；后续层输入维度为 expert_hidden_units[-1]。
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

        # ========= 单向级联 Transfer =========
        # hidden_dim 与 expert 输出维度一致，作为 transfer 的表示维度。
        hidden_dim = int(expert_hidden_units[-1])
        self.transfer = CascadedTransfer(
            hidden_dim=hidden_dim,
            transfer_type=transfer_type,
            gate_hidden_units=transfer_gate_hidden_units,
            gate_net_type=transfer_gate_net_type,
            hidden_activations=hidden_activations,
            net_dropout=net_dropout,
            batch_norm=batch_norm,
            use_prev_logit=use_prev_logit,
            detach_prev_rep=detach_prev_rep,
            detach_prev_logit=detach_prev_logit,
            hypernet_config=hypernet_config,
            condition_dim=condition_dim,
            condition_mode=condition_mode,
            rank_k=rank_k,
            overparam_p=overparam_p,
            generate_bias=generate_bias,
        )

        # ========= Tower（每任务一个） =========
        # tower_type:
        # - dnn：普通 MLP_Block
        # - apg：APG_MLP（同样由 condition_z 控制参数生成）
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

        # ========= 编译与设备初始化 =========
        # compile 会根据 optimizer/loss 构造优化器与 loss_fn。
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def get_condition_z(self, X, feature_emb_dict):
        """构造 condition_z。

        - self-wise：返回 None
        - 其它模式：将 condition_features 的 embedding 拼接成一个条件向量 (B, condition_dim)
        """
        condition_z = None
        if self.condition_mode != "self-wise":
            if self.condition_emb_layer is not None:
                # 使用独立的 condition_emb_layer
                condition_z = self.condition_emb_layer(X, flatten_emb=True)
            else:
                # 复用主 embedding_layer 的结果，从 feature_emb_dict 中抽取并拼接
                condition_z = self.embedding_layer.dict2tensor(
                    feature_emb_dict, feature_list=self.condition_features, flatten_emb=True
                )
        return condition_z

    def forward(self, inputs):
        """前向计算。

        流程：
        1) 取模型输入并做 embedding
        2) 构造 condition_z（供 APG experts/tower 使用）
        3) 构造 bottom 输入 feature_emb（可选择是否包含 condition features）
        4) 经过多层 CGC 得到各任务表示 task_h
        5) 按任务顺序做级联 transfer（task_{t} 融合 task_{t-1} 信息）
        6) 进入每任务 tower 输出 logit，并得到 pred
        7) 返回 {"label_pred": pred} 字典
        """
        X = self.get_inputs(inputs)
        # 1) embedding：返回 dict，每个特征一个 embedding
        feature_emb_dict = self.embedding_layer(X)
        # 2) condition_z：给 APG 使用的条件向量
        condition_z = self.get_condition_z(X, feature_emb_dict)

        # 3) bottom 输入：决定 condition features 是否参与 bottom
        if self.condition_participate_bottom:
            # 所有特征 embedding 拼接为 (B, D)
            feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)
        else:
            # 移除 condition_features，只用非 condition 的 embedding 拼接
            non_condition = [k for k in feature_emb_dict.keys() if k not in self.condition_features]
            feature_emb = self.embedding_layer.dict2tensor(
                feature_emb_dict, feature_list=non_condition, flatten_emb=True
            )

        # CGC takes a list of branch inputs with length = num_tasks + 1:
        #   [task_0_input, ..., task_{T-1}_input, shared_input]
        # Each element is a tensor of shape (B, D), where D == input_dim for the first CGC layer.
        # NOTE: this creates a list of repeated references (no tensor copy); it is safe as long as
        # downstream modules do not perform in-place ops on the inputs.
        cgc_inputs = [feature_emb for _ in range(self.num_tasks + 1)]
        # 4) 多层 CGC：逐层更新各任务分支与 shared 分支表示
        for layer in self.cgc_layers:
            cgc_inputs = layer(cgc_inputs, condition_z)

        # task_h: list[tensor]，长度 num_tasks；每个 tensor 为该任务的 bottom 输出表示
        task_h = cgc_inputs[: self.num_tasks]

        tower_logits = []
        y_pred = []
        prev_h = None
        prev_logit = None

        # 5) 按 task_id 顺序做级联 transfer + tower
        for task_id in range(self.num_tasks):
            h = task_h[task_id]
            if task_id > 0:
                # task_{t} 融合 task_{t-1} 的信息（可选 detach 以保证单向梯度）
                h = self.transfer(h, prev_h, prev_logit, condition_z=condition_z)

            # 6) tower 输出 logit（每任务独立）
            if self.tower_type == "apg":
                logit = self.tower[task_id](h, condition_z)
            else:
                logit = self.tower[task_id](h)

            # 7) 输出层激活（由 MultiTaskModel 根据 task 类型决定，例如 sigmoid）
            pred = self.output_activation[task_id](logit)

            tower_logits.append(logit)
            y_pred.append(pred)
            prev_h = h
            prev_logit = logit

        # 8) 组装返回值：key 使用 feature_map.labels 中的标签名
        return_dict = {}
        labels = self.feature_map.labels
        for task_id in range(self.num_tasks):
            return_dict[f"{labels[task_id]}_pred"] = y_pred[task_id]
        return return_dict
