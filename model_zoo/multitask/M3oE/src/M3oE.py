import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict

class Weights(torch.nn.Module):
    def __init__(self, weight_shape, tau, tau_step, initial_deep, softmax_type=2):
        super().__init__()
        assert isinstance(weight_shape, (int, list))
        norm = weight_shape[-1] if isinstance(weight_shape, list) else weight_shape

        if initial_deep == None:
            initial_deep = np.ones(weight_shape, dtype=np.float32)/norm
        else:
            initial_deep = np.ones(weight_shape, dtype=np.float32)*initial_deep
        self.deep_weights = torch.nn.Parameter(torch.from_numpy(initial_deep), requires_grad=True)
        self.softmax_type = softmax_type
        self.tau = tau
        self.tau_step = tau_step

    def forward(self,):
        if self.tau > 0.01:
            self.tau -= self.tau_step
            
        if self.softmax_type == 0:
            return F.softmax(self.deep_weights, dim=-1)
        elif self.softmax_type == 1:
            output = F.softmax(self.deep_weights/self.tau, dim=-1)
            return output
        elif self.softmax_type == 2:
            output = F.gumbel_softmax(self.deep_weights, tau=self.tau, hard=False, dim=-1)
            return output
        elif self.softmax_type == 3:
            output = torch.sigmoid(self.deep_weights)
            return output
        else:
            raise ValueError('No such softmax_type')

class MLP_N(nn.Module):
    '''
    fcn_dim: list of dimensions of mlp layers, e.g., [1024, 512, 512, 256, 256, 64] 
    return [linear, bn1d, relu]*n
    '''
    def __init__(self, fcn_dim):
        super().__init__()
        self.fcn_dim = fcn_dim
        self.n = len(fcn_dim)
        
        self.domain_specific = nn.ModuleList()
        for (i) in range(self.n-1):
            self.domain_specific.append(nn.Linear(self.fcn_dim[i], self.fcn_dim[i+1]))
            self.domain_specific.append(nn.LayerNorm(self.fcn_dim[i+1]))
            self.domain_specific.append(nn.ReLU())            
        
    def forward(self, x):
        output = x
        for f in self.domain_specific:
            output = f(output)
        return output

class M3oE(MultiTaskModel):
    def __init__(self, 
                 feature_map, 
                 model_id="M3oE", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 num_tasks=2,
                 num_domains=2,
                 domain_feature='product',
                 expert_num=4,
                 expert_hidden_units=[512, 256, 64],
                 tower_hidden_units=[64],
                 exp_d=0.1,
                 exp_t=0.1,
                 bal_d=0.1,
                 bal_t=0.1,
                 tau=1,
                 tau_step=0.00005,
                 softmax_type=3,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(M3oE, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   embedding_regularizer=embedding_regularizer, 
                                   net_regularizer=net_regularizer,
                                   num_tasks=num_tasks,
                                   **kwargs)
        
        self.num_tasks = num_tasks
        self.num_domains = num_domains
        self.domain_feature = domain_feature
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        
        input_dim = feature_map.sum_emb_out_dim()
        
        # M3oE specific init
        # The original code assumes fcn_dims has at least 3 layers for Star
        # We construct fcn_dim from input_dim and expert_hidden_units
        self.fcn_dim = [input_dim] + expert_hidden_units
        
        assert len(self.fcn_dim) > 3, f'too few layers assigned, must larger than 3. Star owns 3 layers, mmoe owns the rest.'
        self.star_dim = self.fcn_dim[:3]
        self.fcn_dim_mmoe = self.fcn_dim[3:]
        self.expert_num = expert_num
        
        self._weight_exp_d = Weights(1, tau, tau_step, exp_d, softmax_type)
        self._weight_exp_t = Weights(1, tau, tau_step, exp_t, softmax_type)
        self._weight_bal_d = Weights(1, tau, tau_step, bal_d, softmax_type)
        self._weight_bal_t = Weights(1, tau, tau_step, bal_t, softmax_type)
        
        self.skip_conn = MLP_N([self.star_dim[0], self.star_dim[2]]) 
        self.shared_weight = nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1]))
        self.shared_bias = nn.Parameter(torch.zeros(self.star_dim[1]))
        
        self.slot_weight = nn.ParameterList([nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1])) for i in range(self.num_domains)])
        self.slot_bias = nn.ParameterList([nn.Parameter(torch.zeros(self.star_dim[1])) for i in range(self.num_domains)])
        
        self.star_mlp = MLP_N([self.star_dim[1], self.star_dim[2]])
        
        # Init weights
        torch.nn.init.xavier_uniform_(self.shared_weight.data)
        for m in (self.slot_weight):
            torch.nn.init.xavier_uniform_(m.data)
            
        # Experts
        self.expert = nn.ModuleList()
        for d in range(expert_num):
            self.expert.append(MLP_N(self.fcn_dim_mmoe))
            
        self.domain_expert = nn.ModuleList()
        for d in range(num_domains):
            self.domain_expert.append(MLP_N(self.fcn_dim_mmoe))
            
        self.task_expert = nn.ModuleList()
        for d in range(num_tasks):
            self.task_expert.append(MLP_N(self.fcn_dim_mmoe))
            
        # Gates
        # Input to gate is output of Star, which is star_dim[2]
        # star_dim[2] is also fcn_dim_mmoe[0]
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.fcn_dim_mmoe[0], expert_num), torch.nn.Softmax(dim=1)) for i in range(num_domains*num_tasks)])
        
        # Towers
        self.tower = nn.ModuleList()
        tower_input_dim = self.fcn_dim_mmoe[-1]
        
        for d in range(num_domains*num_tasks):
            domain_specific = nn.Sequential(
                nn.Linear(tower_input_dim, tower_input_dim),
                nn.LayerNorm(tower_input_dim),
                nn.ReLU(),
                nn.Linear(tower_input_dim, 1)
            )
            self.tower.append(domain_specific)
            
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        input_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)
        
        # Get domain_id
        domain_id = X[self.domain_feature].long() # [batch_size]
        
        _device = input_emb.device
        mask = []
        for d in range(self.num_domains):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

        skip = self.skip_conn(input_emb)
        emb = torch.zeros((input_emb.shape[0], self.star_dim[1])).to(_device)
        for i, (_weight, _bias) in enumerate(zip(self.slot_weight, self.slot_bias)):
            _output = torch.matmul(input_emb, torch.multiply(_weight, self.shared_weight))+_bias+self.shared_bias
            emb = torch.where(mask[i].unsqueeze(1).to(_device), _output, emb)
        emb = self.star_mlp(emb)+skip
        
        gate_value = [self.gate[i](emb.detach()).unsqueeze(1) for i in range(self.num_tasks*self.num_domains)] # [domain_num*task_num, batch_size, 1, expert_num]
        
        out = [] # batch_size, expert_num, embedding_size
        for i in range(self.expert_num):
            domain_input = self.expert[i](emb)
            out.append(domain_input)       
        domain_exp_out = []
        for i in range(self.num_domains):
            domain_input = self.domain_expert[i](emb)
            domain_exp_out.append(domain_input)  
        task_exp_out = []
        for i in range(self.num_tasks):
            task_input = self.task_expert[i](emb)
            task_exp_out.append(task_input)       
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim = 1) # batch_size, expert_num, 1
        domain_fea = torch.cat([domain_exp_out[i].unsqueeze(1) for i in range(self.num_domains)], dim = 1) # batch_size, domain_num, 1
        task_fea = torch.cat([task_exp_out[i].unsqueeze(1) for i in range(self.num_tasks)], dim = 1) # batch_size, task_num, 1

        weighted_domain_fea = []
        for i in range(self.num_domains):
            temp_ = self._weight_bal_d() * domain_fea[:, i, :]
            for j in range(self.num_domains):
                if j != i:
                    temp_ += (1-self._weight_bal_d())/(self.num_domains-1) * domain_fea[:, j, :]
            weighted_domain_fea.append(temp_)
        weighted_task_fea = []
        for i in range(self.num_tasks):
            temp_ = self._weight_bal_t() * task_fea[:, i, :]
            for j in range(self.num_tasks):
                if j != i:
                    temp_ += (1-self._weight_bal_t())/(self.num_tasks-1) * task_fea[:, j, :]
            weighted_task_fea.append(temp_)
        
        fused_fea = [torch.bmm(gate_value[i], fea).squeeze(1) + 
                     self._weight_exp_d() * weighted_domain_fea[i%self.num_domains] + 
                     self._weight_exp_t() * weighted_task_fea[i//self.num_domains]
                     for i in range(self.num_tasks*self.num_domains)]
        
        results = [self.tower[i](fused_fea[i]) for i in range(self.num_tasks*self.num_domains)]

        y_pred = []
        for t in range(self.num_tasks):
            task_domain_preds = []
            for d in range(self.num_domains):
                idx = t * self.num_domains + d
                task_domain_preds.append(results[idx])
            
            task_domain_preds = torch.stack(task_domain_preds, dim=1)
            pred = task_domain_preds.gather(1, domain_id.unsqueeze(1).unsqueeze(2)).squeeze(1)
            y_pred.append(torch.sigmoid(pred))
            
        return_dict = {}
        labels = self.feature_map.labels
        for i in range(self.num_tasks):
            return_dict["{}_pred".format(labels[i])] = y_pred[i]
        return return_dict
