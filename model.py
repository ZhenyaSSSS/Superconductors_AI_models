import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
import torchmetrics
from torch import Tensor
import gc
from timm.models.layers import DropPath
class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()
        self.act = nn.Mish()
    def forward(self, x):
      x, gate = x.chunk(2, dim = -1)
      return x * self.act(gate)
class SwiGLU_in(SwiGLU):
    def __init__(self):
        super(SwiGLU_in, self).__init__()
    def forward(self, x, gate):
      return x * self.act(gate)
class FFTLinear(nn.Module):
    def __init__(self,
                 d_model_input: int,
                 d_model_output: int,
                 **kwargs
                 ):
        super(FFTLinear, self).__init__()
        self.d_model_input = d_model_input
        self.d_model_output = d_model_output
        self.input_mul = nn.Parameter(torch.ones(d_model_input), requires_grad=True)
        self.deep_mul = nn.Parameter(torch.ones(d_model_output), requires_grad=True)
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        x = torch.fft.fft(x * self.input_mul, axis=-1, n=self.d_model_output)
        x = x * torch.fft.fft(self.deep_mul)
        x = torch.fft.ifft(x, axis=-1, n=x.shape[-1])
        return x.real

class FeedForwardModule(nn.Module):
    """
    FeedForwardModule applies a feed-forward neural network to the input data.
    """
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.0,
                 layer_norm: bool = True,
                 average_norm:bool = True,
                 **kwargs
                 ):
        """
        Initialize the FeedForwardModule.

        Args:
            d_model (int): Dimension of the model.
            scale (float): Scaling factor.
            type_act (list): Activation type.
            dropout (float, optional): Dropout probability.
            layer_norm (bool, optional): Whether to apply layer normalization.
        """
        super(FeedForwardModule, self).__init__()
        self.act = SwiGLU()
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model, d_model)
        if layer_norm:
            if average_norm:
                self.layer_norm = Norm()
            else:
                self.layer_norm = TokenNorm()
        else:
            self.layer_norm = nn.Identity()

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Forward pass of the FeedForwardModule.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the feed-forward neural network.
        """
        x = self.layer_norm(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
class Pos_bias(nn.Module):
    def __init__(self,
                 d_model: int,
                 **kwargs
                 ):
        super().__init__()
        self.pos_bias = nn.Parameter(torch.zeros(1, 1, d_model, d_model), requires_grad=True)
    def forward(self, shape, **kwargs):
        return torch.nn.functional.interpolate(self.pos_bias, size=[shape, shape], mode='bilinear')[0,0]
class AFTModule(nn.Module):
    def __init__(self,
                 d_model: int,
                 seq_len: int,
                 dropout: float = 0.0,
                 layer_norm: bool = False,
                 average_norm:bool = True,
                 **kwargs
                 ):
        """
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        * `seq_len` is $T$
        """
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.pos_bias = Pos_bias(seq_len)
        self.activation = nn.Sigmoid()
        self.output = nn.Linear(d_model, d_model)#nn.Linear(d_model, d_model)
        if layer_norm:
            if average_norm:
                self.layer_norm = Norm()
            else:
                self.layer_norm = TokenNorm()
        else:
            self.layer_norm = nn.Identity()
    def forward(self, data: torch.Tensor, **kwargs):
        """
        `query`, `key` and `value` are the tensors that store
        collection of token embeddings for  *query*, *key* and *value*.
        They have shape `[batch_size, seq_len, d_model]`.
        `mask` has shape `[seq_len, seq_len, batch_size]` and
        `mask[i, j, b]` indicates whether for batch `b`,
        query at position `i` has access to key-value at position `j`.
        """
        query, key, value = self.qkv(self.layer_norm(data)).chunk(3, dim = -1)
        pos_bias = self.pos_bias(len(data))
        pos_bias = torch.exp(pos_bias.unsqueeze(-1))
        key = torch.exp(key)
        num = torch.einsum('ijb,jbd->ibd', pos_bias, torch.mul(key, value))
        weighted = num / torch.einsum('ijb,jbd->ibd', pos_bias, key)
        output = torch.mul(self.activation(query), weighted)
        return self.output(output)
class Residual(nn.Module):
    def __init__(self, 
                 **kwargs):
        super().__init__()

    def forward(self, data, residual, **kwargs):
        for i in data:
          residual = residual + i
        return residual
    
class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
    def forward(self, x, **kwargs):
        return torch.nn.functional.layer_norm(x, [x.shape[-1]], eps=torch.finfo(torch.float32).tiny)
class TokenNorm(nn.Module):
    def __init__(self, eps=torch.finfo(torch.float32).tiny):
        super(TokenNorm, self).__init__()
        self.eps = torch.finfo(torch.float32).tiny
    def forward(self, x, **kwargs):
        x = x - x.mean(dim=[0,2], keepdim=True)
        return x / (x.std(dim=[0,2], keepdim=True) + self.eps)
class Layer(nn.Module):
    def __init__(self,
                 module_configuration,
                 layer_norm,
                 grad_checkpointing=False,
                 drop_path=0.0,
                 dropout=0.0,
                 ):
        super(Layer, self).__init__()
        self.module_configuration = module_configuration
        self.modules_list = nn.ModuleList()
        self._initialize_modules()
        self.res_attention = Residual()
        self.grad_checkpointing = grad_checkpointing
    def forward(self, data, **kwargs):
        residual = data.clone()
        module_outputs = []
        for module, module_config in zip(self.modules_list, self.module_configuration):
#             if "pre_permute" in module_config["params"]:
#                 input_data = input_data.permute(*module_config["params"]["pre_permute"])

            if self.grad_checkpointing:
              output_data = torch.utils.checkpoint.checkpoint(module, data, **kwargs)
            else:
              output_data = module(data, **kwargs)

            if "post_permute" in module_config["params"]:
                output_data = output_data.permute(*module_config["params"]["post_permute"])

            module_outputs.append(output_data)

        return self.res_attention(module_outputs, residual)
    def _initialize_modules(self):
        module_mapping = {
            "FeedForwardModule": FeedForwardModule,
            "AFTModule": AFTModule,
        }
        wrapper_mapping = {
            "PatchWrapper": None,
        }

        for module_config in self.module_configuration:

            module_type = module_config["type"]
            module_params = module_config["params"]
            module_count = module_config["count"]

            if module_type not in module_mapping:
                raise ValueError(f"Unknown module type: {module_type}")

            module_class = module_mapping[module_type]
            for _ in range(module_count):
                if "wrapper" in module_config:
                  wrapper_type = module_config["wrapper"]["type"]
                  if wrapper_type not in wrapper_mapping:
                    raise ValueError(f"Unknown wrapper type: {wrapper_type}")
                  wrapper_class = wrapper_mapping[wrapper_type]
                  module = wrapper_class(module_class(**module_params), **module_config["wrapper"])
                else:
                  module = SequentialModule([RegulationLayer(p_GaussianDropout_Mul=0.0, p_GaussianDropout_Add=0.0, p_DropPath=0.0, DropPath_dims=[1], p_Dropout=0.0),
                                         module_class(**module_params),
                                         RegulationLayer(p_GaussianDropout_Mul=0.0, p_GaussianDropout_Add=0.0, p_DropPath=0.0, DropPath_dims=[1], p_Dropout=0.0)
                                            ])
                self.modules_list.append(module)
class SequentialModule(nn.Module):
    def __init__(self, modules):
        super(SequentialModule, self).__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, x, **kwargs):
        for module in self.module_list:
            x = module(x, **kwargs)
        return x
class ConstantBlock(nn.Module):
    def __init__(self,
                 module_configuration,
                 num_layers,
                 layer_norm=True,
                 unet=True,
                 grad_checkpointing=False,
                 drop_path=0.0,
                 dropout=0.0,
                 shuffle=False,
                 ):
        super(ConstantBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.extend([Layer(module_configuration, layer_norm, grad_checkpointing, drop_path, dropout) for _ in range(num_layers)])
        if unet:
          self.res = Residual()
        else:
          self.res = None
    def forward(self, x, shuffle=False, **kwargs):
        if shuffle:
          random.shuffle(self.layers)
        if self.res != None:
          data = []
          residual = x.clone()
          for layer in (self.layers):
            x = layer(x, **kwargs)
            data.append(x)
          return self.res(data, residual, **kwargs)
        else:
          for layer in (self.layers):
            x = layer(x, **kwargs)
          return x
class Embedding(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embedding, self).__init__()
        self.n_token = n_token
        self.d_model = d_model
        self.embed_scale = d_model ** 0.5
        self.layer_norm = TokenNorm()
        self.emb = nn.Embedding(n_token, d_model)
    def forward(self, inp, **kwargs):
        embed = self.emb(inp)
        return self.layer_norm(embed.mul_(self.embed_scale).permute(1,0,2))
class RegulationLayer(nn.Module):
    def __init__(self, p_GaussianDropout_Mul=0.0, p_GaussianDropout_Add=0.0, p_DropPath=0.5, DropPath_dims=[0], p_Dropout=0.0):
        super(RegulationLayer, self).__init__()
        if p_GaussianDropout_Mul < 0 or p_GaussianDropout_Mul >= 1:
            raise Exception("p_GaussianDropout_Mul value should accomplish 0 < p_GaussianDropout_Mul < 1")
        if p_GaussianDropout_Add < 0 or p_GaussianDropout_Add >= 1:
            raise Exception("p_GaussianDropout_Add value should accomplish 0 < p_GaussianDropout_Add < 1")
        if p_DropPath < 0 or p_DropPath >= 1:
            raise Exception("p_DropPath value should accomplish 0 < p_DropPath < 1")
        if p_Dropout < 0 or p_Dropout >= 1:
            raise Exception("p_Dropout value should accomplish 0 < p_Dropout < 1")
        self.p_GaussianDropout_Mul = p_GaussianDropout_Mul
        self.p_GaussianDropout_Add = p_GaussianDropout_Add
        self.p_DropPath = p_DropPath
        self.DropPath_dims = DropPath_dims
        self.p_Dropout = p_Dropout
    def forward(self, x, freeze=False, skip=False, **kwargs):
        if self.training and not skip:
            if self.p_GaussianDropout_Mul != 0:
                if not freeze:
                    stddev = (self.p_GaussianDropout_Mul / (1.0 - self.p_GaussianDropout_Mul))**0.5
                    self.epsilon_mul = torch.randn_like(x) * stddev
                x = x * self.epsilon_mul
            if self.p_GaussianDropout_Add != 0:
                if not freeze:
                    stddev = (self.p_GaussianDropout_Add / (1.0 - self.p_GaussianDropout_Add))**0.5
                    self.epsilon_add = torch.randn_like(x) * stddev
                x = x + self.epsilon_add
            if self.p_DropPath != 0:
                if not freeze:
                    shape = list(x.shape)
                    for i in range(len(shape)):
                        if i not in self.DropPath_dims:
                            shape[i] = 1
                    self.mask_DropPath = x.new_empty(shape).bernoulli_(self.p_DropPath)
#                     self.mask_DropPath.div_(self.p_DropPath)
                x = x * self.mask_DropPath
            if self.p_Dropout != 0:
                if not freeze:
                    shape = list(x.shape)
                    self.mask_Dropout = x.new_empty(shape).bernoulli_(self.p_Dropout)
                    self.mask_Dropout.div_(self.p_Dropout)
                x = x * self.mask_Dropout
        return x
    def __repr__(self):
        return f"{self.__class__.__name__}(p_GaussianDropout_Mul={self.p_GaussianDropout_Mul})"
      
class Encoder(nn.Module):
    def __init__(self,
                 seq_len=32,
                 num_layers=4,
                 layer_norm=True,
                 unet=True,
                 grad_checkpointing=False,
                 drop_path=0.0,
                 dropout=0.0,
                 shuffle=False,
                 d_models=[512, 256, 128, 64, 32]
                ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            ConstantBlock(module_configuration = [
                {"type": "FeedForwardModule", "params": {"d_model": d, "dropout": dropout, "layer_norm": layer_norm}, "count": 1},
                {"type": "AFTModule", "params": {"d_model": d, "seq_len": seq_len, "dropout": dropout, "layer_norm": layer_norm}, "count": 1},
 ], 
                 num_layers=num_layers, 
                 unet=unet,
                 grad_checkpointing=grad_checkpointing,
                 drop_path=drop_path,
                 dropout=dropout,
                         )
            for d in d_models
        ])
        
        self.downsamples = nn.ModuleList([
            nn.Linear(d_models[i], d_models[i+1])
            for i in range(len(d_models)-1)
        ])
        self.norm = Norm()
        
    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.blocks) - 1:
                x = self.norm(self.downsamples[i](x))
        return x


class Decoder(nn.Module):
    def __init__(self,
                 seq_len=32,
                 num_layers=4,
                 layer_norm=True,
                 unet=True,
                 grad_checkpointing=False,
                 drop_path=0.0,
                 dropout=0.0,
                 shuffle=False,
                 d_models=[512, 256, 128, 64, 32]
                ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            ConstantBlock(module_configuration = [
                {"type": "FeedForwardModule", "params": {"d_model": d, "dropout": dropout, "layer_norm": layer_norm}, "count": 1},
                {"type": "AFTModule", "params": {"d_model": d, "seq_len": seq_len, "dropout": dropout, "layer_norm": layer_norm}, "count": 1},
 ], 
                 num_layers=num_layers, 
                 unet=unet,
                 grad_checkpointing=grad_checkpointing,
                 drop_path=drop_path,
                 dropout=dropout,
                         )
            for d in d_models[::-1]
        ])
        
        self.upsamples = nn.ModuleList([
            nn.Linear(d_models[i], d_models[i-1]) 
            for i in range(len(d_models)-1, 0, -1)
        ])
        self.norm = Norm()
        
    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.blocks) - 1:
                x = self.norm(self.upsamples[i](x))
        return x
    def get_hidden(self, x):
        hidden = [x]
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.blocks) - 1:
                x = self.norm(self.upsamples[i](x))
                hidden.append(x)
        return hidden
        
class NumberModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.00002,
        batch_size: int = 256,
        batch_size_step: int = 256,
        seed_value: int = 1,
        dropout: float = 0.0,
        d_model: int = 16,
        seq_len: int = 32,
        layer_norm: bool = True,
        drop_path: float = 0.0,
        unet: bool = True,
        grad_checkpointing: bool = False,
        num_layers: int = 12,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        if seed_value != None:
          pl.seed_everything(seed_value)
          torch.manual_seed(seed_value)
          torch.cuda.manual_seed(seed_value)
          random.seed(seed_value)
#         self.main = ConstantBlock(module_configuration = [
#                 {"type": "FeedForwardModule", "params": {"d_model": d_model, "dropout": dropout, "layer_norm": layer_norm, "average_norm": False}, "count": 1},
#                 {"type": "AFTModule", "params": {"d_model": d_model, "seq_len": seq_len, "dropout": dropout, "layer_norm": layer_norm, "average_norm": False}, "count": 1},
#  ], 
#                  num_layers=num_layers, 
#                  unet=unet,
#                  grad_checkpointing=grad_checkpointing,
#                  drop_path=drop_path,
#                  dropout=dropout,
#                                  )
        self.head_mantissa = nn.Linear(512, 1, bias=False)
        self.head_tanh_value = nn.Linear(512, 1, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.head_exponent = nn.Linear(512, 2**8)
        self.head_log10 = nn.Linear(512, 128)
        self.head_decoder = nn.Linear(512, 32 * 2)
        self.pool = nn.Flatten(1)
        self.embedding = Embedding(2, 16)
        self.pos_embedding = nn.Parameter(torch.randn(32, 1, 16))
        self.norm = Norm() #nn.Identity()#
#         self.post_norm = Norm()
        
        self.sin_head = nn.Linear(512, 8, bias=False)
        self.post_net = ConstantBlock(module_configuration = [
                {"type": "FeedForwardModule", "params": {"d_model": 512, "dropout": dropout, "layer_norm": layer_norm, "average_norm": True}, "count": 1},
 ], 
                 num_layers=num_layers, 
                 unet=unet,
                 grad_checkpointing=grad_checkpointing,
                 drop_path=drop_path,
                 dropout=dropout,
                                 )
        self.sq_head = nn.Linear(512, 2*32, bias=False)
        self.log2_head = nn.Linear(512, 2*32, bias=False)
        self.log10_head = nn.Linear(512, 2*32, bias=False)
        self.int_n_head = nn.Linear(512, 2*64, bias=False)
        
    def forward(self, data):
        data = self.embedding(data) + self.pos_embedding
        data = self.pool(data.permute(1,2,0))
        data = self.norm(self.post_net(data))
        return data, self.sigmoid(self.head_mantissa(data)), self.head_exponent(data), self.tanh(self.head_tanh_value(data)), self.head_log10(data), self.head_decoder(data)
    def get_features(self, data):
        data = self.embedding(data) + self.pos_embedding
        data = self.pool(data.permute(1,2,0))
        data = self.norm(self.post_net(data))
        return data
    def on_validation_epoch_end(self):
        gc.collect()
        self.trainer.save_checkpoint(filepath="/kaggle/working/checkpoint.ckpt")
    def binary(self, x, bits):
        mask = 2**torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).long()
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        if batch_idx == 0:
            optimizer.lr[0].data = optimizer.lr[0].data - optimizer.lr[0].data + 0.000000025
            self.hparams.batch_size_step = 2048*8
        # Generate random bits tensor
        random_bits = torch.randint(0, 2, (self.hparams.batch_size_step, 32), device=self.device) 

        # Convert to float
        random_floats = bits_to_float(random_bits, 23, 8).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        sq = random_floats[:,None].abs().sqrt()
        log2 = random_floats[:,None].abs().log2()
        log10 = random_floats[:,None].abs().log10() #float_to_bits(log10, 23, 8)
        int_n = random_floats[:,None].long() 
        
        add_info = torch.cat([random_floats[:,None].sin().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0), random_floats[:,None].cos().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0),
                              log2.sin().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0), log10.sin().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0),
                              log2.cos().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0), log10.cos().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0),
                              sq.sin().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0), sq.cos().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                             ],dim=-1)
        
        sq = float_to_bits(sq[:,0], 23, 8)
        log2 = float_to_bits(log2[:,0].nan_to_num(nan=0.0, posinf=0.0, neginf=0.0), 23, 8)
        log10 = float_to_bits(log10[:,0].nan_to_num(nan=0.0, posinf=0.0, neginf=0.0), 23, 8)
        int_n = self.binary(int_n[:,0], 64)

        # Get mantissa and exponent
        float_bits, mantissa, exponent = float_to_bits(random_floats, 23, 8, retun_ME=True)

        # Calculate x and y 
        x, y = self.calc_xy(random_floats)

#         # Split tensor in half
#         random_floats1 = random_floats[:self.hparams.batch_size_step//2] 
#         random_floats2 = random_floats[self.hparams.batch_size_step//2:]

#         # Calculate sum and difference
#         sum_floats = random_floats1 + random_floats2
#         diff_floats = random_floats1 - random_floats2

#         # Get features for sum and diff
#         sum_bits, sum_mantissa, sum_exponent = float_to_bits(sum_floats, 23, 8, retun_ME=True)
#         diff_bits, diff_mantissa, diff_exponent = float_to_bits(diff_floats, 23, 8, retun_ME=True)

#         sum_x, sum_y = self.calc_xy(sum_floats)

#         diff_x, diff_y = self.calc_xy(diff_floats)
#         if random.uniform(0, 1) > 10.4:
#             logit, mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out = self(float_bits)
#             loss_1 = self.calc_loss(mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out, mantissa, exponent, x, y, float_bits)
# #             loss_sum = self.calc_loss(mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out, mantissa, exponent, x, y, float_bits)

#             self.log('Loss/Train', loss_1, prog_bar=True, on_epoch=True)
#             self.manual_backward(loss_1)
#             optimizer.first_step(zero_grad=True)
#             logit, mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out = self(float_bits)
#             loss_2 = self.calc_loss(mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out, mantissa, exponent, x, y, float_bits)

#             self.log('Loss/Train_SAM', loss_2, prog_bar=True, on_epoch=True)
#             self.manual_backward(loss_2)
#             optimizer.second_step(zero_grad=True)
#         else:
        logit, mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out = self(float_bits)
        loss_1 = self.calc_loss(mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out, mantissa, exponent, x, y, float_bits)
        info_logit = self.tanh(self.sin_head(logit))
        info_loss = F.mse_loss(info_logit, add_info) + F.l1_loss(info_logit, add_info)
        for i in range(8):
            self.log('InfoLoss/' + str(i), F.l1_loss(info_logit[:, i], add_info[:, i]), prog_bar=True, on_epoch=True)
            self.log('Real/' + str(i), add_info[0, i], prog_bar=True, on_epoch=True)
            self.log('Fake/' + str(i), info_logit[0, i], prog_bar=True, on_epoch=True)
        
        sq_logit = self.sq_head(logit)
        log2_logit = self.log2_head(logit)
        log10_logit = self.log10_head(logit)
        int_n_logit = self.int_n_head(logit)
        
        loss_sq = F.cross_entropy(sq_logit.reshape(-1, 2, 32), sq)
        self.log('Sq/Train', loss_sq, prog_bar=True, on_epoch=True)
        self.log('Sq/Train_F1', torchmetrics.functional.f1_score(sq_logit.reshape(-1, 2, 32), sq, task="multiclass", num_classes=2, average="macro"), prog_bar=True, on_epoch=True)
        loss_log2 = F.cross_entropy(log2_logit.reshape(-1, 2, 32), log2)
        self.log('Log2/Train', loss_log2, prog_bar=True, on_epoch=True)
        self.log('Log2/Train_F1', torchmetrics.functional.f1_score(log2_logit.reshape(-1, 2, 32), log2, task="multiclass", num_classes=2, average="macro"), prog_bar=True, on_epoch=True)
        loss_log10 = F.cross_entropy(log10_logit.reshape(-1, 2, 32), log10)
        self.log('Log10/Train', loss_log10, prog_bar=True, on_epoch=True)
        self.log('Log10/Train_F1', torchmetrics.functional.f1_score(log10_logit.reshape(-1, 2, 32), log10, task="multiclass", num_classes=2, average="macro"), prog_bar=True, on_epoch=True)
        loss_int_n = F.cross_entropy(int_n_logit.reshape(-1, 2, 64), int_n)
        self.log('Int_n/Train', loss_int_n, prog_bar=True, on_epoch=True)
        self.log('Int_n/Train_F1', torchmetrics.functional.f1_score(int_n_logit.reshape(-1, 2, 64), int_n, task="multiclass", num_classes=2, average="macro"), prog_bar=True, on_epoch=True)

        self.log('Loss/Train', loss_1, prog_bar=True, on_epoch=True)
        self.log('InfoLoss/Train', info_loss, prog_bar=True, on_epoch=True)
        hessian_calc = batch_idx % 5 == 0
        self.manual_backward(loss_1 + info_loss + loss_sq + loss_log2 + loss_log10 + loss_int_n, create_graph=hessian_calc)
        if batch_idx % 8 != 765:
#             optimizer.step(SAM_step=True)
#             optimizer.zero_grad()
#             logit, mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out = self(float_bits)
#             loss_1 = self.calc_loss(mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out, mantissa, exponent, x, y, float_bits)
#             info_logit = self.tanh(self.sin_head(logit))
#             info_loss = F.mse_loss(info_logit, add_info) + F.l1_loss(info_logit, add_info)

#             self.log('Loss/Train_SAM', loss_1, prog_bar=True, on_epoch=True)
#             self.log('InfoLoss/Train_SAM', info_loss, prog_bar=True, on_epoch=True)
#             self.manual_backward(loss_1 + info_loss, create_graph=True)
            optimizer.step(SAM_step=False, hessian_calc=hessian_calc)
            optimizer.zero_grad()
        else:
            optimizer.step(SAM_step=False, update=False)
#             weight = optimizer.get_weight()
            optimizer.zero_grad()
            for i in range(len(optimizer.lr) - 1):
                optimizer.meta_step(i)
                logit, mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out = self(float_bits)
                loss_meta = self.calc_loss(mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out, mantissa, exponent, x, y, float_bits)
#                 print(loss_meta)
                self.manual_backward(loss_meta)
            optimizer.return_state_step(len(optimizer.lr) - 1)
            
#             logit, mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out = self(float_bits)
#             loss_meta = self.calc_loss(mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out, mantissa, exponent, x, y, float_bits)
            
#             print("new loss", loss_meta)
            self.log('Loss/Train_Correct', loss_meta, prog_bar=True, on_epoch=True)
#             print("new lr",optimizer.get_lr(optimizer.lr[0].detach()))
            for name, value in optimizer.lr.items():
                self.log('Lr_' + str(name), optimizer.get_lr(value.detach()), prog_bar=True, on_epoch=True)
#             optimizer.install_weight(weight)
            
#             logit, mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out = self(float_bits)
#             loss_meta = self.calc_loss(mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out, mantissa, exponent, x, y, float_bits)
#             print("old loss", loss_meta)
            
#             optimizer.update_step()
            
#             logit, mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out = self(float_bits)
#             loss_meta = self.calc_loss(mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out, mantissa, exponent, x, y, float_bits)
#             print("true loss", loss_meta)
#             print("")
        return loss_1
    def validation_step(self, batch, batch_idx):
#         random_bits = batch[0]
#         random_floats = bits_to_float(random_bits, 23, 8)

#         # Get mantissa and exponent
#         float_bits, mantissa, exponent = float_to_bits(random_floats, 23, 8, retun_ME=True)

#         # Calculate x and y 
#         x, y = self.calc_xy(random_floats)
        
#         mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out = self(float_bits)
#         loss = self.calc_loss(mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out, mantissa, exponent, x, y, float_bits)
        
#         self.log('Loss/Validation', loss, prog_bar=True, on_epoch=True)
        return None
    def calc_xy(self, random_floats, max_y=127):
        float_abs = torch.abs(random_floats)
        y = torch.ceil(torch.log10(float_abs))
        x = (random_floats / 10**y)
        x[float_abs < 1] = random_floats[float_abs < 1]
        y = torch.clamp(y, 0, max_y)
        return x, y.long()
    def calc_loss(self, mantisa_out, exponent_out, tanh_out, log_10_out, decoder_out, mantisa_target, exponent_target, tanh_target, log_10_target, decoder_target):
        loss_reg = F.mse_loss(mantisa_out[:,0], mantisa_target) + F.l1_loss(mantisa_out[:,0], mantisa_target) + F.mse_loss(tanh_out[:,0], tanh_target) + F.l1_loss(tanh_out[:,0], tanh_target)
        loss_cls = F.cross_entropy(exponent_out, exponent_target) + F.cross_entropy(log_10_out, log_10_target) + F.cross_entropy(decoder_out.reshape(-1, 2, 32), decoder_target)
        return loss_reg + loss_cls
    def configure_optimizers(self):
#         monolithic_param = nn.Parameter(torch.cat([p.data.view(-1) for p in self.parameters()]))
#         print(monolithic_param.shape)
#         offset = 0
#         for name, param in self.named_parameters():
#             if '.' in name:
#                 module_name, param_name = name.rsplit('.', 1)
#                 module = reduce(getattr, module_name.split('.'), model)
#             else:
#                 module = self
#                 param_name = name
#             size = param.size()
#             setattr(module, param_name, MonolithicParameter(monolithic_param, offset, *size))
#             offset += np.prod(size)
#         for name, param in self.named_parameters():
#             size = param.size()
#             setattr(self, name, MonolithicParameter(monolithic_param, offset, *size))
#             offset += np.prod(size)
        optimizer = ModernOptimizer(list(self.parameters()),
                         lr=self.hparams.learning_rate, SAM=False, eps=0.00000000001
                         )
        return optimizer
        
class AutoEncoderModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.00002,
        batch_size: int = 2,
        seed_value: int = 1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        if seed_value != None:
          pl.seed_everything(seed_value)
          torch.manual_seed(seed_value)
          torch.cuda.manual_seed(seed_value)
          random.seed(seed_value)
        self.encoder = Encoder(seq_len=32,
                 num_layers=4,
                 layer_norm=True,
                 unet=True,
                 grad_checkpointing=False,
                 drop_path=0.0,
                 dropout=0.0,
                 shuffle=False,
                 d_models=[512, 256, 128, 64, 32]
                )
        self.decoder = Decoder(seq_len=32,
                 num_layers=4,
                 layer_norm=True,
                 unet=True,
                 grad_checkpointing=False,
                 drop_path=0.0,
                 dropout=0.0,
                 shuffle=False,
                 d_models=[512, 256, 128, 64, 32]
                )
        self.cls_head = nn.Linear(512, 82)
        self.decoder_head = nn.Linear(512, 256)
        self.embedding = Embedding(256, 512)
        self.pos_embedding = nn.Parameter(torch.randn(32, 1, 512))
        self.norm = Norm()
    def forward(self, data):
        data = self.embedding(data) + self.pos_embedding
        data = self.norm(self.encoder(data))
        data = self.decoder(data)
        return self.decoder_head(data[:-1]), self.cls_head(data[-1])
    def decoder_forward(self, data):
        data = self.decoder(self.norm(data))
        return self.decoder_head(data[:-1]), self.cls_head(data[-1])
    def on_validation_epoch_end(self):
        gc.collect()
        self.trainer.save_checkpoint(filepath="/kaggle/working/checkpoint.ckpt")
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        material, target = batch
        if random.uniform(0, 1) > 0.7:
            for idx, m in enumerate(model.named_modules()): 
                path = m[0]
                component = m[1]
                if isinstance(component, nn.Dropout):
                    component.p = 0.0
            logit, cls_token = self(material)
            loss_1 = F.cross_entropy(logit.permute(1,2,0), material[:,:-1], ignore_index=2, label_smoothing=0.1)
            accuracy_1 = torchmetrics.functional.accuracy(logit.detach().permute(1,2,0), material[:,:-1], task="multiclass", num_classes=256, ignore_index=2)

            cls_loss_1 = F.huber_loss(cls_token, target)

            self.log('Accuracy/Train', accuracy_1, prog_bar=True, on_epoch=True)
            self.log('Loss/Train', loss_1, prog_bar=True, on_epoch=True)
            self.log('Cls_loss/Train', cls_loss_1, prog_bar=True, on_epoch=True)
            self.manual_backward(loss_1 + cls_loss_1)
            optimizer.first_step(zero_grad=True)
            logit, cls_token = self(material)
            loss_2 = F.cross_entropy(logit.permute(1,2,0), material[:,:-1], ignore_index=2, label_smoothing=0.1)
            accuracy_2 = torchmetrics.functional.accuracy(logit.detach().permute(1,2,0), material[:,:-1], task="multiclass", num_classes=256, ignore_index=2)

            cls_loss_2 = F.huber_loss(cls_token, target)
            self.log('Accuracy/Train_SAM', accuracy_2, prog_bar=True, on_epoch=True)
            self.log('Loss/Train_SAM', loss_2, prog_bar=True, on_epoch=True)
            self.log('Cls_loss/Train_SAM', cls_loss_2, prog_bar=True, on_epoch=True)
            self.manual_backward(loss_2 + cls_loss_2)
            optimizer.second_step(zero_grad=True)
        else:
            for idx, m in enumerate(model.named_modules()): 
                path = m[0]
                component = m[1]
                if isinstance(component, nn.Dropout):
                    component.p = 0.3
            logit, cls_token = self(material)
            loss_1 = F.cross_entropy(logit.permute(1,2,0), material[:,:-1], ignore_index=2, label_smoothing=0.1)
            accuracy_1 = torchmetrics.functional.accuracy(logit.detach().permute(1,2,0), material[:,:-1], task="multiclass", num_classes=256, ignore_index=2)

            cls_loss_1 = F.huber_loss(cls_token, target)

            self.log('Accuracy/Train', accuracy_1, prog_bar=True, on_epoch=True)
            self.log('Loss/Train', loss_1, prog_bar=True, on_epoch=True)
            self.log('Cls_loss/Train', cls_loss_1, prog_bar=True, on_epoch=True)
            self.manual_backward(loss_1 + cls_loss_1)
            optimizer.base_optimizer.step()
            optimizer.zero_grad()
        return loss_1 + cls_loss_1
    def validation_step(self, batch, batch_idx):
        material, target = batch
        logit, cls_token = self(material)
        loss = F.cross_entropy(logit.permute(1,2,0), material[:,:-1], ignore_index=2, label_smoothing=0.1)
        accuracy = torchmetrics.functional.accuracy(logit.detach().permute(1,2,0), material[:,:-1], task="multiclass", num_classes=256, ignore_index=2)
        
        cls_loss = F.huber_loss(cls_token, target)
        
        self.log('Accuracy/Validation', accuracy, prog_bar=True, on_epoch=True)
        self.log('Loss/Validation', loss, prog_bar=True, on_epoch=True)
        self.log('Cls_loss/Validation', cls_loss, prog_bar=True, on_epoch=True)
        return loss + cls_loss
    def configure_optimizers(self):
        optimizer = SAM(list(self.parameters()), ModernOptimizer,
                         lr=self.hparams.learning_rate, rho=0.2, adaptive=True
                         )
        return optimizer
