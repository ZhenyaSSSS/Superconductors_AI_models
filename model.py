import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
import torchmetrics
import torch.nn as nn
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
      
class FeedForwardModule(nn.Module):
    """
    FeedForwardModule applies a feed-forward neural network to the input data.
    """
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.0,
                 layer_norm: bool = True,
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
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = Norm() if layer_norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the FeedForwardModule.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the feed-forward neural network.
        """
        x = self.layer_norm(x)
        x = self.act(self.dropout(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x
      
class Pos_bias(nn.Module):
    def __init__(self,
                 d_model: int,
                 **kwargs
                 ):
        super().__init__()
        self.pos_bias = nn.Parameter(torch.zeros(1, 1, d_model, d_model), requires_grad=True)
    def forward(self, shape):
        return torch.nn.functional.interpolate(self.pos_bias, size=[shape, shape], mode='bilinear')[0,0]
      
class AFTModule(nn.Module):
    def __init__(self,
                 d_model: int,
                 seq_len: int,
                 dropout: float = 0.0,
                 layer_norm: bool = False,
                 **kwargs
                 ):
        """
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        * `seq_len` is $T$
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.pos_bias = Pos_bias(seq_len)
        self.activation = nn.Sigmoid()
        self.output = nn.Linear(d_model, d_model)#nn.Linear(d_model, d_model)
        if layer_norm:
          self.layer_norm = Norm()
        else:
          self.layer_norm = nn.Identity()
    def forward(self, data: torch.Tensor):
        """
        `query`, `key` and `value` are the tensors that store
        collection of token embeddings for  *query*, *key* and *value*.
        They have shape `[seq_len, batch_size, d_model]`.
        `mask` has shape `[seq_len, seq_len, batch_size]` and
        `mask[i, j, b]` indicates whether for batch `b`,
        query at position `i` has access to key-value at position `j`.
        """
        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        # [seq_len, channels, batch_size, d_model]
        if len(data.shape) == 4:
          reshape_bool = True
          data = data.permute(0,3,2,1)
          shape = data.shape
          data = data.reshape(shape[0], shape[1], -1).permute(0,2,1)
        else:
          reshape_bool = False
        query,key,value = self.qkv(self.layer_norm(self.dropout(data))).chunk(3, dim = -1)
        pos_bias = self.dropout(self.pos_bias(len(data)))#self.pos_bias[:len(data), :len(data)]  #16896, 16896
        pos_bias = pos_bias.unsqueeze(-1)
        # pos_bias.masked_fill_(~mask, float('-inf'))
        max_key = key.max(dim=0, keepdims=True)[0]
        max_pos_bias = pos_bias.max(dim=1,  keepdims=True)[0]

        exp_key = torch.exp(key - max_key)
        exp_pos_bias = torch.exp(pos_bias - max_pos_bias)
        num = torch.einsum('ijb,jbd->ibd', exp_pos_bias, exp_key * value)
        den = torch.einsum('ijb,jbd->ibd', exp_pos_bias, exp_key)
        y = self.output(self.activation(query) * num / den)
        if reshape_bool:
          y = y.permute(0,2,1)
          y = y.reshape(shape[0], shape[1], shape[2], shape[3]).permute(0,3,2,1)
        return y
      
class ResAttention(nn.Module):
    def __init__(self, 
                 dropout,
                 **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, data, residual):
        for i in data:
          residual = residual + self.dropout(i)
        return residual
    
class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
    def forward(self, x):
        return torch.nn.functional.layer_norm(x, [x.shape[-1]], eps=1e-20)
      
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
        self.res_attention = ResAttention(dropout)
        self.grad_checkpointing = grad_checkpointing
        self.drop_path = DropPath(drop_prob=drop_path)
    def forward(self, data, **kwargs):
        residual = data.clone()
        module_outputs = []
        for module, module_config in zip(self.modules_list, self.module_configuration):
            if "pre_permute" in module_config["params"]:
                input_data = input_data.permute(*module_config["params"]["pre_permute"])

            if self.grad_checkpointing:
              output_data = torch.utils.checkpoint.checkpoint(module, data, **kwargs)
            else:
              output_data = module(data, **kwargs)

            if "post_permute" in module_config["params"]:
                output_data = output_data.permute(*module_config["params"]["post_permute"])

            module_outputs.append(self.drop_path(output_data))

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
                  module = module_class(**module_params)
                self.modules_list.append(module)
              
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
        self.shuffle = shuffle
        self.layers.extend([Layer(module_configuration, layer_norm, grad_checkpointing, drop_path, dropout) for _ in range(num_layers)])
        if unet:
          self.res = ResAttention(dropout=dropout)
        else:
          self.res = None
    def forward(self, x, **kwargs):
        if self.shuffle:
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
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.emb = nn.Embedding(n_token, d_model)
    def forward(self, inp):
        embed = self.emb(inp)
        return self.layer_norm(embed.mul_(self.embed_scale)).permute(1,0,2)
      
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
