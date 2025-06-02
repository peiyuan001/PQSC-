import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2
from typing import Tuple
from helper import HelperModule
import channel.channel as CH
from typing import List

# Function to perform the TensorFlow operation in the forward pass
def sionna_model(input_tensor, snr, args):

    device = input_tensor.device
    channel_model = CH.channel(channel_type=args.channel_type, snr=snr, ebno=snr, device=device, channel_selection=args.channel_selection, lwe_mode=args.lwe_mode)
    output_tensor = channel_model(input_tensor)

    return output_tensor

# Custom PyTorch autograd function for STE
class STEWithTensorFlow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, snr, args):
        # Apply the TensorFlow operation
        z_tf = sionna_model(z, snr, args)

        # Save the original mu for backward pass
        ctx.save_for_backward(z)

        # Return the result of the TensorFlow operation
        return z_tf

    @staticmethod
    def backward(ctx, grad_output):
        # torch.cuda.synchronize() # Ensure all prior CUDA operations are complete
        # In the backward pass, just pass the gradient straight through
        z, = ctx.saved_tensors
        grad_input = grad_output.clone()  # Straight through estimator: propagate the gradients as is
        return grad_input

class ReZero(HelperModule):
    def build(self, in_channels: int, res_channels: int):
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x) * self.alpha + x

class ResidualStack(HelperModule):
    def build(self, in_channels: int, res_channels: int, nb_layers: int):
        self.stack = nn.Sequential(*[ReZero(in_channels, res_channels)
                        for _ in range(nb_layers)
                    ])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.stack(x)

class Encoder(HelperModule):
    def build(self,
            in_channels: int, hidden_channels: int,
            res_channels: int, nb_res_layers: int,
            downscale_factor: int,
        ):
        assert log2(downscale_factor) % 1 == 0, "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))
        layers = []
        c_channel, n_channel = in_channels, hidden_channels // 2
        for _ in range(downscale_steps):
            layers.append(nn.Sequential(
                nn.Conv2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, hidden_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        layers.append(ResidualStack(n_channel, res_channels, nb_res_layers))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

class Decoder(HelperModule):
    def build(self,
            in_channels: int, hidden_channels: int, out_channels: int,
            res_channels: int, nb_res_layers: int,
            upscale_factor: int,
        ):
        assert log2(upscale_factor) % 1 == 0, "Downscale must be a power of 2"
        upscale_steps = int(log2(upscale_factor))
        layers = [nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1)]
        layers.append(ResidualStack(hidden_channels, res_channels, nb_res_layers))
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        for _ in range(upscale_steps):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, out_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        # layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)



#
class CodeLayer(HelperModule):
    def build(self, in_channels: int, embed_dim: int, nb_entries: int, args, cfg):

        self.conv_in = nn.Conv2d(in_channels, embed_dim, 1)
        self.dim = embed_dim
        self.n_embed = nb_entries
        self.decay = 0.99
        self.eps = 1e-5

        torch.manual_seed(1)
        embed = torch.randn(embed_dim, nb_entries, dtype=torch.float32)

        # Register the buffer
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(nb_entries, dtype=torch.float32))
        self.register_buffer("embed_avg", embed.clone())
        self.args = args
        self.cfg = cfg
        

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, float, torch.LongTensor]:

        self.embed.data = torch.clamp(self.embed.data, min=-1.0, max=1.0)

        x = self.conv_in(x.float()).permute(0,2,3,1)
        flatten = x.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])

        if self.args.sionna_channel:
            embed_ind_prime = STEWithTensorFlow.apply(embed_ind, self.cfg.noise_db, self.args)
            quantize = self.embed_code(embed_ind_prime)
        else:
            quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        return quantize.permute(0, 3, 1, 2), diff, embed_ind

    def embed_code(self, embed_id: torch.LongTensor) -> torch.FloatTensor:
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class Upscaler(HelperModule):
    def build(self,
            embed_dim: int,
            scaling_rates: List[int],
        ):

        self.stages = nn.ModuleList()
        for sr in scaling_rates:
            upscale_steps = int(log2(sr))
            layers = []
            for _ in range(upscale_steps):
                layers.append(nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(embed_dim))
                layers.append(nn.ReLU(inplace=True))
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x: torch.FloatTensor, stage: int) -> torch.FloatTensor:
        return self.stages[stage](x)

"""
    Main VQ-VAE Module, capable of support arbitrary number of levels
"""
class VQVAE(HelperModule):
    def build(self,
            in_channels: int                = 3,
            hidden_channels: int            = 128,
            res_channels: int               = 32,
            nb_res_layers: int              = 2,
            nb_levels: int                  = 3,
            embed_dim: int                  = 64,
            nb_entries: int                 = 512,
            scaling_rates: List[int]        = [8, 4, 2],
            args                            = None,
            cfg                             = None
        ):

        self.args = args
        self.cfg = cfg
        self.nb_levels = nb_levels
        assert len(scaling_rates) == nb_levels, "Number of scaling rates not equal to number of levels!"

        self.encoders = nn.ModuleList([Encoder(in_channels, hidden_channels, res_channels, nb_res_layers, scaling_rates[0])])

        self.codebooks = nn.ModuleList()

        self.codebooks.append(CodeLayer(hidden_channels, embed_dim, nb_entries, self.args, self.cfg))

        self.decoders = nn.ModuleList([Decoder(embed_dim*nb_levels, hidden_channels, in_channels, res_channels, nb_res_layers, scaling_rates[0])])

        self.upscalers = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.upscalers.append(Upscaler(embed_dim, scaling_rates[1:len(scaling_rates) - i][::-1]))


    def forward(self, x):
        encoder_outputs = []  # final encoder output
        code_outputs = []
        decoder_outputs = []
        upscale_counts = []
        id_outputs = []
        diffs = []

        for enc in self.encoders:
            if len(encoder_outputs):
                encoder_outputs.append(enc(encoder_outputs[-1]))
            else:
                encoder_outputs.append(enc(x))

        for l in range(self.nb_levels - 1, -1, -1):

            codebook, decoder = self.codebooks[l], self.decoders[l]

            if len(decoder_outputs):  # if we have previous levels to condition on
                code_q, code_d, emb_id = codebook(torch.cat([encoder_outputs[l], decoder_outputs[-1]], axis=1))

            else:
                # if True:
                code_q, code_d, emb_id = codebook(encoder_outputs[l])

            """
                code_q: index list in embedding space
                code_d: mapping error, difference between real data and embedding space
                emb_id: embedding layer id 
            """

            diffs.append(code_d)
            id_outputs.append(emb_id)
            code_outputs = [self.upscalers[i](c, upscale_counts[i]) for i, c in enumerate(code_outputs)]
            upscale_counts = [u + 1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1)))
            code_outputs.append(code_q)
            upscale_counts.append(0)

        return decoder_outputs[-1], diffs, encoder_outputs, decoder_outputs, id_outputs, code_outputs

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.eval()

    def unfreeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.train()


