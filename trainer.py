import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from vqvae import VQVAE
from helper import get_device
from pytorch_msssim import SSIM

def get_shape(arr):
    if not isinstance(arr, list):
        # If it's not a list, it has no shape.
        return None
    shape = []
    while isinstance(arr, list):
        shape.append(len(arr))
        if len(arr) > 0:
            arr = arr[0]
        else:
            break
    return shape

def To_0_1(x):
   return (x+1)/2

def To_1_1(x):
    return x*2-1

def tensor2uint(img):
    # Check if the tensor is on GPU and move it to CPU if necessary
    if img.is_cuda:
        img = img.to('cpu')  # Move to CPU for safe operations
    # Convert tensor to float, clamp to [0, 1], and convert to NumPy
    img = img.float().clamp(0, 1).numpy()
    # Return the final uint8 image
    return np.uint8((img * 255.0).round())


def uint2tensor(img):
    img = np.clip(img, 0, 255)
    output = img.astype(np.float32) / 255.0
    output = torch.from_numpy(output)
    return output

def calculate_psnr(img1, img2):

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim_batch(img1, img2):

    batch_tensor1 = torch.from_numpy(img1).float()
    batch_tensor2 = torch.from_numpy(img2).float()

    assert batch_tensor1.shape[0] == batch_tensor2.shape[0], "SSIM input tensor batch size mismatch"
    assert batch_tensor1.shape[1] == batch_tensor2.shape[1], "SSIM input tensor channel mismatch"
    assert batch_tensor1.shape[2] == batch_tensor2.shape[2], "SSIM input tensor Height mismatch"
    assert batch_tensor1.shape[3] == batch_tensor2.shape[3], "SSIM input tensor Width mismatch"

    bs, num_channel, H, W = batch_tensor1.shape[0], batch_tensor1.shape[1], batch_tensor1.shape[2], batch_tensor1.shape[3]
    ssim_scores = []
    ssim_sum = 0
    ssim_calculator = SSIM(data_range=255, size_average=False, channel=num_channel)
    for i in range(bs):
            ssim_score = ssim_calculator(batch_tensor1[i:i+1], batch_tensor2[i:i+1]).item()
            ssim_scores.append(ssim_score)
            ssim_sum += ssim_score

    return ssim_scores, ssim_sum/bs

class VQVAETrainer:
    def __init__(self, cfg, args):

        self.device = get_device(args.device)
        self.cfg = cfg
        self.args = args


        self.net = VQVAE(in_channels=cfg.in_channels,
                    hidden_channels=cfg.hidden_channels,
                    embed_dim=cfg.embed_dim,
                    nb_entries=cfg.nb_entries,
                    nb_levels=cfg.nb_levels,
                    scaling_rates=cfg.scaling_rates,
                    args=args,
                    cfg=cfg)
        self.net = self.net.to(self.device)
        # if torch.cuda.device_count() > 1:
        #     self.net = torch.nn.DataParallel(self.net)
        if args.transfer_learning:
            print("Encoder&Codebook Frozen")
            for name, param in self.net.named_parameters():
                if 'encoder' in name or 'codebooks' in name : #
                    param.requires_grad = False

        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.opt.zero_grad()

        self.beta = cfg.beta
        self.scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

        self.update_frequency = math.ceil(cfg.mini_batch_size / cfg.batch_size) #math.ceil(cfg.batch_size / cfg.mini_batch_size)
        self.train_steps = 0


    def _calculate_loss(self, x: torch.FloatTensor):

        x = x.to(self.device)
        y, q_loss, encoder_outputs, decoder_outputs, latent, code_outputs = self.net(x)
        x_p = tensor2uint((x.detach()+1)/2)
        y_p = tensor2uint((y.detach()+1)/2)

        psnr = calculate_psnr(x_p, y_p)
        ssim, ssim_avg = calculate_ssim_batch(x_p, y_p)
        loss_mse, loss_quanti = y.sub(x).pow(2).mean(), sum(q_loss)  #
        loss = loss_mse + self.beta*loss_quanti

        return loss, loss_mse, loss_quanti, y, latent, psnr, ssim_avg

    def train(self, x: torch.FloatTensor):

        self.net.train()

        with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):

            loss, loss_mse, loss_quanti, y, latent, psnr, ssim_avg = self._calculate_loss(x)

        self.scaler.scale(loss / self.update_frequency).backward()

        self.train_steps += 1
        if self.train_steps % self.update_frequency == 0:
            self._update_parameters()

        return loss.item(), loss_mse.item(), loss_quanti.item(), latent, psnr, ssim_avg

    """
        Use accumulated gradients to step `self.opt`, updating parameters.
    """
    def _update_parameters(self):
        self.scaler.step(self.opt)
        self.opt.zero_grad()
        self.scaler.update()

    @torch.no_grad()
    def eval(self, x: torch.FloatTensor):
        self.net.eval()
        self.opt.zero_grad()
        loss, loss_mse, loss_quanti, y, latent, psnr, ssim_avg = self._calculate_loss(x)
        return loss.item(), loss_mse.item(), loss_quanti.item(), y, latent, psnr, ssim_avg

    def save_checkpoint(self, path):
        torch.save(self.net.state_dict(), path)

    def load_checkpoint(self, path):
        self.net.load_state_dict(torch.load(path))
