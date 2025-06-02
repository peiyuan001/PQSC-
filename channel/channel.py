import torch
import torch.nn as nn
import numpy as np
import sionna
import os

from . import utils_torch as ut
# Load the required Sionna components
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.polar import PolarEncoder, Polar5GEncoder, PolarSCLDecoder, Polar5GDecoder, PolarSCDecoder
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
import tensorflow as tf
from . import lwe
import time
from .sionna_model import System_Model as AWGN_Model
from channel.Rayleigh_model import  System_Model as Rayleigh_Model

def channel(channel_type='exp', snr=20, ebno=20, device=torch.device('cuda'), channel_selection='AWGN', lwe_mode='Authorized'):

    gpus = tf.config.list_physical_devices('GPU')
    # gpu memory allocation
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    def virtual_channel(z_hat: torch.Tensor):
        if z_hat.dim() == 4:
            k = torch.prod(torch.tensor(z_hat.size()[1:]))
            sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True)/k
        elif z_hat.dim() == 3:
            k = torch.prod(torch.tensor(z_hat.size()))
            sig_pwr = torch.sum(torch.abs(z_hat).square())/k
        noi_pwr = sig_pwr / (10 ** (snr / 10))
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr)
        return z_hat + noise

    def exp_channel_int(z_in: torch.Tensor):
        # print('tensorflow device:', str(device.index))
        # tf gpu configuration
        # Determine TensorFlow device string based on PyTorch device
        if device.type == 'cuda' and torch.cuda.is_available():
            gpu_index = device.index  # Extract the GPU index from the PyTorch device
            tf_device = f'/GPU:{gpu_index}'  # Construct the TensorFlow device string
        else:
            tf_device = '/CPU:0'  # Default to CPU if CUDA is not available
        with tf.device(tf_device):
            k = 336  # number of information bits per codeword
            n = 672  # desired codeword length
            num_bits_per_symbol = 12  # 2^(num_bits_per_symbol) QAM
            quanti_level = 4093  # pow(2, num_bits_per_symbol)

            # -------------------------sender-------------------------
            z = z_in
            quantized = levels = z
            binary_level = ut.binarize(levels, num_bits_per_symbol)
            binary_level = ut.flatten_dim_1(binary_level)

            """
            lwe encoding:
            for binarized latent z, each bit will be encrypted to lattic of 4093 (12 bits), resulting a cyphertext of c=[c1, c2] shape=(1,n_2+l),
            (l = number of bits in z)
            each element in c need also 12 bits for storage, resulting to cyphertext_tf_binary shape=(1, 12*(n_2+l))
            """

            binary_level = binary_level.to(torch.float64)

            l = binary_level.numel()
            flattened_z = binary_level.view(1, l).to(z.device)

            lwe_module = lwe.LWE_bk(q=quanti_level, device=z.device, mode=lwe_mode)
            lwe_module.set_device(z.device)

            cyphertext = lwe_module.lwe_bk_encoder(flattened_z)

            lwe_decoded = lwe_module.lwe_bk_decoder(cyphertext)


            cyphertext_binary = [ut.binarize(cyphertext[0].to(torch.int32), num_bits_per_symbol),
                                 ut.binarize(cyphertext[1].to(torch.int32), num_bits_per_symbol)]


            cypher_bsize = [torch.numel(cyphertext_binary[0]), torch.numel(cyphertext_binary[1])]

            cypher_bshape = [cyphertext_binary[0].shape, cyphertext_binary[1].shape]

            reshaped_cypher = [ut.reshape_to_btc_k(cyphertext_binary[0], k),
                               ut.reshape_to_btc_k(cyphertext_binary[1], k)]

            # ----------------channel----------------------------------------------
            reshaped_cypher_tf = [tf.convert_to_tensor(reshaped_cypher[0].cpu().numpy()),
                                  tf.convert_to_tensor(reshaped_cypher[1].cpu().numpy())]
            concated_cypher_tf = tf.concat([reshaped_cypher_tf[0], reshaped_cypher_tf[1]], axis=0)
            shape_1, shape_2 = tf.shape(reshaped_cypher_tf[0])[0], tf.shape(reshaped_cypher_tf[1])[0]

            enc = LDPC5GEncoder(k=k, n=n)
            dec = LDPC5GDecoder(enc, num_iter=20)

            if channel_selection == 'AWGN':
                model = AWGN_Model(k=k,
                                     n=n,
                                     enbo=ebno,
                                     num_bits_per_symbol=num_bits_per_symbol,
                                     encoder=enc,
                                     decoder=dec,
                                     channel_selection=channel_selection)

            elif channel_selection == 'Rayleigh':
                model = Rayleigh_Model(k=k,
                                       n=n,
                                       enbo=ebno,
                                       num_bits_per_symbol=num_bits_per_symbol,
                                       encoder=enc,
                                       decoder=dec,
                                       )
            else:
                raise ValueError(f"Unsupported channel selection: {channel_selection}. Please choose 'AWGN' or 'Rayleigh'.")

            _, cypher_ch = model(concated_cypher_tf, ebno)
            cypher_hat_ch = tf.split(cypher_ch, [shape_1, shape_2], axis=0)


            """
            receiver
            """
            cypher_hat_ch = [torch.tensor(cypher_hat_ch[0].numpy()).to(device), torch.tensor(cypher_hat_ch[1].numpy()).to(device)]
            # recover the shape before LDPC coding
            demodul_tensor = [ut.recover_original_data(cypher_hat_ch[0], cypher_bshape[0], cypher_bsize[0]),
                              ut.recover_original_data(cypher_hat_ch[1], cypher_bshape[1], cypher_bsize[1])]

            """
            lwe decoding
            """
            # the cyphertext in [0,4093]
            cyphertext_hat = [None] * 2
            cyphertext_hat[0], _ = ut.binary_to_float(demodul_tensor[0], num_bits_per_symbol,
                                                      min=0, max=quanti_level - 1)
            cyphertext_hat[1], _ = ut.binary_to_float(demodul_tensor[1], num_bits_per_symbol,
                                                      min=0, max=quanti_level - 1)
            cyphertext_hat = [cyphertext_hat[0].to(torch.float64).squeeze(dim=-1),
                              cyphertext_hat[1].to(torch.float64).squeeze(dim=-1)]


            # decide 0 or 1 with decoder for each bit
            dec_start = time.time()
            lwe_decoded = lwe_module.lwe_bk_decoder(cyphertext_hat)

            # recover the float z
            data_rec = ut.binary_to_integer(lwe_decoded, num_bits_per_symbol)
            data_rec = data_rec.view(z_in.shape)

            data_rec_min = 0
            data_rec_max = 511
            data_rec = torch.clamp(data_rec, min=data_rec_min, max=data_rec_max)

        return data_rec.to(device)

    if channel_type == 'virtual':
        return virtual_channel
    elif channel_type == 'exp':
        return exp_channel_int
    else:
        raise Exception('Unknown type of channel')



