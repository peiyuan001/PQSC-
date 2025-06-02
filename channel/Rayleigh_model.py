import torch
import torch.nn as nn
import numpy as np
import sionna
import os

# import utils_torch as ut
# Load the required Sionna components
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.polar import PolarEncoder, Polar5GEncoder, PolarSCLDecoder, Polar5GDecoder, PolarSCDecoder
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.fec.polar.utils import generate_5g_ranking, generate_rm_code
from sionna.fec.conv import ConvEncoder, ViterbiDecoder, BCJRDecoder
from sionna.fec.turbo import TurboEncoder, TurboDecoder
from sionna.fec.linear import OSDecoder
from sionna.utils import BinarySource, ebnodb2no
from sionna.utils.metrics import  count_block_errors
from sionna.channel import AWGN
from sionna.channel import OFDMChannel
from sionna.channel import RayleighBlockFading

import tensorflow as tf
from sionna.ofdm import ResourceGrid


def recover_input(y, channel_matrix, fft_size):
    """
    Recover input data from the channel output.

    Parameters:
    - y: Tensor output from the channel (e.g., shape (8, 1, 1, 2, 336)).
    - channel_matrix: Known or estimated channel matrix (shape matches y).
    - fft_size: Number of subcarriers.

    Returns:
    - Recovered input data.
    """
    # Step 1: Squeeze dimensions
    y_reduced = tf.squeeze(y, axis=[1, 2])  # Remove Rx/Tx dimensions

    # Step 2: Perform IFFT
    y_time = tf.signal.ifft(y_reduced)

    # Step 3: Equalize channel
    H_conj = tf.math.conj(channel_matrix)
    y_equalized = y_time / (tf.abs(channel_matrix) ** 2 + 1e-8)

    # Step 4: Reshape to match input
    recovered_input = tf.reshape(y_equalized, [y.shape[0], fft_size])

    return recovered_input



class System_Model(tf.keras.Model):
    """System model for channel coding BER simulations.

    This model allows to simulate BERs over an AWGN channel with
    QAM modulation. Arbitrary FEC encoder/decoder layers can be used to
    initialize the model.

    Parameters
    ----------
        k: int
            number of information bits per codeword.

        n: int
            codeword length.

        num_bits_per_symbol: int
            number of bits per QAM symbol.

        encoder: Keras layer
            A Keras layer that encodes information bit tensors.

        decoder: Keras layer
            A Keras layer that decodes llr tensors.

        demapping_method: str
            A string denoting the demapping method. Can be either "app" or "maxlog".

        sim_esno: bool
            A boolean defaults to False. If true, no rate-adjustment is done for the SNR calculation.

         cw_estiamtes: bool
            A boolean defaults to False. If true, codewords instead of information estimates are returned.
    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.

        ebno_db: float or tf.float
            A float defining the simulation SNR.

    Output
    ------
        (u, u_hat):
            Tuple:

        u: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the transmitted information bits.

        u_hat: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the estimated information bits.
    """
    def __init__(self,
                 k,
                 n,
                 num_bits_per_symbol,
                 enbo,
                 encoder,
                 decoder,
                 demapping_method="app",
                 sim_esno=False,
                 cw_estimates=False,
                 channel_selection = 'Rayleigh'
                 ):

        super().__init__()

        self.channel_selection = channel_selection
        # store values internally
        self.k = k
        self.n = n
        self.sim_esno = sim_esno # disable rate-adjustment for SNR calc
        self.cw_estimates = cw_estimates # if true codewords instead of info bits are returned

        # number of bit per QAM symbol
        self.num_bits_per_symbol = num_bits_per_symbol

        # # init components
        # self.source = BinarySource()

        # initialize mapper and demapper for constellation object
        self.constellation = Constellation("qam",
                                num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper(demapping_method,
                                 constellation=self.constellation)

        # the channel can be replaced by more sophisticated models
        if self.channel_selection == 'AWGN':
            self.channel = AWGN()
        elif self.channel_selection == 'Rayleigh':
            Rayleigh = RayleighBlockFading(num_rx = 1,
                               num_rx_ant = 1,
                               num_tx = 1,
                               num_tx_ant = 1)

            rg = ResourceGrid(num_ofdm_symbols=2,
                              fft_size=56,
                              subcarrier_spacing=30e3,
                              num_tx=1,
                              num_streams_per_tx=1,
                              num_guard_carriers=[0,1],
                              dc_null=True,
                              pilot_pattern="kronecker",
                              pilot_ofdm_symbol_indices=[0,1]
                             )

            self.channel = OFDMChannel(channel_model = Rayleigh,
                       resource_grid = rg)
        else:
            raise ValueError(f"Unsupported channel selection: {self.channel_selection}. Please choose 'AWGN' or 'Rayleigh'.")
        # FEC encoder / decoder
        self.encoder = encoder
        self.decoder = decoder

    @tf.function() # enable graph mode for increased throughputs
    def call(self, u, ebno_db):

        # calculate noise variance
        if self.sim_esno:
                no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=1,
                       coderate=1)
        else:
            no = ebnodb2no(ebno_db,
                           num_bits_per_symbol=self.num_bits_per_symbol,
                           coderate=self.k/self.n)

        c = self.encoder(u) # explicitly encode

        x = self.mapper(c) # map c to symbols x

        # Transmit over the channel
        y = self.channel([x, no])

        # Recover input for Rayleigh channel

        # y = recover_input(y, channel_matrix, fft_size=self.constellation.num_points)

        llr_ch = self.demapper([y, no]) # demap y to LLRs

        u_hat = self.decoder(llr_ch) # run FEC decoder (incl. rate-recovery)

        if self.cw_estimates:
            return c, u_hat

        return u, u_hat