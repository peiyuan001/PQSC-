import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import tensorflow as tf
from .sionna_model import System_Model

def image_normalization(norm_type):
    def _inner(tensor: torch.Tensor):
        if norm_type == 'normalization':
            return tensor / 255.0
        elif norm_type == 'denormalization':
            return tensor * 255.0
        else:
            raise Exception('Unknown type of normalization')
    return _inner


def get_psnr(image, gt, max_z=255):

    mse = F.mse_loss(image, gt)

    psnr = 10 * torch.log10(max_z**2 / mse)
    return psnr

def quantization(input, max_z, min_z, quanti_level):

    # Move input, max_z, and min_z to GPU
    input = input
    max_z = max_z
    min_z = min_z

    # Compute the quantization step size
    quant_step = (max_z - min_z) / (quanti_level - 1)

    # Quantize the matrix
    normalized = (input - min_z) / (max_z - min_z)  # Normalize to [0, 1]
    levels = torch.round(normalized * (quanti_level - 1))  # PyTorch uses torch.round for rounding
    quantized = min_z + (levels * quant_step)  # Multiply by quant_step instead

    # Calculate the squared difference
    squared_diff = torch.square(quantized - input)

    # Calculate the mean squared error (MSE)
    mse = torch.mean(squared_diff)

    # Calculate the PSNR
    psnr = 10 * torch.log10((max_z - min_z) ** 2 / mse)

    return levels.to(torch.int32), quantized.to(torch.float32), psnr.to(torch.float32)


def binarize(tensor, num_bits):


    # Calculate the maximum and minimum values that can be represented with the given number of bits
    max_val = (1 << num_bits) - 1
    min_val = 0  # Assuming only non-negative integers for simplicity

    # Check if any tensor elements are out of the allowable range
    out_of_bounds = torch.any(torch.gt(tensor, max_val)) or torch.any(torch.lt(tensor, min_val))

    if out_of_bounds:
        raise ValueError("Error: Tensor contains values that cannot be represented with the number of given bits.")

    # Generate a tensor of bit positions [num_bits-1, ..., 0]
    positions = torch.arange(num_bits - 1, -1, -1, dtype=torch.int32)
    positions = positions.to(tensor.device)

    # Expand dimensions for broadcasting
    expanded_tensor = tensor.unsqueeze(-1)
    expanded_positions = positions.unsqueeze(0)

    # Shift right and extract bits using bitwise operations
    shifted = torch.floor_divide(expanded_tensor, (1 << expanded_positions))
    binary_tensor = (shifted & 1).float()


    return binary_tensor

def flatten_dim_1(tensor):
    # Retrieve the shape of the tensor as a list
    shape = list(tensor.shape)
    # Combine the last two dimensions into one
    new_shape = shape[:-2] + [shape[-2] * shape[-1]]
    # Reshape the tensor to the new shape
    flattened_tensor = tensor.view(*new_shape)
    return flattened_tensor


def reshape_to_btc_k(tensor, k):
    # Calculate the total number of bits in the tensor
    total_bits = tensor.numel()

    # Calculate 'btc' to fit all bits, padding if necessary
    btc = (total_bits + k - 1) // k  # This ensures that all bits fit, adding extra space if needed

    # Flatten the tensor to make it easier to pad and reshape
    flat_tensor = tensor.view(-1)

    # Calculate the number of zeros needed for padding
    padding_size = btc * k - total_bits

    # Pad the tensor with zeros at the end
    padded_tensor = torch.nn.functional.pad(flat_tensor, (0, padding_size), "constant", 0)

    # Reshape the tensor to [btc, k]
    final_tensor = padded_tensor.view(btc, k)

    return final_tensor

def recover_original_data(reshaped_tensor, original_shape, total_bits=None):
    # If total_bits is not provided, calculate it from the original shape
    if total_bits is None:
        total_bits = 1
        for dim in original_shape:
            total_bits *= dim

    # Flatten the reshaped tensor to a single dimension
    flat_tensor = reshaped_tensor.view(-1)

    # Slice the tensor to remove any padding, using the total bits
    original_data_flat = flat_tensor[:total_bits]

    # Reshape back to the original shape
    original_tensor = original_data_flat.view(*original_shape)

    return original_tensor

def calculate_ber(tensor1, tensor2):
    # Ensure both tensors have the same shape
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape for BER calculation"

    # Count the number of bits that are different between the tensors
    num_errors = torch.sum(torch.ne(tensor1, tensor2).int())

    # Calculate the Bit Error Rate (BER)
    total_bits = tensor1.numel()
    ber = num_errors.float() / total_bits

    return ber.item()  # Convert to a Python float for the final result

def binary_to_float(input_tensor, bits_in_symbol, min, max):
    # Ensure input tensor is a tensor of integers (0 or 1)
    binary_tensor = input_tensor.to(torch.int64)

    # Get the shape of the input tensor
    shape = input_tensor.shape

    # Check if the last dimension is divisible by bits_in_symbol
    if shape[-1] % bits_in_symbol != 0:
        raise ValueError(f"The length of the last dimension ({shape[-1]}) must be divisible by {bits_in_symbol}.")

    # Calculate the number of groups of bits_in_symbol bits along the last dimension
    num_groups = shape[-1] // bits_in_symbol

    # New shape with the last dimension grouped by bits_in_symbol
    new_shape = shape[:-1] + (num_groups, bits_in_symbol)

    # Reshape the tensor to group every 'bits_in_symbol' bits
    reshaped_tensor = binary_tensor.view(new_shape)

    # Convert binary groups to decimal numbers
    powers = torch.arange(bits_in_symbol - 1, -1, -1, dtype=torch.int64).to(input_tensor.device)
    quanti_level = torch.sum(reshaped_tensor * (2 ** powers), dim=-1)

    quanti_level_max = (2 ** bits_in_symbol) - 1
    float_data = min + (max - min) * (quanti_level / quanti_level_max)

    return quanti_level, float_data

def binary_to_integer(input_tensor, bits_in_symbol):
    # Ensure input tensor is a tensor of integers (0 or 1)
    binary_tensor = input_tensor.to(torch.int64)

    # Get the shape of the input tensor
    shape = input_tensor.shape

    # Check if the last dimension is divisible by bits_in_symbol
    if shape[-1] % bits_in_symbol != 0:
        raise ValueError(f"The length of the last dimension ({shape[-1]}) must be divisible by {bits_in_symbol}.")

    # Calculate the number of groups of bits_in_symbol bits along the last dimension
    num_groups = shape[-1] // bits_in_symbol

    # New shape with the last dimension grouped by bits_in_symbol
    new_shape = shape[:-1] + (num_groups, bits_in_symbol)

    # Reshape the tensor to group every 'bits_in_symbol' bits
    reshaped_tensor = binary_tensor.view(new_shape)

    # Convert binary groups to decimal numbers (integers)
    powers = torch.arange(bits_in_symbol - 1, -1, -1, dtype=torch.int64).to(input_tensor.device)
    integer_data = torch.sum(reshaped_tensor * (2 ** powers), dim=-1)

    return integer_data

def mse(tensor1, tensor2):
    # Calculate the squared difference between the tensors element-wise
    squared_diff = (tensor1 - tensor2) ** 2

    # Calculate the mean of the squared differences
    mse = torch.mean(squared_diff)

    return mse



if __name__ == '__main__':

    """
    funcitonal test for physical channel
    """

    try:
        import sionna
    except ImportError as e:
        # Install Sionna if package is not already installed
        import os

        os.system("pip install sionna")
        import sionna

    from sionna.mapping import Constellation, Mapper, Demapper
    from sionna.fec.polar import PolarEncoder, Polar5GEncoder, PolarSCLDecoder, Polar5GDecoder, PolarSCDecoder
    from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
    from sionna.fec.polar.utils import generate_5g_ranking, generate_rm_code
    from sionna.fec.conv import ConvEncoder, ViterbiDecoder, BCJRDecoder
    from sionna.fec.turbo import TurboEncoder, TurboDecoder
    from sionna.fec.linear import OSDecoder
    from sionna.utils import BinarySource, ebnodb2no
    from sionna.utils.metrics import count_block_errors
    from sionna.channel import AWGN
    from sionna.utils.plotting import PlotBER



    k = 336  # number of information bits per codeword
    n = 672  # desired codeword length
    num_bits_per_symbol = 12  # 2^(num_bits_per_symbol) QAM
    quanti_level = pow(2, num_bits_per_symbol)
    ebno = 20

    # -------------------sender-----------------------------
    # Generate the random 2-D matrix with data type float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = torch.rand(3, 3, dtype=torch.float32).to(device)
    print("z=", z)

    max_z = torch.max(z)
    min_z = torch.min(z)
    levels, quantized, quanti_psnr = quantization(z, max_z, min_z, quanti_level, device)
    print("max=", max_z, "min=", min_z)
    print("quantized_level=", levels, "quantized value=", quantized, "quanti_psnr=", quanti_psnr)

    binary_level = binarize(levels, num_bits_per_symbol, device)
    print("binary_level=", binary_level, binary_level.shape)

    binary_level = flatten_dim_1(binary_level)
    print("flattened binary_level=", binary_level, binary_level.shape)

    total_bits = torch.numel(binary_level)
    original_shape = binary_level.shape
    reshaped_source = reshape_to_btc_k(binary_level, k)
    print("Reshaped Tensor Shape:", reshaped_source.shape)

   #----------------channel (tf)----------------------------------------------

    reshaped_source_tf = tf.convert_to_tensor(reshaped_source.cpu().numpy())

    enc = LDPC5GEncoder(k=k, n=n)
    dec = LDPC5GDecoder(enc, num_iter=20)
    model = System_Model(k=k,
                         n=n,
                         enbo=ebno,
                         num_bits_per_symbol=num_bits_per_symbol,
                         encoder=enc,
                         decoder=dec)

    _, z_hat = model(reshaped_source_tf, ebno)
    print('z_hat', z_hat, z_hat.device, z_hat.shape)

    # -------------------receiver (torch)-----------------------------
    z_hat = torch.tensor(z_hat.numpy())

    recovered_tensor = recover_original_data(z_hat, original_shape, total_bits).to(device)
    print("Recovered Tensor Shape:", recovered_tensor, recovered_tensor.shape)

    BER = calculate_ber(binary_level, recovered_tensor)
    print('BER:', BER)

    quanti_level, float_data = binary_to_float(recovered_tensor, num_bits_per_symbol, min_z, max_z)
    print("Decimal Values from Binary Strings:")
    print('quanti_level', quanti_level, 'float data:', float_data)

    mse_float = mse(float_data, z)
    print('mse:', mse_float)