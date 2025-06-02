import torch
import torch.nn as nn
import torch.nn.functional as F
import time

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


#------------------------channel related------------------------------------------------------------
import tensorflow as tf

def reshape_to_btc_k(tensor, k):
    # Calculate the total number of bits in the tensor
    total_bits = tf.size(tensor).numpy()

    # Calculate 'btc' to fit all bits, padding if necessary
    btc = (total_bits + k - 1) // k  # This ensures that all bits fit, adding extra space if needed

    # Flatten the tensor to make it easier to pad and reshape
    flat_tensor = tf.reshape(tensor, [-1])

    # Calculate the number of zeros needed for padding
    padding_size = btc * k - total_bits

    # Pad the tensor with zeros at the end
    padded_tensor = tf.pad(flat_tensor, [[0, padding_size]], "CONSTANT")

    # Reshape the tensor to [btc, k]
    final_tensor = tf.reshape(padded_tensor, [btc, k])

    return final_tensor

def recover_original_data(reshaped_tensor, original_shape, total_bits=None):
    # If total_bits is not provided, calculate it from the original shape
    if total_bits is None:
        total_bits = 1
        for dim in original_shape:
            total_bits *= dim

    # Flatten the reshaped tensor to a single dimension
    flat_tensor = tf.reshape(reshaped_tensor, [-1])

    # Slice the tensor to remove any padding, using the total bits
    original_data_flat = flat_tensor[:total_bits]

    # Reshape back to the original shape
    original_tensor = tf.reshape(original_data_flat, original_shape)

    return original_tensor


def quantization(input, max_z, min_z, quanti_level):

    # Compute the quantization step size
    quant_step = (max_z - min_z) / (quanti_level - 1)

    # Quantize the matrix
    normalized = (input - min_z) / (max_z - min_z)  # Normalize to [0, 1]
    levels = tf.round(normalized*(quanti_level-1))
    quantized = min_z + (max_z-min_z)*(levels/(quanti_level-1))

    squared_diff = tf.square(quantized - input)
    # Calculate the mean squared error (MSE)
    mse = tf.reduce_mean(squared_diff)
    psnr = 10 * tf.math.log(tf.square(max_z - min_z) / mse) / tf.math.log(10.0)

    return levels, quantized, psnr


def to_binary_single(n, bits):
        # Convert the integer to binary without the '0b' prefix
        binary = bin(n)[2:]
        # Check if the number of bits is sufficient to represent the integer
        if len(binary) > bits:
            raise ValueError(f"Number of bits is too small to represent the value {n}")
        # Pad the binary string with zeros on the left if necessary
        return binary.zfill(bits)


def is_iterable(item):
        try:
            iter(item)
            return not isinstance(item, str)
        except TypeError:
            return False

def to_binary_single(n, bits):
    # Convert the integer to binary without the '0b' prefix
    binary = bin(n)[2:]
    # Check if the number of bits is sufficient to represent the integer
    if len(binary) > bits:
        raise ValueError(f"Number of bits is too small to represent the value {n}")
    # Pad the binary string with zeros on the left if necessary
    return binary.zfill(bits)

def binarize(input_element, bits):
    # If the input element is iterable, apply binarize recursively
    if is_iterable(input_element):
        return [binarize(sub_element, bits) for sub_element in input_element]
    # If it's a single integer, convert it directly
    else:
        return to_binary_single(input_element, bits)

def flatten_bits(binary_matrix):
    # Flatten the binary matrix by iterating over each string and each bit in the strings
    if isinstance(binary_matrix, list):
        return [bit for sublist in binary_matrix for bit in flatten_bits(sublist)]
    else:
        # Convert each character in the string (bit) to float
        return [float(bit) for bit in binary_matrix]

def get_shape_and_bits(binary_matrix):
    # Determine the shape of the binary matrix and the number of bits per element
    shape = []
    temp = binary_matrix
    while isinstance(temp, list):
        shape.append(len(temp))
        temp = temp[0]
    num_bits = len(temp)  # Get the number of bits from one element
    return shape, num_bits

def to_tf_tensor(binary_matrix):
    # Flatten the binary matrix and convert each bit into float
    flattened_bits = flatten_bits(binary_matrix)

    # Convert the flattened list of floats to a TensorFlow tensor with dtype float32
    tensor = tf.convert_to_tensor(flattened_bits, dtype=tf.float32)

    # Get the shape and number of bits
    shape, num_bits = get_shape_and_bits(binary_matrix)

    # Expand the innermost dimension of the shape by the number of bits
    shape[-1] *= num_bits

    # Reshape the tensor to match the original nested list structure, expanded by the number of bits
    tensor = tf.reshape(tensor, shape)

    return tensor

def binarize_gpu(tensor, num_bits):

    # Calculate the maximum and minimum values that can be represented with the given number of bits
    max_val = (1 << num_bits) - 1
    min_val = 0  # Assuming only non-negative integers for simplicity

    # Check if any tensor elements are out of the allowable range
    condition = tf.logical_or(tf.greater(tensor, max_val), tf.less(tensor, min_val))
    out_of_bounds = tf.reduce_any(condition)

    if out_of_bounds:
        raise ValueError("Error: Tensor contains values that cannot be represented with the number of given bits.")

    # Generate a tensor of bit positions [num_bits-1, ..., 0]
    positions = tf.range(num_bits - 1, -1, -1, dtype=tf.int32)

    # Expand dimensions for broadcasting
    expanded_tensor = tensor[..., tf.newaxis]
    expanded_positions = positions[tf.newaxis, ...]

    # Shift right and extract bits
    binary_tensor = tf.bitwise.bitwise_and(tf.bitwise.right_shift(expanded_tensor, expanded_positions), 1)

    binary_tensor = tf.cast(binary_tensor, dtype=tf.float32)
    return binary_tensor

def flatten_dim_1(tensor):
    # Retrieve the shape of the tensor as a list
    shape = tensor.shape.as_list()
    # Combine the last two dimensions into one
    new_shape = shape[:-2] + [shape[-2] * shape[-1]]
    # Reshape the tensor to the new shape
    flattened_tensor = tf.reshape(tensor, new_shape)
    return flattened_tensor

def flatten_bits(binary_matrix):
    # Flatten the binary matrix by iterating over each string and each bit in the strings
    if isinstance(binary_matrix, list):
        return [bit for sublist in binary_matrix for bit in flatten_bits(sublist)]
    else:
        # Convert each character in the string (bit) to float
        return [float(bit) for bit in binary_matrix]


def bit_to_binary_float(input_tensor, bits_in_symbol, min_z, max_z):
    # Ensure input tensor is a tensor of integers (0 or 1)
    binary_tensor = input_tensor.int()

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
    reshaped_tensor = binary_tensor.view(*new_shape)

    # Convert binary groups to decimal numbers
    powers = torch.pow(2, torch.arange(bits_in_symbol - 1, -1, -1, dtype=torch.int)).to(input_tensor.device)
    quanti_level = torch.matmul(reshaped_tensor, powers)

    quanti_level_max = pow(2, bits_in_symbol) - 1
    float_data = min_z + (max_z-min_z)*(quanti_level/quanti_level_max)

    return quanti_level, float_data

def calculate_ber(tensor1, tensor2):
    # Ensure both tensors have the same shape
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape for BER calculation"

    # Count the number of bits that are different between the tensors
    num_errors = tf.reduce_sum(tf.cast(tf.not_equal(tensor1, tensor2), tf.int32))

    # Calculate the Bit Error Rate (BER)
    total_bits = tf.size(tensor1).numpy()
    ber = num_errors.numpy() / total_bits

    return ber

def mse(tensor_1, tensor_2):

    # Calculate the squared difference between the tensors element-wise
    squared_diff = tf.square(tensor_1 - tensor_2)
    # Calculate the mean of the squared differences
    mse = tf.reduce_mean(squared_diff)

    return mse


