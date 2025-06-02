"""
This file reproduces the GPU accelerated LWE algorithm in [2] with simple test cases.

[1] Lindner, R., & Peikert, C. (2011). Better key sizes (and attacks) for LWE-based encryption. In Topics in Cryptology–CT-RSA 2011: The Cryptographers’ Track at the RSA Conference 2011, San Francisco, CA, USA, February 14-18, 2011. Proceedings (pp. 319-339). Springer Berlin Heidelberg.
[2] Tung, T. Y., & Gündüz, D. (2023, May). Deep joint source-channel and encryption coding: Secure semantic communications. In ICC 2023-IEEE International Conference on Communications (pp. 5620-5625). IEEE.
"""
import torch
import math
from tqdm import tqdm
import torch.nn as nn


def ber_check(matrix1, matrix2):
    """
    # Example usage:
    # matrix1 = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.int32)
    # matrix2 = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.int32)
    # ratio = ber_check(matrix1, matrix2)
    # print(ratio)
    """
    if matrix1.shape != matrix2.shape:
        raise ValueError("The matrices must have the same shape.")

    # Calculate the number of different elements
    different_elements = torch.sum(matrix1 != matrix2).item()

    # Calculate the total number of elements
    total_elements = matrix1.numel()

    # Calculate the ratio of different elements
    ratio = different_elements / total_elements

    return ratio

class LWE_bk():

    def __init__(self,
              seed_se               = 1,
              seed_sk               = 2,
              n_1                   = 192, # size of lattice
              n_2                   = 192,
              q                     = 4093, # prime number
              s_e                   = 3,
              s_k                   = 3,
              device                = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
              mode                  = 'Authorized' #Authorized: normal mode; Eaves: Test on Eavesdropper end
              ):

        """constants"""
        self.seed_se = seed_se
        self.seed_sk = seed_sk
        self.n_1 = n_1
        self.n_2 = n_2
        self.q = q
        self.s_k = s_k
        self.s_e = s_e
        self.device = device
        self.mode = mode

        """lattices"""
        self.A = None
        self.R1 = None
        self.R2 = None
        self.P = None
        self.e_1t = None
        self.e_2t = None
        self.e_3t = None


    def set_device(self, device):
        self.device = device

    def discrete_gaussian(self, modulus, mu, sigma, size):
        """
        # Example usage:
        # modulus = 23
        # mu = 0
        # sigma = 1
        # size = (1, 10)
        # samples = discrete_gaussian(modulus, mu, sigma, size)
        # print(samples)
        """
        # Generate a range of integer values around the mean

        t = (modulus-1)//4# error tolerance t=q/4
        t = torch.tensor(t, dtype=torch.float64)

        values = torch.arange(mu - t, mu + t + 1, dtype=torch.float64)

        # Calculate the PMF for each value
        pmf = torch.exp(-0.5 * ((values - mu) / sigma) ** 2)
        # Normalize the PMF
        pmf /= pmf.sum()

        # Sample from the discrete distribution
        samples = torch.multinomial(pmf, num_samples=size[0] * size[1], replacement=True)
        samples = values[samples].reshape(size).to(torch.float64)
        samples = torch.remainder(samples, modulus)

        return samples.to(torch.float64).to(self.device)

    def discrete_uniform(self, modulus, size):
        """
        # Example usage:
        # modulus = 23
        # size = (1, 10)
        # samples = discrete_uniform(modulus, size)
        # print(samples)
        """
        # Generate a range of integer values from 0 to modulus-1
        values = torch.arange(0, modulus, dtype=torch.float64)

        # Calculate the PMF for each value (uniform distribution)
        pmf = torch.ones(modulus, dtype=torch.float32) / modulus

        # Sample from the discrete uniform distribution
        samples = torch.multinomial(pmf, num_samples=size[0] * size[1], replacement=True)
        samples = values[samples].reshape(size).to(torch.float64)
        samples = torch.remainder(samples, modulus)

        return samples.to(torch.float64).to(self.device)


    def lwe_encoder(self, input, q):
        """
        # Example usage:
        # q = 23
        # input = torch.tensor([1, 0, 1, 1, 0], dtype=torch.int32)  # Example binary input
        # encoded = lwe_encoder(input, q)
        # print(encoded)
        """
        threshold = ((q - 1) // 2)  # [q/2]
        threshold = torch.tensor(threshold, dtype=torch.float64)

        return torch.remainder(input * threshold, q).to(torch.float64).to(self.device)  # [q/2]d


    def lwe_decoder(self, input, q):
        """
        # Example usage:
        # q = 23
        # input = torch.tensor([15, 5, 8, 20, 1], dtype=torch.int32)  # Example encoded input
        # decoded = lwe_decoder(input, q)
        # print(decoded)
        """
        threshold = ((q - 1) // 2)  # [q/2]
        ub = ((threshold * 3) // 2)  # [3/4 q]
        ub = torch.tensor(ub, dtype=torch.float64)
        lb = (threshold // 2)  # [1/4 q]
        lb = torch.tensor(lb, dtype=torch.float64)

        condition = (input <= ub) & (input > lb)
        return condition.clone().detach().to(dtype=torch.float64).to(self.device)




    def new_Lattice(self, z: torch.tensor):

        z = z.reshape(1, -1).to(torch.float64).to(self.device)
        l = z.numel()

        self.A = self.discrete_uniform(modulus=self.q, size=(self.n_1, self.n_2)).to(self.device)  # base, distribution: uniform, size:(n_1,n_2)
        self.R2 = self.discrete_gaussian(modulus=self.q, mu=0, sigma=self.s_k,
                                    size=(self.n_2, l)).to(self.device)  # secret key R2, distribution: discret Gaussian with s_k,size=(n_2, l)
        # print("R2=", R2)
        self.R1 = self.discrete_gaussian(modulus=self.q, mu=0, sigma=self.s_k,
                                    size=(self.n_1, l)).to(self.device)  # R1, distribution: discret Gaussian with s_k, size=(n_1, l)
        # print("R1=", R1)
        self.P = (self.R1 - self.A @ self.R2).to(self.device)  # public key

        self.e_1t = self.discrete_gaussian(modulus=self.q, mu=0, sigma=self.s_e,
                                      size=(1, self.n_1)).to(self.device)  # distribution: discret Gaussian with s_e
        self.e_2t = self.discrete_gaussian(modulus=self.q, mu=0, sigma=self.s_e,
                                      size=(1, self.n_2)).to(self.device)  # distribution: discret Gaussian with s_e
        self.e_3t = self.discrete_gaussian(modulus=self.q, mu=0, sigma=self.s_e,
                                      size=(1, l)).to(self.device)  # distribution: discret Gaussian with s_e


    def lwe_bk_encoder(self, z: torch.Tensor):

        self.new_Lattice(z)

        # encryption
        z_bar = self.lwe_encoder(z, self.q)  # lwe encoded
        # print("z_bar", z_bar, z_bar.shape)
        c_1t = torch.remainder((self.e_1t @ self.A + self.e_2t), self.q).to(self.device)
        c_2t = torch.remainder((self.e_1t @ self.P + self.e_3t + z_bar), self.q).to(self.device)
        cyphertext = [c_1t, c_2t]

        return cyphertext

    def lwe_bk_decoder(self, cyphertext):

        random_secret_key = self.discrete_gaussian(modulus=self.q, mu=0, sigma=self.s_k, size=self.R2.shape).to(self.device)
        # decryption
        if self.mode == 'Authorized':
            decoded_bar = torch.remainder((cyphertext[0] @ self.R2 + cyphertext[1]), self.q) .to(self.device) # [c_1t c_2t] . ([R2 I].T)
        elif self.mode == 'Eaves':
            decoded_bar = torch.remainder((cyphertext[0] @ random_secret_key + cyphertext[1]), self.q).to(self.device)
        else:
            raise Exception('Unknown type of lwe mode')

        decoded = self.lwe_decoder(decoded_bar, self.q).to(self.device)

        return decoded



"""functional testing"""
if __name__ == '__main__':
    print("lwe-test")
    device = 'cuda'
    lwe_module = LWE_bk()
    lwe_module.set_device(device)

    l = int(10000)
    batch_size = 1

    z = torch.randint(0, 2, (batch_size, l), dtype=torch.float64).to(torch.device(device))  # input latent --message bits

    # Perform encoding and decoding using the model
    cyphertext = lwe_module.lwe_bk_encoder(z)
    z_bar = lwe_module.lwe_bk_decoder(cyphertext)
    ber = ber_check(z, z_bar)
    print("ber=", ber)
    print(f"z:{z},z_bar:{z_bar}")
