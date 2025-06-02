from types import SimpleNamespace

"""
    -- VQ-VAE Hyperparameters --
"""
_common = {
    'checkpoint_frequency':         5,
    'image_frequency':              5,
    'test_size':                    0.1,
    'nb_workers':                   4,
}


_cifar10 = {
    'display_name':             'CIFAR10',
    'image_shape':              (3, 32, 32),
    'in_channels':              3,
    'hidden_channels':          128,
    'res_channels':             64,
    'nb_res_layers':            2,
    'embed_dim':                64,
    'nb_entries':               512,
    'nb_levels':                1,
    'scaling_rates':            [2],
    'noise_db':                 20,
    'learning_rate':            1e-3,
    'beta':                     0.25,
    'batch_size':               32,
    'mini_batch_size':          32,
    'max_epochs':               100,
}

HPS_VQVAE = {
    'cifar10':              SimpleNamespace(**{**_common, **_cifar10}),
}


