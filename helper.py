import torch
import torchvision

class HelperModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build(*args, **kwargs)

    def build(self, *args, **kwargs):
        raise NotImplementedError

def get_parameter_count(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def get_device(device):
    if device == 9 or not torch.cuda.is_available():
        return torch.device('cpu')
    else:
        return torch.device(f'cuda:{device}')