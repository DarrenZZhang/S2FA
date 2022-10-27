from torch import nn
import torch

class model(nn.Module):
    def __init__(self, f, c):
        super(model, self).__init__()
        self.f = f
        self.c = c

    def forward(self, x):
        x = self.f(x)
        x = torch.flatten(x, start_dim=1)
        x = self.c(x)

        return x



class GuidedBackPropogation:
    def __init__(self, model):
        self.model = model
        self.hooks = []

        def backward_hook(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return tuple(grad.clamp(min=0.0) for grad in grad_in)

        for name, module in self.model.named_modules():
            self.hooks.append(module.register_backward_hook(backward_hook))

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __call__(self, *args, **kwargs):
        self.model.zero_grad()
        return self.model(*args, **kwargs)

    def get(self, layer):
        relu =  nn.ReLU()
        return relu(layer.grad).detach().cpu().numpy()
