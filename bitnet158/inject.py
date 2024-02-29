import torch
import torch.nn as nn
from bitnet158 import BitLinear


def inject(model: nn.Module, copy_weights: bool = True, module_class=BitLinear):
    for name, child in model.named_modules():
        if isinstance(child, nn.Linear) and "." not in name and (not isinstance(child, module_class)):
            # Replace the nn.Linear with BitLinear matching in features and and out_features, and add it to the model
            bitlinear = module_class(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                dtype=child.weight.dtype,
            )
            # copy the weights and bias
            if copy_weights:
                bitlinear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    bitlinear.bias.data.copy_(child.bias.data)

            setattr(
                model,
                name,
                bitlinear,
            )
        elif isinstance(child, nn.Module) and name != "":
            # print(name)
            inject(child, copy_weights=copy_weights, module_class=module_class)
