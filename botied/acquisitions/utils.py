import torch
import botorch


def get_dummy_gp():
    dummy_tensor = torch.randn(1, 2)
    model = botorch.models.multitask.KroneckerMultiTaskGP(
        train_X=dummy_tensor,
        train_Y=dummy_tensor
    )
    return model
