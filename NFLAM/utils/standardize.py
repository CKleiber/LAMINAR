import torch

def standardize(data: torch.Tensor) -> torch.Tensor:
    return (data - data.mean()) / data.std()