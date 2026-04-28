import torch

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def batch_to_torch(batch, device):
    return tuple(torch.as_tensor(x, device=device) for x in batch)