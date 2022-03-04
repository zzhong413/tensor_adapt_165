import torch


def save_ckp(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)


def load_ckp(checkpoint_fpath, net, ssh, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    net.load_state_dict(checkpoint['state_dict_net'], strict=False)
    ssh.load_state_dict(checkpoint['state_dict_ssh'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
    return net, ssh, optimizer, checkpoint['epoch']
