import torch


def save_ckp(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)


def load_ckp(checkpoint_fpath, net, ssh, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    net.load_state_dict(checkpoint['state_dict_net'], strict=False)
    ssh.load_state_dict(checkpoint['state_dict_ssh'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
    return net, ssh, optimizer, checkpoint['epoch']


def load_trainable_weights(model, data_to_load):
    ii = 0
    for param_name, param in model.named_parameters():
        if 'trainable' in param_name:
            param.data = data_to_load[ii]
            ii += 1


def save_trainable_weights(model):
    data_to_save = []
    for param_name, param in model.named_parameters():
        if 'trainable' in param_name:
            data_to_save.append(param.data)
    return data_to_save


def reset_model(ckpt, model_name):
    for k in ckpt[model_name].keys():
        if 'trainable' in k:
            ckpt[model_name][k] = torch.ones(1, ckpt[model_name][k].shape[1])
