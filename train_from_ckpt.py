from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *
from utils.rotation import rotate_batch
from utils.save_model import save_ckp, load_ckp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default='datasets/')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--tensor', action='store_true')
########################################################################
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--nepoch', default=75, type=int)
parser.add_argument('--milestone_1', default=50, type=int)
parser.add_argument('--milestone_2', default=65, type=int)
parser.add_argument('--rotation_type', default='rand')
########################################################################
parser.add_argument('--outf', default='.')

args = parser.parse_args()

import os

if os.path.isdir('datasets/'):
    args.dataroot = 'datasets/'

my_makedir(args.outf)
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
net, ext, head, ssh = build_model_modules(args)
_, teloader = prepare_test_data(args)
_, trloader = prepare_train_data(args)

# load from checkpoint to resume training
checkpoint_dir = 'results/cifar10_tensor_layer2_gn_expand/ckpt.pth'
ckpt = torch.load(checkpoint_dir)
net.load_state_dict(ckpt['net'], strict=False)
head.load_state_dict(ckpt['head'], strict=False)
print('resume training from checkpoint ' + str(checkpoint_dir))

parameters = list(net.parameters()) + list(head.parameters())
optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)
criterion = nn.CrossEntropyLoss().cuda()

print("net's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

print("ssh's state_dict:")
for param_tensor in ssh.state_dict():
    print(param_tensor, "\t", ssh.state_dict()[param_tensor].size())

all_err_cls = []
all_err_ssh = []
all_loss = []
all_loss_ssh = []
print('Running...')
print('Error (%)\t\ttest\t\tself-supervised\t\tloss_last_batch\t\tloss_ssh_last_batch')
for epoch in range(1, args.nepoch + 1):
    net.train()
    ssh.train()

    for batch_idx, (inputs, labels) in enumerate(trloader):
        optimizer.zero_grad()
        inputs_cls, labels_cls = inputs.cuda(), labels.cuda()
        outputs_cls = net(inputs_cls)
        loss = criterion(outputs_cls, labels_cls)

        if args.shared is not None:
            inputs_ssh, labels_ssh = rotate_batch(inputs, args.rotation_type)
            inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
            outputs_ssh = ssh(inputs_ssh)
            loss_ssh = criterion(outputs_ssh, labels_ssh)
            loss += loss_ssh

        loss.backward()
        optimizer.step()

    err_cls = test(teloader, net)[0]
    err_ssh = 0 if args.shared is None else test(teloader, ssh, sslabel='expand')[0]
    all_err_cls.append(err_cls)
    all_err_ssh.append(err_ssh)
    all_loss.append(loss)
    all_loss_ssh.append(loss_ssh)
    scheduler.step()

    print(('Epoch %d/%d:' % (epoch, args.nepoch)).ljust(24) +
          '%.2f\t\t%.2f\t\t%.2f\t\t%.2f' % (err_cls * 100, err_ssh * 100, loss, loss_ssh))
    torch.save((all_err_cls, all_err_ssh), args.outf + '/loss.pth')
    torch.save((all_loss, all_loss_ssh), args.outf + '/train_loss_last_batch.pth')
    plot_epochs(all_err_cls, all_err_ssh, args.outf + '/loss.pdf')

    if epoch % 5 == 0:
        checkpoint_dir = args.outf + '/checkpoint' + str(epoch) + '.pt'
        print(checkpoint_dir)
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict_net': net.state_dict(),
            'state_dict_ssh': ssh.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_ckp(checkpoint, checkpoint_dir)

state = {'err_cls': err_cls, 'err_ssh': err_ssh,
         'net': net.state_dict(), 'head': head.state_dict(),
         'optimizer': optimizer.state_dict()}
torch.save(state, args.outf + '/ckpt.pth')
