import torch.nn as nn
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')


############################################################################
# Helper Utilities
############################################################################


def new_random_z(bs, z, seed=False):
    # Creates Z vector of normally distributed noise
    if seed:
        torch.manual_seed(seed)
    z = torch.FloatTensor(bs, z, 1, 1).normal_(0, 1).cuda()
    return z


def weights_init_normal(m):
    # Set initial state of weights
    classname = m.__class__.__name__
    if 'ConvTrans' == classname:
        pass
    elif 'Conv2d' in classname or 'ConvTrans' in classname:
        nn.init.orthogonal_(m.weight.data)


def mft(tensor):
    # Return mean float tensor #
    return torch.mean(torch.FloatTensor(tensor))


############################################################################
# Display Images
############################################################################


def show_test(gen, z, denorm, save=False):
    # Generate samples from z vector, show and also save
    gen.eval()
    results = gen(z)
    gen.train()
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for i, ax in enumerate(axes.flat):
        ax.imshow(denorm.denorm(results[i]))

    if save:
        plt.savefig(save)

    plt.show()
    plt.close(fig)
