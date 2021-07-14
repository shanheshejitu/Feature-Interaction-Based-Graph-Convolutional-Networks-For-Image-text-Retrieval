import torch.optim as Optim

def rate(lr_base, step, arg):
    lr = lr_base
    if step == 0:
        r = arg.lr_base * 1/4.
    elif step == 1:
        r = arg.lr_base * 2/4.
    elif step == 2:
        r = arg.lr_base * 3/4.
    # elif step == 3:
    #     r = arg.lr_base
    elif step in arg.lr_decay_list:
        r = adjust_lr(lr, arg.lr_decay_r)
    #elif step ==23:
        #r=adjust_lr(lr,arg.lr_decay_r1)

    else:
        r = lr

    return r


def get_optim(model):
    return  Optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))


def adjust_lr(lr_base, decay_r):
    lr_base *= decay_r
    return lr_base