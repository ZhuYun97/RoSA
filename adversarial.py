import torch

def ad_training(model, node_attack, perturb_shape, args, device):
    model.train()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(-args.step_size, args.step_size).to(device)
    perturb.requires_grad_()
    
    loss = node_attack(perturb)
    loss /= args.m

    for i in range(args.m-1):
        loss.backward()
        perturb_data = perturb.detach() + args.step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        loss = node_attack(perturb)
        loss /= args.m

    return loss