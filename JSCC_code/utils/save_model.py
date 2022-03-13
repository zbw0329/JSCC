import os
import torch


def save_model(args, model, optimizer, current_epoch, resnet):
    out = os.path.join(args.model_path, "{}_checkpoint_{}.tar".format(resnet, current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)
    print(f"the model is saved at {out}")
