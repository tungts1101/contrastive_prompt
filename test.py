import torch

from prompts.vpt_vit_mae import (
    vit_base_patch16
)

if __name__ == "__main__":
    device = torch.device('cuda')

    model = vit_base_patch16({})
    ckpt = "./mae_pretrain_vit_base.pth"

    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['model']

    # model = vit_base()
    # ckpt = "./linear-vit-b-300ep.pth.tar"

    # checkpoint = torch.load(ckpt, map_location="cpu")
    # state_dict = checkpoint['state_dict']
    # for k in list(state_dict.keys()):
    #     # retain only base_encoder up to before the embedding layer
    #     if k.startswith('module.'):
    #         # remove prefix
    #         state_dict[k[len("module."):]] = state_dict[k]
    #     # delete renamed or unused k
    #     del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model.head = torch.nn.Identity()

    print(model)
    model.to(device)
    imgs = torch.randn(64, 3, 224, 224).to(device)
    output = model(imgs)
    print(output.shape)