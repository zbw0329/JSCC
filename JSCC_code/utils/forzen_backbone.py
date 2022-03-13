def forzen_backbone(x):
    for name, value in x.named_parameters():
        if 'layer' in name:
            value.requires_grad = False
    print(f"Backbone's parameters have been frozen")
    return x
