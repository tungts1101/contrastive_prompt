def log_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params}, Trainable params: {trainable_params}, Ratio: {trainable_params * 100 / total_params:.2f}%')