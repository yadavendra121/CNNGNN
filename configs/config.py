class Config:
    num_classes = 16
    patch_size = (256, 256, 32)
    batch_size = 2
    lr = 3e-4
    weight_decay = 1e-4
    epochs = 500
    lambda_aux = 0.1
    lambda_boundary = 0.5
    lambda_contrast = 0.1
    device = "cuda"
    overlap = 0.5
    save_path ="model_weight/Abd/best_model.pth"