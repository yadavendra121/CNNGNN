import torch


def tta_inference(model, volume):

    model.eval()

    with torch.no_grad():

        pred1, _, _, _ = model(volume)

        flipped = torch.flip(volume, dims=[2])
        pred2, _, _, _ = model(flipped)
        pred2 = torch.flip(pred2, dims=[2])

        flipped = torch.flip(volume, dims=[3])
        pred3, _, _, _ = model(flipped)
        pred3 = torch.flip(pred3, dims=[3])

        final = (pred1 + pred2 + pred3) / 3.0

    return final
