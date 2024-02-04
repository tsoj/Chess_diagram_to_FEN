import torch
import torchvision


import skimage.measure
import skimage.morphology

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_bbox(model, img: torch.Tensor):
    model.eval()
    model.to(device)
    with torch.no_grad():
        assert (
            len(img.shape) == 3
        ), "Need input to be of shape [C, H, W] but is: " + str(img.shape)
        assert img.shape[0] == 3, "Channel dimension must be 3 (RGB)"
        img = img.unsqueeze(0)
        mask = torch.where(model(img.to(device)) < 0.5, 0.0, 1.0).cpu().squeeze(1)
        mask = mask.to(bool).numpy()
        labelled = skimage.measure.label(mask)
        rp = skimage.measure.regionprops(labelled)
        size = max([i.area for i in rp] + [1])
        mask = skimage.morphology.remove_small_objects(mask, min_size=size - 1)
        mask = torch.tensor(mask).to(float)


        # from matplotlib import pyplot as plt
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(mask.squeeze(0))
        # ax2.imshow(mask.squeeze(0))
        # plt.show()

        if len(torch.nonzero(mask)) == 0:
            return None
        output_box = torchvision.ops.masks_to_boxes(mask)

    model.train()
    return output_box.squeeze(0)
