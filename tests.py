import cv2

from dataset import BigBIRDObjectData


def test_crop():
    bd = BigBirdObjectData('/home/yuri/Research/Data/BigBIRD/Data/honey_bunches_of_oats_with_almonds')
    bd.crop_dims = (224, 224)
    im = bd.rgb[0][10]
    # mask = bd.mask[0][10]
    im_cropped = bd.cropped_mask[0][10]
    box = bd.bbox[0][10]
    cv2.rectangle(im, (box.left, box.top), (box.right, box.bottom), (255, 255, 0), 2)
    cv2.imshow('abc1', im)
    cv2.imshow('abc2', im_cropped)


def test_preprocess():
    from PIL import Image
    import numpy as np

    import torch
    from torchvision import transforms
    import torchvision.models as models
    # %%
    # bd = BigBirdObjectData('/home/yuri/Research/Data/BigBIRD/Data/honey_bunches_of_oats_with_almonds')
    # bd = BigBirdObjectData('/home/yuri/Research/Data/BigBIRD/Data/listerine_green')
    bd = BigBirdObjectData('/home/yuri/Research/Data/BigBIRD/Data/tapatio_hot_sauce')

    print(bd.cropped_rgb[0][47][:, :, ::-1].shape)
    bd.crop_dims = (224, 224)
    Image.fromarray(bd.cropped_rgb[0][0][:, :, ::-1])
    # Image.fromarray(bd.mask[0][0][:, :, ::-1])
    # %%
    alexnet = models.alexnet(pretrained=True)
    # %%
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    fcn = transforms.Compose([transforms.Lambda(lambda im: im[:, :, ::-1]),
                              transforms.Lambda(lambda im: im.transpose((2, 0, 1))),
                              transforms.Lambda(lambda im: im / 255.0),
                              transforms.Lambda(lambda im: torch.tensor(im, dtype=torch.float32)),
                              normalize])

    # r = fcn(bd.cropped_rgb[0][47])
    bd.preprocess_fcn = fcn
    r = bd.cropped_rgb[0][47]

if __name__ == "__main__":
    bd = BigBIRDObjectData('/home/yuri/Research/Data/BigBIRD/Data/honey_bunches_of_oats_with_almonds')
    pass