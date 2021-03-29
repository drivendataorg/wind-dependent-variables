from skimage import draw
import torch
import numpy as np


def draw_circle(img, center, radius, invert=False):
    x, y, chs = img.shape

    if invert:
        new_image = np.zeros_like(img)
    else:
        new_image = img.copy()

    xc, yc = center
    coords = draw.disk((xc, yc), radius, shape=(x, y))
    for ch in range(chs):
        if invert:
            new_image[..., ch][coords] = img[..., ch][coords]
        else:
            new_image[..., ch][coords] = 0

    return new_image


def draw_ring(img, center, radius_low, radius_high, invert=False):
    x, y, chs = img.shape

    if invert:
        new_image = np.zeros_like(img)
    else:
        new_image = img.copy()

    xc, yc = center
    coords = draw.disk((xc, yc), radius_high, shape=(x, y))
    for ch in range(chs):
        if invert:
            new_image[..., ch][coords] = img[..., ch][coords]
        else:
            new_image[..., ch][coords] = 0

    coords = draw.disk((xc, yc), radius_low, shape=(x, y))
    for ch in range(chs):
        if invert:
            new_image[..., ch][coords] = 0
        else:
            new_image[..., ch][coords] = img[..., ch][coords]

    return new_image


class RandomRingMask(torch.nn.Module):
    def __init__(self, center_delta=(0.0, 0.0), radius_high=(0.5, 1.5), thickness=(0.25, 0.75), p=1.0, invert_p=0.0):
        super().__init__()
        self.center_delta = center_delta
        self.radius_high = radius_high
        self.thickness = thickness
        self.p = p
        self.invert_p = invert_p

    def _random_centre_disp(self, xc, yc):
        dx = int(xc * (torch.rand(1) * (self.center_delta[1] - self.center_delta[0]) + self.center_delta[0]))
        dy = int(yc * (torch.rand(1) * (self.center_delta[1] - self.center_delta[0]) + self.center_delta[0]))
        #print(xc, dx, yc, dy)
        new_xc = xc + dx
        new_yc = yc + dy
        return new_xc, new_yc

    def forward(self, img):
        if torch.rand(1) < self.p:
            tensor = False
            if isinstance(img, torch.Tensor):
                tensor = True
                img = img.numpy().transpose((1, 2, 0))

            x, y, chs = img.shape
            xc = yc = min(x, y) // 2
            xc, yc = self._random_centre_disp(xc, yc)
            radius_high = int(xc * (torch.rand(1) * (self.radius_high[1] - self.radius_high[0]) + self.radius_high[0]))
            radius_low = int(
                radius_high * (torch.rand(1) * (self.thickness[1] - self.thickness[0]) + self.thickness[0]))
            invert = torch.rand(1) < self.invert_p

            new_img = draw_ring(img, (xc, yc), radius_low, radius_high, invert=invert)
            if tensor:
                return torch.from_numpy(new_img.transpose((2, 0, 1)))
            return new_img
        return img


class RandomCircleMask(torch.nn.Module):
    def __init__(self, center_delta=(0.0, 0.0), radius=(0.5, 1.5), p=1.0, invert_p=0.0):
        super().__init__()
        self.center_delta = center_delta
        self.radius = radius
        self.p = p
        self.invert_p = invert_p

    def _random_centre_disp(self, xc, yc):
        dx = int(xc * (torch.rand(1) * (self.center_delta[1] - self.center_delta[0]) + self.center_delta[0]))
        dy = int(yc * (torch.rand(1) * (self.center_delta[1] - self.center_delta[0]) + self.center_delta[0]))
        #print(xc, dx, yc, dy)
        new_xc = xc + dx
        new_yc = yc + dy
        return new_xc, new_yc

    def forward(self, img):
        if torch.rand(1) < self.p:
            tensor = False
            if isinstance(img, torch.Tensor):
                tensor = True
                img = img.numpy().transpose((1, 2, 0))

            x, y, chs = img.shape
            xc = yc = min(x, y) // 2
            xc, yc = self._random_centre_disp(xc, yc)
            radius = int(xc * (torch.rand(1) * (self.radius[1] - self.radius[0]) + self.radius[0]))
            invert = torch.rand(1) < self.invert_p

            new_img = draw_circle(img, (xc, yc), radius, invert=invert)
            if tensor:
                return torch.from_numpy(new_img.transpose((2, 0, 1)))
            return new_img
        return img
