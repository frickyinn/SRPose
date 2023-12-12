import torchvision.transforms as transforms


class RGBDAugmentor:
    """ perform augmentation on RGB-D video """

    def __init__(self):
        p_gray = 0.1
        self.augcolor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.4/3.14),
            transforms.RandomGrayscale(p=p_gray),
            transforms.ToTensor()])

    def color_transform(self, images):
        """ color jittering """
        num, ch, ht, wd = images.shape
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd*num)
        images = self.augcolor(images[[2,1,0]])
        return images[[2,1,0]].reshape(ch, ht, wd, num).permute(3,0,1,2).contiguous()

    def __call__(self, images):
        images = self.color_transform(images)

        return images
    