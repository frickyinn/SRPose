import albumentations as A


class Augmentor(object):
    def __init__(self):
        self.augmentor = A.Compose([
            A.MotionBlur(p=0.25),
            A.ColorJitter(p=0.25),
            A.ImageCompression(p=0.25),
            A.ISONoise(p=0.25),
            A.ToGray(p=0.1)
        ], p=1.0)

    def __call__(self, x):
        return self.augmentor(image=x)['image']
