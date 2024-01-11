import albumentations as A


class Augmentor(object):
    def __init__(self, is_training:bool):
        self.augmentor = A.Compose([
            A.MotionBlur(p=0.25),
            A.ColorJitter(p=0.25),
            A.ImageCompression(p=0.25),
            A.ISONoise(p=0.25),
            A.ToGray(p=0.1)
        ], p=float(is_training))

    def __call__(self, x):
        return self.augmentor(image=x)['image']
