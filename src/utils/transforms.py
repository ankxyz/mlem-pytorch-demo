from torchvision import transforms as T
from typing import List



def get_transforms() -> T.Compose:

    transforms = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((28, 28)),
        T.ToTensor()
    ])

    return transforms

