from PIL import Image
from random import random

class StyleLoader:

    def __init__(self, stylize_image_path: str, p: float = 0.1) -> None:
        """Initialize StyleLoader with the the path to style images and 
        the probability of converting to a stylized image.

        :param stylize_image_path: Path to stylized version of current dataset,
        the structure must match the current dataset structure
        :type stylize_image_path: str
        :param p: Probability of returning stylized image, defaults to 0.1
        :type p: float, optional
        """
        self.stylize_image_path = stylize_image_path
        self.replaced_with_style = False
    
    def __call__(self, image_path: str) -> Image:
        """This function is called when the object is called,
        it takes in a image path and returns a PIL image or a stylized
        version of a PIL image with some probability.

        :param image_path: Path to original image
        :type image_path: str
        :return: PIL image.
        :rtype: Image
        """
        draw = random()
        img = None
        if draw < self.p:
            self.replaced_with_style = True
            # TODO: Implement Stylization code in the loader format.
        else:
            self.replaced_with_style = False
            img = Image.open(image_path)
        
    def was_replaced(self) -> bool:
        """Indicates whether the last image returned was
        stylized or not.

        :return: [description]
        :rtype: bool
        """
        return self.replaced_with_style