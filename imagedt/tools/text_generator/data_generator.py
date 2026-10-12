import os
import random

from PIL import Image, ImageFilter

from .background_generator import BackgroundGenerator
from .computer_text_generator import ComputerTextGenerator
from .distorsion_generator import DistorsionGenerator


class FakeTextDataGenerator:
    @classmethod
    def generate_from_tuple(cls, t):
        """
        Same as generate, but takes all parameters as one tuple
        """
        return cls.generate(*t)

    @classmethod
    def generate(
        cls,
        index,
        text,
        font,
        out_dir,
        size,
        extension,
        skewing_angle,
        random_skew,
        blur,
        random_blur,
        background_type,
        distorsion_type,
        distorsion_orientation,
        is_handwritten,
        name_format,
        width,
        alignment,
        text_color,
        orientation,
        space_width,
        save=False,
    ):

        image = None
        ##########################
        # Create picture of text #
        ##########################
        # remove handwirtten style
        # if is_handwritten:
        #     if orientation == 1:
        #         raise ValueError("Vertical handwritten text is unavailable")
        #     image = HandwrittenTextGenerator.generate(text)
        # else:
        if size <= 10 or width < 0:
            raise ValueError("size must exceed 10 and width must be nonnegative")
        if is_handwritten:
            raise NotImplementedError(
                "Handwritten generation requires external pretrained assets and is not supported by this generator"
            )
        image = ComputerTextGenerator.generate(
            text, font, text_color, size, orientation, space_width
        )

        random_angle = random.randint(0 - skewing_angle, skewing_angle)

        rotated_img = image.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        #############################
        # Apply distorsion to image #
        #############################
        if distorsion_type == 0:
            distorted_img = rotated_img  # Mind = blown
        elif distorsion_type == 1:
            distorted_img = DistorsionGenerator.sin(
                rotated_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        elif distorsion_type == 2:
            distorted_img = DistorsionGenerator.cos(
                rotated_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        else:
            distorted_img = DistorsionGenerator.random(
                rotated_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )

        ##################################
        # Resize image to desired format #
        ##################################

        # Horizontal text
        if orientation == 0:
            new_width = int(
                float(distorted_img.size[0] + 10)
                * (float(size) / float(distorted_img.size[1] + 10))
            )
            resized_img = distorted_img.resize(
                (new_width, size - 10), Image.Resampling.LANCZOS
            )
            background_width = width if width > 0 else new_width + 10
            background_height = size
        # Vertical text
        elif orientation == 1:
            new_height = int(
                float(distorted_img.size[1] + 10)
                * (float(size) / float(distorted_img.size[0] + 10))
            )
            resized_img = distorted_img.resize(
                (size - 10, new_height), Image.Resampling.LANCZOS
            )
            background_width = size
            background_height = new_height + 10
        else:
            raise ValueError("Invalid orientation")

        #############################
        # Generate background image #
        #############################
        if background_type == 0:
            background = BackgroundGenerator.gaussian_noise(
                background_height, background_width
            )
        elif background_type == 1:
            background = BackgroundGenerator.plain_white(
                background_height, background_width
            )
        elif background_type == 2:
            background = BackgroundGenerator.quasicrystal(
                background_height, background_width
            )
        else:
            background = BackgroundGenerator.picture(
                background_height, background_width
            )

        #############################
        # Place text with alignment #
        #############################

        new_text_width, _ = resized_img.size

        if alignment == 0:
            background.paste(resized_img, (5, 5), resized_img)
        elif alignment == 1:
            background.paste(
                resized_img,
                (int(background_width / 2 - new_text_width / 2), 5),
                resized_img,
            )
        else:
            background.paste(
                resized_img, (background_width - new_text_width - 5, 5), resized_img
            )

        ##################################
        # Apply gaussian blur #
        ##################################

        final_image = background.filter(
            ImageFilter.GaussianBlur(
                radius=(blur if not random_blur else random.randint(0, blur))
            )
        )

        #####################################
        # Generate name for resulting image #
        #####################################
        if name_format == 0:
            image_name = f"{text}_{str(index)}.{extension}"
        elif name_format == 1:
            image_name = f"{str(index)}_{text}.{extension}"
        elif name_format == 2:
            image_name = f"{str(index).zfill(8)}.{extension}"
        else:
            print(f"{name_format} is not a valid name format. Using default.")
            image_name = f"{text}_{str(index)}.{extension}"

        # Save the image
        if save:
            os.makedirs(out_dir, exist_ok=True)
            final_image.convert("RGB").save(os.path.join(out_dir, image_name))
        return final_image, text
