"""Text rasterization using current Pillow font metrics."""

import random

from PIL import Image, ImageColor, ImageDraw, ImageFont


class ComputerTextGenerator:
    @classmethod
    def generate(cls, text, font, text_color, font_size, orientation, space_width):
        if not text:
            raise ValueError("text must be nonempty")
        if font_size <= 0 or space_width < 0:
            raise ValueError("font size must be positive and space_width nonnegative")
        if orientation not in (0, 1):
            raise ValueError(f"Unknown orientation {orientation}")
        image_font = ImageFont.truetype(font=font, size=int(font_size))
        colors = [ImageColor.getrgb(color) for color in text_color.split(",")]
        fill = tuple(
            random.randint(min(a, b), max(a, b))
            for a, b in zip(colors[0], colors[-1], strict=True)
        )
        if orientation == 0:
            words = text.split(" ")
            gap = round(image_font.getlength(" ") * space_width)
            widths = [max(1, round(image_font.getlength(word))) for word in words]
            height = max(1, max(image_font.getbbox(word)[3] for word in words))
            image = Image.new("RGBA", (sum(widths) + gap * (len(words) - 1), height))
            draw = ImageDraw.Draw(image)
            x = 0
            for word, width in zip(words, widths, strict=True):
                draw.text((x, 0), word, font=image_font, fill=fill)
                x += width + gap
        else:
            boxes = [image_font.getbbox(char) for char in text]
            heights = [
                max(1, box[3])
                if char != " "
                else max(1, round(font_size * space_width))
                for char, box in zip(text, boxes, strict=True)
            ]
            image = Image.new(
                "RGBA", (max(1, max(box[2] for box in boxes)), sum(heights))
            )
            draw = ImageDraw.Draw(image)
            y = 0
            for char, height in zip(text, heights, strict=True):
                draw.text((0, y), char, font=image_font, fill=fill)
                y += height
        return image
