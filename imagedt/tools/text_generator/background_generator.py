import math
import os
import random

import cv2
import numpy as np
from PIL import Image


class BackgroundGenerator:
    @classmethod
    def gaussian_noise(cls, height, width):
        """
        Create a background with Gaussian noise (to mimic paper)
        """

        # We create an all white image
        image = np.ones((height, width)) * 255

        # We add gaussian noise
        cv2.randn(image, 235, 10)

        return Image.fromarray(image).convert("RGB")

    @classmethod
    def plain_white(cls, height, width):
        """
        Create a plain white background
        """

        return Image.new("L", (width, height), 255).convert("RGB")

    @classmethod
    def quasicrystal(cls, height, width):
        """
        Create a background with quasicrystal (https://en.wikipedia.org/wiki/Quasicrystal)
        """

        if width < 1 or height < 1:
            raise ValueError("background dimensions must be positive")
        y = np.linspace(-2 * math.pi, 2 * math.pi, width) if width > 1 else np.zeros(1)
        x = (
            np.linspace(-2 * math.pi, 2 * math.pi, height)
            if height > 1
            else np.zeros(1)
        )
        yy, xx = np.meshgrid(y, x)
        frequency = random.random() * 30 + 20
        phase = random.random() * 2 * math.pi
        rotations = random.randint(10, 20)
        radius = np.hypot(xx, yy)
        angle = np.arctan2(yy, xx)
        z = np.zeros_like(radius)
        for i in range(rotations):
            z += np.cos(
                radius * np.sin(angle + i * math.pi * 2 / rotations) * frequency + phase
            )
        pixels = np.clip(255 - np.rint(255 * z / rotations), 0, 255).astype(np.uint8)
        return Image.fromarray(pixels).convert("RGB")

    @classmethod
    def picture(cls, height, width):
        """
        Create a background with a picture
        """

        from PIL import ImageOps

        pictures = [
            os.path.join("./pictures", name)
            for name in sorted(os.listdir("./pictures"))
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        if not pictures:
            raise ValueError("No images found in the pictures folder")
        with Image.open(random.choice(pictures)) as picture:
            return ImageOps.fit(
                picture.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS
            )
