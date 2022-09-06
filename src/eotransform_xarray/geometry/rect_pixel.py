from dataclasses import dataclass


@dataclass
class RectPixel:
    min_x: int
    min_y: int
    max_x: int
    max_y: int

    @property
    def width(self):
        return self.max_x - self.min_x

    @property
    def height(self):
        return self.max_y - self.min_y
