from typing import List, Mapping


class Coordinates:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Coordinates(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Coordinates(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return Coordinates(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        return Coordinates(self.x / other, self.y / other, self.z / other)

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def __repr__(self):
        return f"Coordinates({self.x}, {self.y}, {self.z})"

    def _transform(self, shift_x, shift_y, shift_z, factor_x, factor_y, factor_z):
        self.x = (self.x - shift_x) / factor_x if factor_x != 0 else 0
        self.y = (self.y - shift_y) / factor_y if factor_y != 0 else 0
        self.z = (self.z - shift_z) / factor_z if factor_z != 0 else 0

    def norm(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def normalize(self) -> None:
        norm = self.norm()
        self.x /= norm
        self.y /= norm
        self.z /= norm
        return self

    @classmethod
    def normalize_list(cls, coords: List["Coordinates"], method: str = "minmax") -> List["Coordinates"]:
        if method == "minmax":
            range_x = max([coord.x for coord in coords]) - min([coord.x for coord in coords])
            range_y = max([coord.y for coord in coords]) - min([coord.y for coord in coords])
            range_z = max([coord.z for coord in coords]) - min([coord.z for coord in coords])
            shift_x = min([coord.x for coord in coords])
            shift_y = min([coord.y for coord in coords])
            shift_z = min([coord.z for coord in coords])
        for coord in coords:
            coord._transform(shift_x, shift_y, shift_z, range_x, range_y, range_z)
        return coords

    @classmethod
    def normalize_dict(cls, coords: Mapping[str, "Coordinates"], method: str = "minmax") -> Mapping[str, "Coordinates"]:
        if method == "minmax":
            range_x = max([coord.x for coord in coords.values()]) - min([coord.x for coord in coords.values()])
            range_y = max([coord.y for coord in coords.values()]) - min([coord.y for coord in coords.values()])
            range_z = max([coord.z for coord in coords.values()]) - min([coord.z for coord in coords.values()])
            shift_x = min([coord.x for coord in coords.values()])
            shift_y = min([coord.y for coord in coords.values()])
            shift_z = min([coord.z for coord in coords.values()])
        for coord in coords.values():
            coord._transform(shift_x, shift_y, shift_z, range_x, range_y, range_z)
        return coords

    @classmethod
    def cross_product(cls, a, b) -> "Coordinates":
        return Coordinates(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
