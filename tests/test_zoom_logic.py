import unittest
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QTransform


class CanvasLogic:
    def __init__(self, w, h, zoom, px, py, img_w, img_h):
        self.w = w
        self.h = h
        self.zoom = zoom
        self.px = px
        self.py = py
        self.img_w = img_w
        self.img_h = img_h

        # Calculate display_rect (Fit center)
        ratio = min(w / img_w, h / img_h)
        nw, nh = int(img_w * ratio), int(img_h * ratio)
        self.display_rect = QRectF((w - nw) / 2, (h - nh) / 2, nw, nh)

    def get_transform(self):
        t = QTransform()
        t.translate(self.w / 2, self.h / 2)
        t.scale(self.zoom, self.zoom)
        t.translate(self.px * self.w, self.py * self.h)
        t.translate(-self.w / 2, -self.h / 2)
        return t

    def map_to_image(self, screen_pos):
        inv, ok = self.get_transform().inverted()
        local_pos = inv.map(screen_pos)

        if not self.display_rect.contains(local_pos):
            return None

        nx = (local_pos.x() - self.display_rect.x()) / self.display_rect.width()
        ny = (local_pos.y() - self.display_rect.y()) / self.display_rect.height()
        return nx, ny


class TestMapping(unittest.TestCase):
    def test_zoom_mapping(self):
        # 1000x1000 view, 1000x1000 image, 2x zoom
        logic = CanvasLogic(1000, 1000, 2.0, 0, 0, 1000, 1000)

        # Center click (500,500)
        nx, ny = logic.map_to_image(QPointF(500, 500))
        self.assertAlmostEqual(nx, 0.5)

        nx, ny = logic.map_to_image(QPointF(250, 250))
        self.assertAlmostEqual(nx, 0.375)

    def test_max_zoom_panned(self):
        # 400% zoom, panned right by 0.2
        logic = CanvasLogic(1000, 1000, 4.0, 0.2, 0, 1000, 1000)

        # Center of screen (500, 500)
        # 1. screen(500,500) -> inv center -> (0,0)
        # 2. inv scale 4x -> (0,0)
        # 3. inv pan 0.2 -> (0-200, 0) = (-200, 0)
        # 4. center back -> (300, 500)
        # local(300,500) relative to image(0,0,1000,1000) -> nx=0.3
        nx, ny = logic.map_to_image(QPointF(500, 500))
        self.assertAlmostEqual(nx, 0.3)


if __name__ == "__main__":
    unittest.main()
