AOI_NAMES = ("AOI1", "AOI2", "B1", "B2", "B3", "B4")


def build_rects(screen_w, screen_h, image_rect):
    """AOI1/AOI2 = dua belahan layar; B1-B4 = empat kuadran gambar soal."""
    x1, y1, x2, y2 = image_rect
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    return {
        "AOI1": (0, 0, screen_w // 2, screen_h),
        "AOI2": (screen_w // 2, 0, screen_w, screen_h),
        "B1": (x1, y1, mx, my),
        "B2": (mx, y1, x2, my),
        "B3": (x1, my, mx, y2),
        "B4": (mx, my, x2, y2),
    }


class AoiTracker:
    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._rects = {}
        self._prev_ts = None
        self.dwell = {k: 0.0 for k in AOI_NAMES}

    def reset(self):
        self._prev_ts = None
        self.dwell = {k: 0.0 for k in AOI_NAMES}

    def set_rects(self, rects):
        # Ganti objeknya, jangan ubah dict lama: on_gaze jalan di thread lain.
        self._rects = dict(rects)

    def on_gaze(self, ts_us, x_norm, y_norm):
        rects = self._rects
        if not rects:
            return                      # sebelum menyentuh timestamp (perilaku lama)
        prev, self._prev_ts = self._prev_ts, ts_us
        if prev is None:
            return
        dt = (ts_us - prev) / 1_000_000
        if not (0 < dt < 0.5):          # timestamp sudah maju walau dt ditolak
            return
        gx = x_norm * self.screen_w
        gy = y_norm * self.screen_h
        for name, (x1, y1, x2, y2) in rects.items():
            if x1 <= gx < x2 and y1 <= gy < y2:
                self.dwell[name] += dt

    def snapshot(self):
        return dict(self.dwell)