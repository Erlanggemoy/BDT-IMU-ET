import math
import time

#Rotation Counter

# Pola 4 titik sampel (kiri, bawah, kanan, atas) pada threshold -> kode sisi balok
# 1 = merah penuh, 2 = putih penuh, 3-6 = diagonal
FACE_STATE_MAP = {
    (0, 0, 0, 0): 1,
    (255, 255, 255, 255): 2,
    (255, 255, 0, 0): 3,
    (255, 0, 0, 255): 4,
    (0, 0, 255, 255): 5,
    (0, 255, 255, 0): 6,
}
# Urutan diagonal searah jarum jam pada tampilan kamera
# (sudut terang: kiri-bawah -> kiri-atas -> kanan-atas -> kanan-bawah)
DIAG_CYCLE = [3, 4, 5, 6]

class BlockRotationTracker:
    """Menghitung rotasi / flip / pengangkatan tiap balok.

    Terlihat dari sisi atas (dihitung langsung):
      - diagonal -> diagonal  = rotasi in-plane (kelipatan 90 derajat)
      - perubahan jenis sisi  = flip

    Tidak terlihat dari sisi atas (dihitung sebagai "hidden"):
      Flip merah -> merah atau putih -> putih (dan putar 90/180 derajat pada sisi polos)
      tidak mengubah tampilan sisi atas. Yang bisa dideteksi hanya jejaknya:
      balok disentuh tangan lalu terangkat (ukuran bbox berubah / sisi atas tertutup),
      kemudian diletakkan lagi dengan sisi atas yang sama. Angka ini adalah batas atas:
      ia mencakup flip ke sisi kembar DAN sekadar mengambil-lalu-memindahkan balok.

    Balok dilacak antar frame lewat kedekatan posisi. Ganti level mereset tracking.
    """

    def __init__(self, stable_frames=6, max_match_dist=120, handled_match_dist=350,
                 max_missed=90, hand_margin=30, lift_area_ratio=0.12,
                 lost_frames=8, release_frames=8):
        self.stable_frames = stable_frames
        self.max_match_dist = max_match_dist          # jarak pencocokan balok diam (px)
        self.handled_match_dist = handled_match_dist  # jarak pencocokan balok yang sedang dipegang (px)
        self.max_missed = max_missed
        self.hand_margin = hand_margin                # px, perluasan bbox untuk cek tangan
        self.lift_area_ratio = lift_area_ratio        # perubahan luas bbox dianggap terangkat
        self.lost_frames = lost_frames                # frame tertutup berturut-turut = terangkat
        self.release_frames = release_frames          # frame tenang berturut-turut = selesai dipegang
        self.tracks = {}
        self.next_id = 0
        self.level = None
        self.stats = {}
        self.events = []

    def _level_stats(self, level):
        return self.stats.setdefault(level, {"rotations": 0, "degrees": 0, "flips": 0, "hidden": 0})

    def _hand_near(self, bbox, hand_points):
        if not hand_points or bbox is None:
            return False
        x1, y1, x2, y2 = bbox
        m = self.hand_margin
        for hx, hy in hand_points:
            if x1 - m <= hx <= x2 + m and y1 - m <= hy <= y2 + m:
                return True
        return False

    def _match(self, observations):
        pairs = []
        for tid, t in self.tracks.items():
            limit = self.handled_match_dist if t["ep"] is not None else self.max_match_dist
            for oi, obs in enumerate(observations):
                d = math.hypot(t["x"] - obs[0], t["y"] - obs[1])
                if d <= limit:
                    pairs.append((d, tid, oi))
        pairs.sort()
        used_t, used_o, result = set(), set(), {}
        for d, tid, oi in pairs:
            if tid in used_t or oi in used_o:
                continue
            used_t.add(tid)
            used_o.add(oi)
            result[oi] = tid
        return result

    def _register_change(self, tid, old, new):
        s = self._level_stats(self.level)
        if old in DIAG_CYCLE and new in DIAG_CYCLE:
            step = (DIAG_CYCLE.index(new) - DIAG_CYCLE.index(old)) % 4  # 1=90 CW, 2=180, 3=90 CCW
            s["rotations"] += 1
            s["degrees"] += 180 if step == 2 else 90
            kind = "rotate"
        else:
            s["flips"] += 1
            kind = "flip"
        if self.tracks[tid]["ep"] is not None:
            self.tracks[tid]["ep"]["changed"] = True
        self.events.append((time.time(), self.level, tid, old, new, kind))

    def _episode(self, tid, hand, occluded, size_off, observed, hand_available):
        """Deteksi satu siklus 'balok dipegang' -> 'balok kembali tenang'."""
        t = self.tracks[tid]
        t["occl"] = t["occl"] + 1 if occluded else 0
        # Tertutup lama hanya dianggap terangkat jika ada tangan di dekatnya
        # (kalau pelacakan tangan tidak tersedia, cukup tertutup lama)
        lift_now = size_off or (t["occl"] >= self.lost_frames and (hand or not hand_available))

        ep = t["ep"]
        if ep is None:
            if hand or lift_now:
                t["ep"] = {"lifted": lift_now, "changed": False, "calm": 0}
            return
        ep["lifted"] = ep["lifted"] or lift_now
        calm = observed and not occluded and not hand and not size_off
        ep["calm"] = ep["calm"] + 1 if calm else 0
        if ep["calm"] >= self.release_frames:
            if ep["lifted"] and not ep["changed"]:
                self._level_stats(self.level)["hidden"] += 1
                self.events.append((time.time(), self.level, tid, t["state"], t["state"], "hidden"))
            t["ep"] = None

    def update(self, level, observations, hand_points=(), hand_available=False):
        """observations: list of (cx, cy, face_state, x1, y1, x2, y2[, conf]); face_state 0 = tidak dikenali.
        hand_points: titik landmark tangan (px) dari MediaPipe."""
        if level != self.level:
            self.level = level
            self.tracks = {}
            self._level_stats(level)

        matched = self._match(observations)
        seen = set()

        for oi, (ox, oy, state, x1, y1, x2, y2, *_) in enumerate(observations):
            tid = matched.get(oi)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"x": ox, "y": oy, "state": None, "buf": [], "missed": 0,
                                    "bbox": None, "ref_area": None, "ep": None, "occl": 0}
            seen.add(tid)
            t = self.tracks[tid]
            t["x"], t["y"], t["missed"] = ox, oy, 0
            t["bbox"] = (x1, y1, x2, y2)

            area = max(1, (x2 - x1) * (y2 - y1))
            known = state != 0
            hand = self._hand_near(t["bbox"], hand_points)
            if t["ep"] is None and not hand and known:      # luas acuan hanya saat balok diam
                t["ref_area"] = area if t["ref_area"] is None else 0.9 * t["ref_area"] + 0.1 * area
            size_off = t["ref_area"] is not None and abs(area / t["ref_area"] - 1) > self.lift_area_ratio

            if known:
                t["buf"] = (t["buf"] + [state])[-self.stable_frames:]
                if len(t["buf"]) == self.stable_frames and len(set(t["buf"])) == 1:
                    confirmed = t["buf"][0]
                    if t["state"] is None:
                        t["state"] = confirmed          # baseline, tidak dihitung
                    elif confirmed != t["state"]:
                        self._register_change(tid, t["state"], confirmed)
                        t["state"] = confirmed

            self._episode(tid, hand, not known, size_off, True, hand_available)

        for tid in list(self.tracks):
            if tid in seen:
                continue
            t = self.tracks[tid]
            t["missed"] += 1
            # balok tidak terdeteksi (tertutup tangan): perlakukan sebagai oklusi
            self._episode(tid, self._hand_near(t["bbox"], hand_points), True, False, False, hand_available)
            if t["missed"] > self.max_missed:
                del self.tracks[tid]

    def level_counts(self, level):
        s = self._level_stats(level)
        return s["rotations"], s["flips"], s["hidden"]

    def summary(self):
        total = {"rotations": 0, "degrees": 0, "flips": 0, "hidden": 0}
        for s in self.stats.values():
            for k in total:
                total[k] += s[k]
        return total
