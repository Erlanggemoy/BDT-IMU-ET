import pytest
from aoi import AoiTracker, build_rects, AOI_NAMES

W = H = 1000
IMG = (250, 250, 750, 750)          # titik tengah (500, 500)


def make(rects=True):
    tr = AoiTracker(W, H)
    if rects:
        tr.set_rects(build_rects(W, H, IMG))
    return tr


def feed(tr, samples):
    for ts, x, y in samples:
        tr.on_gaze(ts, x, y)


def only(tr, *names):
    """Pastikan hanya AOI di 'names' yang > 0."""
    for k, v in tr.dwell.items():
        if k in names:
            assert v > 0, k
        else:
            assert v == 0, k


def test_build_rects_layout():
    r = build_rects(1920, 1080, (1000, 100, 1400, 500))
    assert r["AOI1"] == (0, 0, 960, 1080)
    assert r["AOI2"] == (960, 0, 1920, 1080)
    assert r["B1"] == (1000, 100, 1200, 300)
    assert r["B2"] == (1200, 100, 1400, 300)
    assert r["B3"] == (1000, 300, 1200, 500)
    assert r["B4"] == (1200, 300, 1400, 500)


def test_build_rects_odd_size_uses_floor_midpoint():
    r = build_rects(1000, 1000, (101, 101, 500, 500))
    assert r["B2"] == (300, 101, 500, 300)


def test_first_sample_counts_nothing():
    tr = make()
    feed(tr, [(0, 0.3, 0.3)])
    only(tr)


def test_dwell_accumulates_in_overlapping_aois():
    tr = make()
    feed(tr, [(0, 0.3, 0.3), (30_000, 0.3, 0.3)])   # (300,300): AOI1 + B1
    only(tr, "AOI1", "B1")
    assert tr.dwell["AOI1"] == pytest.approx(0.03)
    assert tr.dwell["B1"] == pytest.approx(0.03)


def test_split_boundary_belongs_to_right_half():
    tr = make()
    feed(tr, [(0, 0.5, 0.1), (30_000, 0.5, 0.1)])    # x = 500 tepat
    only(tr, "AOI2")


def test_just_left_of_split_belongs_to_aoi1():
    tr = make()
    feed(tr, [(0, 0.499, 0.1), (30_000, 0.499, 0.1)])
    only(tr, "AOI1")


def test_quadrant_boundaries_are_half_open():
    tr = make()
    feed(tr, [(0, 0.5, 0.5), (30_000, 0.5, 0.5)])    # (500,500)
    only(tr, "AOI2", "B4")
    tr = make()
    feed(tr, [(0, 0.25, 0.25), (30_000, 0.25, 0.25)])  # (250,250) tepi kiri-atas gambar
    only(tr, "AOI1", "B1")
    tr = make()
    feed(tr, [(0, 0.75, 0.75), (30_000, 0.75, 0.75)])  # (750,750) tepi kanan-bawah: di luar B
    only(tr, "AOI2")


@pytest.mark.parametrize("t2", [0, -1000, 500_000, 600_000])
def test_dt_outside_window_is_ignored(t2):
    tr = make()
    feed(tr, [(1_000, 0.3, 0.3), (1_000 + t2, 0.3, 0.3)])
    only(tr)


def test_dt_just_inside_window_counts():
    tr = make()
    feed(tr, [(0, 0.3, 0.3), (490_000, 0.3, 0.3)])
    assert tr.dwell["AOI1"] == pytest.approx(0.49)


def test_rejected_dt_still_advances_previous_timestamp():
    tr = make()
    feed(tr, [(0, 0.3, 0.3), (600_000, 0.3, 0.3), (630_000, 0.3, 0.3)])
    assert tr.dwell["AOI1"] == pytest.approx(0.03)


def test_gaze_outside_screen_counts_nothing_but_time_advances():
    tr = make()
    feed(tr, [(0, 0.3, 0.3), (30_000, 1.2, 0.3), (60_000, 0.3, 0.3)])
    assert tr.dwell["AOI1"] == pytest.approx(0.03)   # bukan 0.06


def test_no_rects_does_not_touch_timestamp():
    tr = make(rects=False)
    feed(tr, [(0, 0.3, 0.3), (30_000, 0.3, 0.3)])
    only(tr)
    tr.set_rects(build_rects(W, H, IMG))
    feed(tr, [(60_000, 0.3, 0.3)])                   # sampel "pertama" lagi
    only(tr)
    feed(tr, [(90_000, 0.3, 0.3)])
    assert tr.dwell["AOI1"] == pytest.approx(0.03)


def test_reset_zeroes_and_forgets_previous_sample():
    tr = make()
    feed(tr, [(0, 0.3, 0.3), (30_000, 0.3, 0.3)])
    tr.reset()
    assert tr.snapshot() == {k: 0.0 for k in AOI_NAMES}
    feed(tr, [(60_000, 0.3, 0.3)])
    only(tr)


def test_snapshot_is_a_copy():
    tr = make()
    feed(tr, [(0, 0.3, 0.3), (30_000, 0.3, 0.3)])
    snap = tr.snapshot()
    snap["AOI1"] = 99
    assert tr.dwell["AOI1"] == pytest.approx(0.03)


def test_set_rects_copies_input():
    tr = make(rects=False)
    rects = build_rects(W, H, IMG)
    tr.set_rects(rects)
    rects.clear()
    feed(tr, [(0, 0.3, 0.3), (30_000, 0.3, 0.3)])
    assert tr.dwell["AOI1"] > 0