import pytest
from scoring import (
    estimate_cognitive_age,
    average_cognitive_age,
    cognitive_fitness,
)


# Nilai acuan diambil dari fungsi lama (TimeIn.estimate_cognitive_age).
@pytest.mark.parametrize("seconds, expected", [
    (0, 20),     # di bawah batas -> clamp ke 20
    (5, 20),     # prediksi < 20 -> clamp ke 20
    (9, 25),
    (12, 32),
    (20, 50),
    (33, 71),
    (50, 81),
    (60, 78),    # perilaku asli: setelah ~51 dtk kurva turun
    (100, 20),   # perilaku asli: >= 95 dtk jatuh ke batas bawah
])
def test_estimate_cognitive_age(seconds, expected):
    assert estimate_cognitive_age(seconds) == expected


def test_estimate_returns_int():
    assert isinstance(estimate_cognitive_age(20), int)


def test_estimate_never_outside_clamp():
    for s in range(0, 300):
        assert 20 <= estimate_cognitive_age(s) <= 90


@pytest.mark.parametrize("ages, expected", [
    ([30, 41], 35),      # 35.5 dipotong, bukan dibulatkan
    ([30, 31], 30),      # 30.5 -> 30
    ([50], 50),
    ([20, 20, 21], 20),  # 20.33 -> 20
])
def test_average_cognitive_age_truncates(ages, expected):
    assert average_cognitive_age(ages) == expected


@pytest.mark.parametrize("age_cog, age_real, expected", [
    (40, 40, 100),   # sama -> 100
    (30, 40, 100),   # lebih muda -> 100
    (41, 40, 99),
    (90, 20, 30),
])
def test_cognitive_fitness(age_cog, age_real, expected):
    assert cognitive_fitness(age_cog, age_real) == expected