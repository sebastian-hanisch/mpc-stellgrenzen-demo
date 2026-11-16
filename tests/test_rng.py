"""Der Zufallsgenerator der Demo (Kopie der Vorlage): bitgleich auf jeder Plattform; erste Werte des SplitMix64 mit Zufalls-Seed 0 sind festgelegt."""

from msg_rng import SplitMix64


def test_known_first_outputs_of_splitmix64_seed_zero():
    """Referenzwerte des SplitMix64 (Vigna) bei Zufalls-Seed 0: 0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F."""
    r = SplitMix64(0)
    assert [r.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_random_is_in_unit_interval_and_reproducible():
    a, b = SplitMix64(14), SplitMix64(14)
    xs = [a.random() for _ in range(200)]
    assert xs == [b.random() for _ in range(200)] and all(0.0 <= x < 1.0 for x in xs)
    assert 0.35 < sum(xs) / len(xs) < 0.65


def test_below_stays_in_range():
    r = SplitMix64(5)
    assert {r.below(3) for _ in range(100)} == {0, 1, 2}
