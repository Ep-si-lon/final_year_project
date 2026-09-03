import numpy as np

from uav_sim.src.params import CrazyflieParams, compute_gammas


def test_gammas_match_hand_computed_values():
    p = CrazyflieParams()
    denom = p.Jxx * p.Jzz - p.Jxz ** 2

    expected = dict(
        g1=p.Jxz * (p.Jxx - p.Jyy + p.Jzz) / denom,
        g2=(p.Jzz * (p.Jzz - p.Jyy) + p.Jxz ** 2) / denom,
        g3=p.Jzz / denom,
        g4=p.Jxz / denom,
        g5=(p.Jzz - p.Jxx) / p.Jyy,
        g6=p.Jxz / p.Jyy,
        g7=((p.Jxx - p.Jyy) * p.Jxx + p.Jxz ** 2) / denom,
        g8=p.Jxx / denom,
    )

    gammas = compute_gammas(p)
    for name, value in expected.items():
        assert np.isclose(getattr(gammas, name), value), name
