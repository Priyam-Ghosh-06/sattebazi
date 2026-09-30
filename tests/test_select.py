"""Run: python tests/test_select.py"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from shared.optimization.selection import BOUNDS, select_xi  # noqa: E402


def test_constraints_and_determinism():
    rng = np.random.default_rng(0)
    fallbacks = 0
    for _ in range(1000):
        n = rng.integers(22, 36)
        role = rng.choice(["WK", "BAT", "AR", "BOWL", "UNK"], n, p=[.12, .3, .2, .34, .04])
        team = np.where(np.arange(n) < n // 2, "A", "B")
        score = rng.normal(40, 20, n).round()  # ties on purpose
        pid = rng.permutation(n)
        idx, ok = select_xi(score, role, team, pid)
        assert len(idx) == 11 and len(set(team[idx])) == 2
        if ok:
            r = np.where(role[idx] == "UNK", "BAT", role[idx])
            for k, (a, b) in BOUNDS.items():
                assert a <= (r == k).sum() <= b
        else:
            fallbacks += 1
        assert (select_xi(score, role, team, pid)[0] == idx).all()
    print("fallbacks", fallbacks)


if __name__ == "__main__":
    test_constraints_and_determinism(); print("ok")
