"""E3 formal test: is pooled F0 across the chapter two speakers?

Pure-stdlib 1-D EM (2 Gaussians) vs 1 Gaussian on log-F0, BIC comparison.
Log domain because pitch is perceived logarithmically (SCIENCE.md 1.5).
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import miniaudio  # audio-venv only


def window_f0(t0, dur=90):
    dec = miniaudio.decode_file(str(ROOT / "examples/realaudio/alice01.mp3"),
                                nchannels=1, sample_rate=16000,
                                output_format=miniaudio.SampleFormat.FLOAT32)
    import numpy as np
    a = np.frombuffer(dec.samples, dtype=np.float32)[t0*16000:(t0+dur)*16000]
    from src.pitch import PitchTracker
    return [f.hz for f in PitchTracker(16000).track(a.tolist())
            if f.hz is not None and 75 < f.hz < 395]


def gauss_ll(xs, mu, var):
    if var <= 0:
        return -1e18
    c = -0.5 * (math.log(2*math.pi*var))
    return sum(c - (x-mu)**2/(2*var) for x in xs)


def em2(xs, iters=200):
    mu1, mu2 = min(xs), max(xs)
    p = 0.5; var = var1 = var2 = (max(xs)-min(xs))**2/4
    for _ in range(iters):
        r1 = [p*math.exp(gauss_ll([x], mu1, var1)) for x in xs]
        r2 = [(1-p)*math.exp(gauss_ll([x], mu2, var2)) for x in xs]
        g = [a/(a+b) if (a+b) > 0 else 0.5 for a, b in zip(r1, r2)]
        n = len(xs)
        p = sum(g)/n
        mu1 = sum(a*x for a, x in zip(g, xs))/max(sum(g), 1e-9)
        mu2 = sum((1-a)*x for a, x in zip(g, xs))/max(n-sum(g), 1e-9)
        var1 = sum(a*(x-mu1)**2 for a, x in zip(g, xs))/max(sum(g), 1e-9) or 1e-6
        var2 = sum((1-a)*(x-mu2)**2 for a, x in zip(g, xs))/max(n-sum(g), 1e-9) or 1e-6
    ll = sum(math.log(max(p*math.exp(gauss_ll([x], mu1, var1))
                        + (1-p)*math.exp(gauss_ll([x], mu2, var2)), 1e-300))
             for x in xs)
    return ll, p, (math.exp(mu1), math.sqrt(var1)), (math.exp(mu2), math.sqrt(var2))


def bic(ll, k, n):
    return k*math.log(n) - 2*ll


def main():
    xs = []
    for t0 in (20, 600, 1200, 1800, 2200):
        hz = window_f0(t0)
        xs += [math.log(h) for h in hz]
        print(f"t={t0}s n={len(hz)}")
    n = len(xs)
    mu = sum(xs)/n
    var = sum((x-mu)**2 for x in xs)/n
    ll1 = gauss_ll(xs, mu, var)
    ll2, p, c1, c2 = em2(xs)
    print(f"\nn={n}")
    print(f"1-Gaussian: ll={ll1:.0f} BIC={bic(ll1,2,n):.0f}")
    print(f"2-Gaussian: ll={ll2:.0f} BIC={bic(ll2,5,n):.0f}")
    d = bic(ll1, 2, n) - bic(ll2, 5, n)
    print(f"BIC difference (positive favors 2): {d:.0f}")
    print(f"components: {c1[0]:.0f}Hz +/-{c1[1]:.0f} (share {p:.2f})  "
          f"{c2[0]:.0f}Hz +/-{c2[1]:.0f} (share {1-p:.2f})")


if __name__ == "__main__":
    main()
