"""Sanity tests: limiting cases with a known answer. Run: python sanity.py"""
import numpy as np
from model import Params, POLICIES, run, replace, draw_scenarios, demand_path, regime_path, simulate

N = 150
B = Params()


def check(name, cond, info=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {info}")
    return bool(cond)


def mean(p, key="profit", pols=POLICIES):
    return {k: v[key].mean() for k, v in run(p, n=N, seed=1, policies=pols).items()}


ok = True

# 1. Accounting identities, every month, every policy
sc = draw_scenarios(30, B.T, seed=3)
ident = True
for i in range(30):
    M = demand_path(B, sc["zD"][i], sc["zt0"][i], sc["zm"][i])
    s = regime_path(B, sc["u"][i])
    for pol in POLICIES:
        h = simulate(pol, B, M, s)["hist"]
        ident &= np.allclose(h["U"] + h["C"] + h["lost"], h["W"])
        ident &= np.all(h["W"] <= M + 1e-9) and np.all(h["W"] <= h["A"] + 1e-9)
        ident &= np.all(h["U"] <= np.minimum(h["K"], h["E"]) + 1e-9)
        ident &= np.allclose(h["U"] + h["idle"] + h["dark"], h["K"])
        if pol == "cloud_first":
            ident &= h["K"].max() == 0
ok &= check("1 served+cloud+lost = workload <= min(demand, data); owned split adds up", ident)

# 2. No shortages ever -> shortage gaming changes nothing
p = replace(B, p_ns=0.0, s0=0)
a = mean(p, pols=("hardware_first",))["hardware_first"]
b = mean(replace(p, gaming=False), pols=("hardware_first",))["hardware_first"]
ok &= check("2 no shortage: gaming irrelevant", np.isclose(a, b), f"{a:,.0f} vs {b:,.0f}")

# 3. Readiness never binds (data & power abundant), no shortage -> gated == hardware_first
p = replace(B, A0=1e4, E0=1e4, p_ns=0.0, s0=0)
m = mean(p, pols=("hardware_first", "gated"))
ok &= check("3 readiness slack: gated == hardware_first",
            np.isclose(m["gated"], m["hardware_first"]),
            f"gated={m['gated']:,.0f} hf={m['hardware_first']:,.0f}")

# 4. Value per GPU-hour raises profit, leaves costs unchanged
lo, hi = run(B, n=N, seed=1), run(replace(B, v=12.0), n=N, seed=1)
ok &= check("4 v monotonicity", all(hi[k]["profit"].mean() > lo[k]["profit"].mean()
                                    and np.isclose(hi[k]["total_cost"].mean(), lo[k]["total_cost"].mean())
                                    for k in POLICIES))

# 5. Dearer GPUs hurt owning policies, cloud_first unchanged
hi = run(replace(B, P0=60_000), n=N, seed=1)
ok &= check("5 GPU price monotonicity",
            all(hi[k]["profit"].mean() < lo[k]["profit"].mean() for k in ("hardware_first", "gated"))
            and np.isclose(hi["cloud_first"]["profit"].mean(), lo["cloud_first"]["profit"].mean()))

# 6. Shortage gaming inflates purchases and the glut (the bullwhip mechanism)
p = replace(B, p_ns=0.10, p_sn=0.10)
g = run(p, n=N, seed=1, policies=("hardware_first",))["hardware_first"]
ng = run(replace(p, gaming=False), n=N, seed=1, policies=("hardware_first",))["hardware_first"]
ok &= check("6 gaming -> more GPUs bought, bigger glut",
            g["gpus_bought"].mean() > ng["gpus_bought"].mean() and g["peak_excess"].mean() > ng["peak_excess"].mean(),
            f"bought {g['gpus_bought'].mean():.0f} vs {ng['gpus_bought'].mean():.0f}")

# 7. Cloud never constrained and very cheap -> cloud_first most profitable
p = replace(B, c_cloud=0.3, m_cloud=0.0, cloud_cap_short=1e3)
m = mean(p)
ok &= check("7 cheap unlimited cloud: cloud_first wins", max(m, key=m.get) == "cloud_first",
            str({k: round(v / 1e6, 2) for k, v in m.items()}))

# 8. Zero lead times work (orders delivered same month)
p = replace(B, L_normal=0, L_short=0, L_E=0, p_ns=0.0)
r = run(p, n=20, seed=1, policies=("gated",))["gated"]
ok &= check("8 zero lead time: gated still buys and serves", r["gpus_bought"].mean() > 0
            and r["served_share"].mean() > 0.9, f"served={r['served_share'].mean():.2f}")

print("\nALL PASS" if ok else "\nSOME FAILED")
