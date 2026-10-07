"""
Deployment gap model v2: GPUs vs data and power readiness, with bullwhip effects.

One firm turns a noisy, S-shaped AI demand signal M_t into useful compute over T months.
Useful compute needs three things at once (Leontief):

    workload W_t  = min(M_t, A_t)          data readiness A caps what can actually run
    running GPUs  = min(K_t, E_t)          power E caps owned GPUs that can be switched on
    owned served  U_t = min(running, W_t)
    cloud served  C_t = min(W_t - U_t, cloud cap)     (cloud needs data readiness, not our power)

All quantities are in H100-equivalents (GPU-eq). Each useful GPU-hour is worth v.
The firm maximises NPV profit = value of useful compute - all costs.

Supply side (where bullwhip enters):
  * GPU market switches normal <-> shortage (Markov). In shortage: longer lead time,
    price premium, and the vendor fills only a fraction `fill` of each order on time;
    the rest is backlogged and delivered after the shortage ends.
  * Power is ordered in blocks with a long lead time (grid / facility work).
  * Data readiness grows at most `a_rate` GPU-eq per month (data-team throughput).

Policies:
  hardware_first - order GPUs to the extrapolated demand forecast; in a shortage, inflate
                   orders by 1/fill (shortage gaming); build power and data reactively,
                   once GPUs or demand are already there. The "buy GPUs, sort the rest later" firm.
  buy_ahead      - disciplined hardware_first: no gaming, data and power built ahead like gated,
                   but GPUs still ordered to the full forecast (no readiness cap). The fair
                   comparison for gating alone.
  gated          - build data and power ahead to the forecast, then order GPUs only up to
                   min(forecast, data readiness reachable by arrival, power arriving by arrival).
                   No gaming. Deployment-gated acquisition.
  cloud_first    - build data readiness like gated, own no GPUs or power, use cloud.

All parameter defaults are PLACEHOLDERS (see README).
"""
from dataclasses import dataclass, replace
import math
import numpy as np

POLICIES = ("hardware_first", "buy_ahead", "gated", "cloud_first")


@dataclass(frozen=True)
class Params:
    # --- horizon & finance ---
    T: int = 48                  # months
    hours: float = 730.0         # hours per month
    r_annual: float = 0.10       # discount rate

    # --- demand signal M_t (business demand for AI compute) ---
    D_max: float = 256.0         # planned peak, GPU-eq (32 eight-GPU nodes)
    k: float = 0.30              # S-curve steepness (1/month)
    t0: float = 14.0             # S-curve midpoint (month)
    sigma_D: float = 0.4         # lognormal sd of final scale
    sigma_t0: float = 4.0        # sd of midpoint (months)
    sigma_month: float = 0.10    # month-to-month noise (lognormal sd)
    v: float = 6.0               # value per useful GPU-hour, USD

    # --- forecasting (Holt linear trend; the bullwhip "demand signal processing" cause) ---
    holt_a: float = 0.4          # level smoothing
    holt_b: float = 0.3          # trend smoothing

    # --- data readiness A ---
    A0: float = 16.0             # initial ready workload, GPU-eq
    a_rate: float = 16.0         # max data readiness added per month, GPU-eq (16 months to D_max)
    c_A: float = 15_000.0        # cost per GPU-eq of data readiness, USD

    # --- power readiness E (on-prem only) ---
    E0: float = 32.0             # existing powered capacity, GPU-eq
    L_E: int = 12                # power lead time, months
    E_block: int = 64            # power is added in blocks of this many GPU-eq (~100 kW)
    c_E: float = 15_000.0        # cost per GPU-eq of power (~1.5 kW incl. cooling), USD
    E_life_months: int = 180     # power infrastructure life, for salvage

    # --- GPU supply ---
    P0: float = 30_000.0         # price per GPU-eq installed, t=0
    g_annual: float = 0.28       # price decline per GPU-eq
    L_normal: int = 3            # lead time, normal
    L_short: int = 10            # lead time, shortage (8-11 months reported for H100, 2023; TrendForce)
    p_ns: float = 0.04           # monthly P(normal -> shortage)
    p_sn: float = 0.08           # monthly P(shortage -> normal): ~12-15 month spell, 2023 to early 2024
    m: float = 0.45              # GPU price premium in shortage (resale >USD 40k vs 25-30k list; SiliconANGLE)
    fill: float = 0.5            # share of an order filled on time in shortage
    s0: int = 0                  # initial regime
    life_months: int = 60        # GPU life, for salvage
    tranche: int = 8             # order granularity (one node)
    # hardware_first behaviour switches (used to decompose hardware_first -> gated)
    gaming: bool = True          # inflate orders by 1/fill in shortage, ignore backlog
    hf_power_ahead: bool = False # build power ahead to forecast (as gated does)
    hf_data_ahead: bool = False  # build data ahead to forecast (as gated does)

    # --- operations & energy ---
    W_act: float = 1.0           # kW per GPU-eq at load
    W_idle: float = 0.60         # kW per GPU-eq powered but idle (BLOOM cluster: idle 64 vs dynamic 109 kWh)
    embodied_kg: float = 460.0   # kgCO2e per GPU-eq: (2500 server + 8*150 GPU)/8 (Luccioni et al. 2023)
    cloud_util: float = 0.85     # average utilisation of cloud GPUs, for amortising their embodied carbon
    PUE: float = 1.67            # worldwide average PUE (Lannelongue et al. 2021)
    e_price: float = 0.10        # USD/kWh
    CI: float = 0.475            # kgCO2/kWh, worldwide average (Lannelongue et al. 2021); range 0.02-0.8+
    opex_month: float = 150.0    # USD per owned GPU-eq-month

    # --- cloud ---
    c_cloud: float = 2.5         # USD per GPU-eq-hour, normal
    m_cloud: float = 1.5         # cloud premium in shortage (~USD 8/h peak vs ~2-3/h normal)
    cloud_cap_short: float = 0.25  # cloud available in shortage, as share of D_max
    PUE_cloud: float = 1.10      # hyperscale PUE (Patterson et al. 2022 report ~1.1 for Google) - check
    CI_cloud: float = 0.475      # same grid as on-prem, so only PUE and utilisation differ


def logistic(t, k, t0):
    return 1.0 / (1.0 + np.exp(-k * (np.asarray(t, float) - t0)))


# ----------------------------------------------------------------------
# Scenarios (common random numbers across policies)
# ----------------------------------------------------------------------
def draw_scenarios(n, T, seed=0):
    rng = np.random.default_rng(seed)
    return {"zD": rng.standard_normal(n), "zt0": rng.standard_normal(n),
            "zm": rng.standard_normal((n, T)), "u": rng.random((n, T))}


def demand_path(p, zD, zt0, zm):
    t = np.arange(p.T)
    scale = p.D_max * math.exp(p.sigma_D * zD - 0.5 * p.sigma_D ** 2)
    base = scale * logistic(t, p.k, p.t0 + p.sigma_t0 * zt0)
    return base * np.exp(p.sigma_month * zm - 0.5 * p.sigma_month ** 2)


def regime_path(p, u):
    s = np.empty(p.T, dtype=int)
    s[0] = p.s0
    for t in range(1, p.T):
        if s[t - 1] == 0:
            s[t] = 1 if u[t] < p.p_ns else 0
        else:
            s[t] = 0 if u[t] < p.p_sn else 1
    return s


def holt(p, M):
    """level[t], trend[t] after observing M[0..t]. Forecast made at month t (seeing M[<t])
    for month t+h is level[t-1] + (h+1) * trend[t-1]."""
    lvl, tr = np.empty(p.T), np.empty(p.T)
    lvl[0], tr[0] = M[0], 0.0
    for t in range(1, p.T):
        lvl[t] = p.holt_a * M[t] + (1 - p.holt_a) * (lvl[t - 1] + tr[t - 1])
        tr[t] = p.holt_b * (lvl[t] - lvl[t - 1]) + (1 - p.holt_b) * tr[t - 1]
    return lvl, tr


def _up(x, step):
    return int(math.ceil(x / step - 1e-9)) * step if x > 0 else 0


# ----------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------
def simulate(policy, p, M, s):
    if policy == "buy_ahead":
        p = replace(p, gaming=False, hf_power_ahead=True, hf_data_ahead=True)
        policy = "hardware_first"
    T, h = p.T, p.hours
    r_m = (1 + p.r_annual) ** (1 / 12) - 1
    disc = (1 + r_m) ** -np.arange(T + 1)
    P_norm = p.P0 * np.exp(-p.g_annual / 12 * np.arange(T + 1))
    lvl, tr = holt(p, M)

    def fc(t, horizon):                       # forecast at start of t for month t+horizon
        if t == 0:
            return M[0]
        return max(0.0, lvl[t - 1] + (horizon + 1) * tr[t - 1])

    own = policy != "cloud_first"
    K, A, E = 0.0, p.A0, (p.E0 if own else 0.0)
    gpu_arr = np.zeros(T + 64)                # scheduled GPU arrivals
    gpu_price_arr = np.zeros(T + 64)          # money owed on those arrivals (paid on delivery)
    backlog_q, backlog_cost = 0.0, 0.0        # filled after shortage ends
    pow_arr = np.zeros(T + 64)
    delivered = []                            # (order month, qty) for salvage

    cost = {"gpu": 0.0, "power": 0.0, "data": 0.0, "energy": 0.0, "opex": 0.0, "cloud": 0.0}
    value = co2 = co2_emb = co2_waste = 0.0
    emb_month = p.embodied_kg / p.life_months          # embodied kg per GPU-eq-month owned
    orders = np.zeros(T)
    hist = {k: np.zeros(T) for k in ("K", "A", "E", "W", "U", "C", "lost", "dark", "idle")}

    for t in range(T):
        # --- arrivals ---
        if t > 0 and s[t - 1] == 1 and s[t] == 0 and backlog_q > 0:   # shortage just ended
            gpu_arr[t + p.L_normal] += backlog_q
            gpu_price_arr[t + p.L_normal] += backlog_cost
            backlog_q = backlog_cost = 0.0
        if gpu_arr[t] > 0:
            K += gpu_arr[t]
            cost["gpu"] += disc[t] * gpu_price_arr[t]
            delivered.append((t, gpu_arr[t]))
        E += pow_arr[t]

        L = p.L_short if s[t] else p.L_normal
        fill = p.fill if s[t] else 1.0
        sched = gpu_arr[t + 1:].sum()                  # on-time deliveries still to come
        pow_pipe = pow_arr[t + 1:].sum()

        # --- data readiness (effective next month) ---
        if policy == "hardware_first" and not p.hf_data_ahead:
            dA_target = M[t - 1] if t > 0 else M[0]                    # reactive
        else:
            dA_target = fc(t, p.L_normal + 1)                          # ahead of GPU arrival
        dA = min(p.a_rate, max(0.0, dA_target - A))

        # --- power and GPU orders ---
        q = 0
        if own:
            if policy == "hardware_first":
                target = fc(t, L)
                pos = K + sched + (0.0 if p.gaming else backlog_q)     # gamers ignore backlog
                gap = target - pos
                if gap > 0:
                    q = _up(gap / fill if (p.gaming and s[t]) else gap, p.tranche)
                if p.hf_power_ahead:
                    p_need = min(fc(t, p.L_E), A + p.a_rate * p.L_E) - (E + pow_pipe)
                else:  # reactive: cover GPUs owned or expected, once they are bought
                    p_need = K + sched + q * fill - (E + pow_pipe)
            else:  # gated
                tgt_E = min(fc(t, p.L_E), A + p.a_rate * p.L_E)
                p_need = tgt_E - (E + pow_pipe)
                E_by = E + pow_arr[t + 1:t + L + 1].sum()
                target = min(fc(t, L), A + dA + p.a_rate * (L - 1), E_by)
                gap = target - (K + sched + backlog_q)
                q = _up(gap, p.tranche)
            if p_need > 0 and t + p.L_E < T:
                qe = _up(p_need, p.E_block)
                if p.L_E == 0:
                    E += qe
                else:
                    pow_arr[t + p.L_E] += qe
                cost["power"] += disc[t] * p.c_E * qe
            if q > 0 and t + L < T:
                price = P_norm[t] * (1 + p.m * s[t])
                on_time = q * fill
                if L == 0:                                   # same-month delivery
                    K += on_time
                    cost["gpu"] += disc[t] * on_time * price
                    delivered.append((t, on_time))
                else:
                    gpu_arr[t + L] += on_time
                    gpu_price_arr[t + L] += on_time * price
                backlog_q += q - on_time
                backlog_cost += (q - on_time) * price
                orders[t] = q

        # --- operate ---
        W = min(M[t], A)
        run_cap = min(K, E)
        U = min(run_cap, W)
        cap = p.cloud_cap_short * p.D_max if s[t] else np.inf
        C = min(W - U, cap)
        lost = W - U - C
        e_own = h * p.PUE * (U * p.W_act + (run_cap - U) * p.W_idle)
        e_cld = h * p.PUE_cloud * C * p.W_act
        cost["energy"] += disc[t] * p.e_price * e_own
        cost["opex"] += disc[t] * p.opex_month * K
        cost["cloud"] += disc[t] * p.c_cloud * (1 + p.m_cloud * s[t]) * h * C
        co2 += p.CI * e_own + p.CI_cloud * e_cld
        # embodied carbon, amortised over life: owned GPU-months + cloud GPU-hours at cloud_util
        co2_emb += emb_month * K + C * p.embodied_kg / (p.life_months * p.cloud_util)
        # wasted carbon: embodied share of unused owned GPUs + energy of powered-but-idle GPUs
        co2_waste += emb_month * (K - U) + p.CI * h * p.PUE * (run_cap - U) * p.W_idle
        value += disc[t] * p.v * h * (U + C)
        for k_, x in (("K", K), ("A", A), ("E", E), ("W", W), ("U", U), ("C", C), ("lost", lost),
                      ("dark", K - run_cap), ("idle", run_cap - U)):
            hist[k_][t] = x

        # data readiness spend lands next month
        cost["data"] += disc[t] * p.c_A * dA
        A += dA

    gpu_salv = sum(q_ * P_norm[T] * max(0.0, 1 - (T - t_) / p.life_months) for t_, q_ in delivered)
    pow_salv = (E - (p.E0 if own else 0.0)) * p.c_E * max(0.0, 1 - T / p.E_life_months)
    salvage = disc[T] * (gpu_salv + pow_salv)
    total_cost = sum(cost.values()) - salvage
    useful_h = h * (hist["U"] + hist["C"]).sum()

    # capacity bullwhip ratio, quarterly: Var(GPU-eq ordered per quarter) /
    # Var(change in demand signal per quarter), over quarters 1.. before the end cutoff
    nq = (T - p.L_short) // 3
    oq = orders[3:3 * nq].reshape(-1, 3).sum(axis=1)
    dMq = M[3:3 * nq:3] - M[0:3 * nq - 3:3]
    bw = oq.var() / dMq.var() if own and dMq.var() > 0 else np.nan

    Ksum = hist["K"].sum()
    return {
        "profit": value - total_cost, "value": value, "total_cost": total_cost,
        **{f"c_{k}": v for k, v in cost.items()}, "salvage_pv": salvage,
        "cost_per_useful_gpuh": total_cost / useful_h if useful_h > 0 else np.nan,
        "served_share": (hist["U"] + hist["C"]).sum() / M.sum(),
        "readiness_gap_share": (M - hist["W"]).sum() / M.sum(),     # demand not data-ready
        "lost_share": hist["lost"].sum() / M.sum(),                 # ready but no compute
        "util": hist["U"].sum() / Ksum if Ksum > 0 else np.nan,
        "dark_share": hist["dark"].sum() / Ksum if Ksum > 0 else np.nan,   # owned, no power
        "idle_share": hist["idle"].sum() / Ksum if Ksum > 0 else np.nan,   # powered, no workload
        "bullwhip": bw,
        "peak_excess": float((hist["K"] - hist["W"]).max()) if own else 0.0,
        "gpus_bought": float(sum(q_ for _, q_ in delivered)),
        "co2_t": (co2 + co2_emb) / 1000,
        "co2_op_t": co2 / 1000, "co2_emb_t": co2_emb / 1000, "co2_waste_t": co2_waste / 1000,
        "co2_per_useful_gpuh": (co2 + co2_emb) / useful_h if useful_h > 0 else np.nan,
        "hist": hist, "orders": orders,
    }


SCALAR_KEYS = ("profit", "value", "total_cost", "c_gpu", "c_power", "c_data", "c_energy",
               "c_opex", "c_cloud", "salvage_pv", "cost_per_useful_gpuh", "served_share",
               "readiness_gap_share", "lost_share", "util", "dark_share", "idle_share",
               "bullwhip", "peak_excess", "gpus_bought", "co2_t", "co2_op_t", "co2_emb_t",
               "co2_waste_t", "co2_per_useful_gpuh")


def run(p, n=200, seed=0, policies=POLICIES):
    sc = draw_scenarios(n, p.T, seed)
    out = {pol: {k: np.empty(n) for k in SCALAR_KEYS} for pol in policies}
    for i in range(n):
        M = demand_path(p, sc["zD"][i], sc["zt0"][i], sc["zm"][i])
        s = regime_path(p, sc["u"][i])
        for pol in policies:
            r = simulate(pol, p, M, s)
            for k in SCALAR_KEYS:
                out[pol][k][i] = r[k]
    return out


__all__ = ["Params", "POLICIES", "run", "simulate", "demand_path", "regime_path",
           "draw_scenarios", "holt", "replace"]
