"""
Wang et al. (2010) airway smooth muscle Ca2+ model extended with the acute
(non-genomic) actions of 17-beta oestradiol, following the three-layer scheme
described in Group 4 Doc (Sections 6 to 9).

Layer 1: Phenomenological Hill-type parameter modulation of g_Ca, V_e (SERCA),
         V_p (PMCA) and the force gain alpha.
Layer 2: Receptor-resolved drive (ER-alpha, ER-beta, GPER) replacing the single
         oestradiol input with weighted occupancies.
Layer 3: Mechanistic NO/cGMP/PKG cascade. The downstream parameter modifiers
         f_X are then driven by [PKG] rather than by E directly.

Optional: explicit myosin light chain (MLC) phosphorylation submodel for force
generation, replacing the algebraic Hill-type force law.

Each layer is independently toggleable via params["estrogen_layer"] in
{0, 1, 2, 3} and params["use_mlc_force"] in {True, False}.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# -----------------------------
# Parameters
# -----------------------------
params = {
    # ---------- Calcium flux parameters (Wang et al. 2010) ----------
    "k_PMCA": 0.4,
    "V_Pmax": 4.5,
    "V_s": 4.5,
    "k_s": 0.1,
    "k_leak": 0.1,
    "F": 96485.3329,

    "P": 0.35,            # IP3 concentration proxy (mM)

    "gamma": 5.5,
    "delta": 0.05,

    "g_Ca": 9.0,
    "V_m": -50.0,
    "k_m": 12.0,
    "R": 8314.0,
    "T": 310.0,

    "a_0": 0.05,
    "a_1": 0.25,
    "a_2": 1.0,

    "k_1": 2000, "k_-1": 260,  "K_1": 0.13,
    "k_2": 1,    "k_-2": 1.05, "K_2": 1.05,
    "k_3": 2000, "k_-3": 1886, "K_3": 0.943,
    "k_4": 1,    "k_-4": 0.145,"K_4": 0.145,
    "k_5": 100,  "k_-5": 8.2,  "K_5": 0.082,

    # Channel gains
    "k_IP3R": 5.55,
    "k_RyR": 5.0,
    "k_ryr0": 0.0072,
    "k_ryr1": 0.334,
    "k_ryr2": 0.5,
    "k_ryr3": 38.0,

    # Voltage parameters
    "c_m": 1.0,
    "I_stim": 0.1175,

    # Algebraic contraction (Lata, uterine)
    "alpha": 3.0,
    "beta": 0.001,
    "n_F": 4,

    # Functional parameters
    "n": 4,
    "ns": 2,
    "n2": 3,

    # ============================================================
    # Oestrogen extension
    # ============================================================
    "estrogen_layer": 1,        # 0 = off, 1 = phenomenological, 2 = receptor, 3 = NO/cGMP/PKG
    "use_mlc_force": False,     # True = explicit MLC phosphorylation force law

    # Oestradiol concentration (mM). Replace with a callable for time-varying E(t).
    "E": 1e-6,                  # ~1 nM, within the physiological range of 17-beta-oestradiol

    # ---------- Layer 1: Hill-type modifiers ----------
    # f_X(E) = 1 + (s_X - 1) E^h / (EC50^h + E^h)
    "s_VOCC":  0.7,  "EC50_VOCC":  1e-6, "h_VOCC":  1.0,   # inhibition of L-type Ca current
    "s_SERCA": 1.4,  "EC50_SERCA": 1e-6, "h_SERCA": 1.0,
    "s_PMCA":  1.3,  "EC50_PMCA":  1e-6, "h_PMCA":  1.0,
    "s_alpha": 0.6,  "EC50_alpha": 1e-6, "h_alpha": 1.0,   # reduced force gain (MLCP activation)

    # ---------- Layer 2: Receptor-resolved drive ----------
    # Equilibrium dissociation constants (mM). Order-of-magnitude placeholders.
    "Kd_ERa":  1e-6,
    "Kd_ERb":  1e-6,
    "Kd_GPER": 5e-6,

    # Receptor weights w_{X,i} for each modulated process X and receptor i
    # (rows = process, columns = [alpha, beta, GPER])
    "w_VOCC":  (0.5, 0.2, 0.5),
    "w_SERCA": (0.6, 0.2, 0.4),
    "w_PMCA":  (0.6, 0.2, 0.4),
    "w_alpha": (0.7, 0.1, 0.4),

    # Per-receptor saturated effects s_{X,i}
    "s_VOCC_i":  (0.7, 1.05, 0.75),
    "s_SERCA_i": (1.4, 0.95, 1.25),
    "s_PMCA_i":  (1.3, 0.95, 1.20),
    "s_alpha_i": (0.55, 1.05, 0.65),

    # ---------- Layer 3: NO/cGMP/PKG cascade ----------
    "k_eNOS":     1.0e-3,    # NO production rate per receptor occupancy (mM s^-1)
    "omega_GPER": 0.5,       # relative GPER contribution to eNOS activation
    "k_NO_deg":   1.0,       # NO degradation rate (s^-1)
    "V_sGC":      5.0e-3,    # max sGC activity (mM s^-1)
    "K_sGC":      1.0e-4,    # sGC half-activation [NO] (mM)
    "n_sGC":      1.5,
    "k_PDE":      0.5,       # cGMP degradation (s^-1)
    "K_PKG":      1.0e-3,    # cGMP for half-maximal PKG activity (mM)

    # Saturating fractional shifts (Layer 3)
    "a_VOCC":  0.3,
    "a_PLB":   0.4,          # SERCA potentiation through phospholamban
    "a_PMCA":  0.3,
    "a_MLCP":  1.0,          # MLCP activation by PKG (used in MLC submodel)

    # ---------- MLC phosphorylation submodel ----------
    "k_MLCK_max": 4.0,
    "K_M":        0.3e-3,    # half-activation [Ca] for MLCK (mM)
    "n_M":        4,
    "k_MLCP_0":   0.5,
    "F_max":      1.0,
}


# =============================================================================
# Oestrogen modulation
# =============================================================================

def E_value(t, p):
    """Return oestradiol concentration at time t. Allows callable for E(t)."""
    E = p["E"]
    return E(t) if callable(E) else E


def hill_modifier(E, s, EC50, h):
    """Standard sigmoidal pharmacological dose-response modifier."""
    if E <= 0.0:
        return 1.0
    return 1.0 + (s - 1.0) * (E**h) / (EC50**h + E**h)


def receptor_occupancies(E, p):
    """Equilibrium fractional occupancy of ER-alpha, ER-beta, GPER."""
    if E <= 0.0:
        return 0.0, 0.0, 0.0
    R_a    = E / (E + p["Kd_ERa"])
    R_b    = E / (E + p["Kd_ERb"])
    R_gper = E / (E + p["Kd_GPER"])
    return R_a, R_b, R_gper


def receptor_modifier(R_a, R_b, R_gper, w, s_i):
    """Layer 2: f_X = 1 + sum_i w_{X,i} (s_{X,i} - 1) R_i."""
    return (
        1.0
        + w[0] * (s_i[0] - 1.0) * R_a
        + w[1] * (s_i[1] - 1.0) * R_b
        + w[2] * (s_i[2] - 1.0) * R_gper
    )


def pkg_modifiers(PKG, p):
    """Layer 3: parameter shifts driven by PKG activity."""
    f_VOCC  = 1.0 - p["a_VOCC"]  * PKG
    f_SERCA = 1.0 + p["a_PLB"]   * PKG
    f_PMCA  = 1.0 + p["a_PMCA"]  * PKG
    return f_VOCC, f_SERCA, f_PMCA


def estrogen_factors(t, p, PKG=0.0):
    """Return modulation factors (f_VOCC, f_SERCA, f_PMCA, f_alpha) for the
    selected layer. f_alpha is only used when the algebraic force law is active.
    """
    layer = p["estrogen_layer"]

    if layer == 0:
        return 1.0, 1.0, 1.0, 1.0

    E = E_value(t, p)

    if layer == 1:
        f_VOCC  = hill_modifier(E, p["s_VOCC"],  p["EC50_VOCC"],  p["h_VOCC"])
        f_SERCA = hill_modifier(E, p["s_SERCA"], p["EC50_SERCA"], p["h_SERCA"])
        f_PMCA  = hill_modifier(E, p["s_PMCA"],  p["EC50_PMCA"],  p["h_PMCA"])
        f_alpha = hill_modifier(E, p["s_alpha"], p["EC50_alpha"], p["h_alpha"])
        return f_VOCC, f_SERCA, f_PMCA, f_alpha

    R_a, R_b, R_gper = receptor_occupancies(E, p)

    if layer == 2:
        f_VOCC  = receptor_modifier(R_a, R_b, R_gper, p["w_VOCC"],  p["s_VOCC_i"])
        f_SERCA = receptor_modifier(R_a, R_b, R_gper, p["w_SERCA"], p["s_SERCA_i"])
        f_PMCA  = receptor_modifier(R_a, R_b, R_gper, p["w_PMCA"],  p["s_PMCA_i"])
        f_alpha = receptor_modifier(R_a, R_b, R_gper, p["w_alpha"], p["s_alpha_i"])
        return f_VOCC, f_SERCA, f_PMCA, f_alpha

    if layer == 3:
        f_VOCC, f_SERCA, f_PMCA = pkg_modifiers(PKG, p)
        # Algebraic force law gain is left at 1 in Layer 3; force modulation
        # is delivered through the MLC phosphorylation submodel (Eq. 30).
        return f_VOCC, f_SERCA, f_PMCA, 1.0

    raise ValueError(f"Unknown estrogen_layer: {layer}")


# =============================================================================
# Original Wang fluxes, with optional oestrogen scaling
# =============================================================================

def m_inf(V, p):
    return 1.0 / (1.0 + np.exp(-(V - p["V_m"]) / p["k_m"]))


def V_Ca(V, Ca_in, Ca_0, p):
    F, R, T = p["F"], p["R"], p["T"]
    exp_term = np.exp(-2.0 * V * F / (R * T))
    denom = 1.0 - exp_term
    if np.abs(denom) < 1e-8:
        return 1e-8
    return V * (Ca_in - Ca_0 * exp_term) / denom


def I_Ca(V, Ca_in, Ca_0, p, f_VOCC=1.0):
    """L-type Ca current, scaled through g_Ca by the oestrogen modifier."""
    m = m_inf(V, p)
    Vca = V_Ca(V, Ca_in, Ca_0, p)
    g_Ca_eff = p["g_Ca"] * f_VOCC
    return g_Ca_eff * (m**2) * Vca


def J_in_Wang(V, Ca_in, Ca_0, p, f_VOCC=1.0):
    return (
        p["a_0"]
        - p["a_1"] * I_Ca(V, Ca_in, Ca_0, p, f_VOCC) / (2.0 * p["F"])
        + p["a_2"] * p["P"]
    )


def J_PMCA_Hill(Ca_in, p, f_PMCA=1.0):
    V_Pmax_eff = p["V_Pmax"] * f_PMCA
    return V_Pmax_eff * (Ca_in**p["n"]) / (p["k_PMCA"]**p["n"] + Ca_in**p["n"])


def J_SERCA_Hill(Ca_in, p, f_SERCA=1.0):
    V_s_eff = p["V_s"] * f_SERCA
    return V_s_eff * (Ca_in**p["ns"]) / (p["k_s"]**p["ns"] + Ca_in**p["ns"])


def J_leak(Ca_SR, Ca_in, p):
    return p["k_leak"] * (Ca_SR - Ca_in)


def P_IP3R(Ca_in, y, p):
    num = p["P"] * Ca_in * (1.0 - y)
    den = (p["P"] + p["K_1"]) * (Ca_in + p["K_5"])
    return (num / den) ** 3


def J_IP3R_Wang(Ca_SR, Ca_in, y_g, p):
    return p["k_IP3R"] * P_IP3R(Ca_in, y_g, p) * (Ca_SR - Ca_in)


def dy_dt(y, p, Ca_in):
    f1 = (
        (p["k_-4"] * p["K_2"] * p["K_1"] + p["k_-2"] * p["K_4"] * p["P"]) * Ca_in
        / (p["K_4"] * p["K_2"] * (p["K_1"] + p["P"]))
    )
    f2 = (p["k_-2"] * p["P"] + p["k_-4"] * p["K_3"]) / (p["K_3"] + p["P"])
    return f1 * (1.0 - y) - f2 * y


def P_RyR(Ca_in, Ca_SR, p):
    activation = p["k_ryr0"] + (p["k_ryr1"] * Ca_in**3) / (p["k_ryr2"]**3 + Ca_in**3)
    sr_term = Ca_SR**4 / (p["k_ryr3"]**4 + Ca_SR**4)
    return activation * sr_term


def J_RyR_Wang(Ca_SR, Ca_in, p):
    return p["k_RyR"] * P_RyR(Ca_in, Ca_SR, p) * (Ca_SR - Ca_in)


# =============================================================================
# Layer 3: NO / cGMP / PKG cascade
# =============================================================================

def pkg_from_cgmp(cGMP, p):
    """Quasi steady-state PKG activity (Eq. 23)."""
    return cGMP / (p["K_PKG"] + cGMP)


def dNO_dt(NO, t, p):
    E = E_value(t, p)
    R_a, _R_b, R_gper = receptor_occupancies(E, p)
    return p["k_eNOS"] * (R_a + p["omega_GPER"] * R_gper) - p["k_NO_deg"] * NO


def dcGMP_dt(NO, cGMP, p):
    n = p["n_sGC"]
    sGC_rate = p["V_sGC"] * (NO**n) / (p["K_sGC"]**n + NO**n)
    return sGC_rate - p["k_PDE"] * cGMP


# =============================================================================
# MLC phosphorylation submodel
# =============================================================================

def k_MLCK(Ca_in, p):
    return p["k_MLCK_max"] * (Ca_in**p["n_M"]) / (p["K_M"]**p["n_M"] + Ca_in**p["n_M"])


def k_MLCP_eff(PKG, p):
    return p["k_MLCP_0"] * (1.0 + p["a_MLCP"] * PKG)


def dMp_dt(Mp, Ca_in, PKG, p):
    return k_MLCK(Ca_in, p) * (1.0 - Mp) - k_MLCP_eff(PKG, p) * Mp


# =============================================================================
# ODE system
# =============================================================================

def model(t, state, p):
    """
    State layout depends on the active layers:

        [Ca_0, Ca_in, Ca_SR, V, y_g]                         (base)
        + [NO, cGMP]    if estrogen_layer == 3
        + [Mp]          if use_mlc_force is True
    """
    layer = p["estrogen_layer"]
    use_mlc = p["use_mlc_force"]

    Ca_0, Ca_in, Ca_SR, V, y_g = state[:5]

    idx = 5
    if layer == 3:
        NO, cGMP = state[idx], state[idx + 1]
        idx += 2
        PKG = pkg_from_cgmp(cGMP, p)
    else:
        NO, cGMP, PKG = 0.0, 0.0, 0.0

    Mp = state[idx] if use_mlc else 0.0

    # Oestrogen-dependent parameter scalings
    f_VOCC, f_SERCA, f_PMCA, _f_alpha = estrogen_factors(t, p, PKG=PKG)

    # Fluxes
    Jin     = p["delta"] * J_in_Wang(V, Ca_in, Ca_0, p, f_VOCC=f_VOCC)
    JPMCA   = p["delta"] * J_PMCA_Hill(Ca_in, p, f_PMCA=f_PMCA)
    JSERCA  = J_SERCA_Hill(Ca_in, p, f_SERCA=f_SERCA)
    Jip3r   = J_IP3R_Wang(Ca_SR, Ca_in, y_g, p)
    Jryr    = J_RyR_Wang(Ca_SR, Ca_in, p)
    Jleak   = J_leak(Ca_SR, Ca_in, p)
    dyg_dt  = dy_dt(y_g, p, Ca_in)

    # Calcium dynamics
    dCa0_dt  = JPMCA - Jin
    dCain_dt = Jin - JPMCA - JSERCA + Jip3r + Jryr + Jleak
    dCaSR_dt = p["gamma"] * (JSERCA - Jip3r - Jryr - Jleak)

    # Membrane voltage (simple RC model retained from parent code)
    dV_dt = (1.0 / p["c_m"]) * (Ca_0 - Ca_in)

    derivs = [dCa0_dt, dCain_dt, dCaSR_dt, dV_dt, dyg_dt]

    if layer == 3:
        derivs += [dNO_dt(NO, t, p), dcGMP_dt(NO, cGMP, p)]

    if use_mlc:
        derivs += [dMp_dt(Mp, Ca_in, PKG, p)]

    return derivs


# =============================================================================
# Force outputs
# =============================================================================

def algebraic_force(Ca_in, t_array, p):
    """Hill-type force law with optional oestrogen scaling of alpha (Eq. 18)."""
    F = np.empty_like(Ca_in)
    for k, t in enumerate(t_array):
        # PKG only matters when computing f_alpha through Layer 3, but Layer 3
        # delegates force modulation to the MLC submodel, so PKG is unused here.
        _, _, _, f_alpha = estrogen_factors(t, p, PKG=0.0)
        alpha_eff = p["alpha"] * f_alpha
        F[k] = alpha_eff * Ca_in[k]**p["n_F"] / (p["beta"]**p["n_F"] + Ca_in[k]**p["n_F"])
    return F


def mlc_force(Mp, p):
    """Force proportional to fraction of phosphorylated MLCs (Eq. 29)."""
    return p["F_max"] * Mp


# =============================================================================
# Simulation driver
# =============================================================================

def initial_state(p):
    base = [2.0, 0.112, 24.0, -60.0, 0.0]      # Ca_0, Ca_in, Ca_SR, V, y_g
    extra = []
    if p["estrogen_layer"] == 3:
        extra += [0.0, 0.0]                    # NO, cGMP
    if p["use_mlc_force"]:
        extra += [0.0]                         # Mp
    return base + extra


def run_simulation(p, t_span=(0.0, 100.0), n_eval=1000, method="BDF"):
    y0 = initial_state(p)
    t_eval = np.linspace(*t_span, n_eval)
    sol = solve_ivp(model, t_span, y0, args=(p,), t_eval=t_eval,
                    method=method, rtol=1e-6, atol=1e-9)
    return sol


def unpack(sol, p):
    """Return a dict of named trajectories from the ODE solution."""
    out = {
        "t":     sol.t,
        "Ca_0":  sol.y[0],
        "Ca_in": sol.y[1],
        "Ca_SR": sol.y[2],
        "V":     sol.y[3],
        "y_g":   sol.y[4],
    }
    idx = 5
    if p["estrogen_layer"] == 3:
        out["NO"]   = sol.y[idx]
        out["cGMP"] = sol.y[idx + 1]
        out["PKG"]  = pkg_from_cgmp(out["cGMP"], p)
        idx += 2
    if p["use_mlc_force"]:
        out["Mp"] = sol.y[idx]
        out["F"]  = mlc_force(out["Mp"], p)
    else:
        out["F"] = algebraic_force(out["Ca_in"], out["t"], p)
    return out


# =============================================================================
# Example: control vs. oestradiol-stimulated cell
# =============================================================================

if __name__ == "__main__":
    # Control (no oestrogen)
    p_ctrl = dict(params)
    p_ctrl["estrogen_layer"] = 0
    sol_ctrl = run_simulation(p_ctrl)
    res_ctrl = unpack(sol_ctrl, p_ctrl)

    # Layer 1: phenomenological modulation by 17-beta-oestradiol
    p_e1 = dict(params)
    p_e1["estrogen_layer"] = 1
    p_e1["E"] = 1e-5            # 10 microM (saturating, illustrative)
    sol_e1 = run_simulation(p_e1)
    res_e1 = unpack(sol_e1, p_e1)

    # Layer 3 + explicit MLC phosphorylation
    p_e3 = dict(params)
    p_e3["estrogen_layer"] = 3
    p_e3["use_mlc_force"] = True
    p_e3["E"] = 1e-5
    sol_e3 = run_simulation(p_e3)
    res_e3 = unpack(sol_e3, p_e3)

    # ---- Plotting ----
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for res, label in [(res_ctrl, "Control"),
                       (res_e1,   "Layer 1 (E2)"),
                       (res_e3,   "Layer 3 + MLC")]:
        axes[0, 0].plot(res["t"], res["Ca_in"], label=label)
        axes[0, 1].plot(res["t"], res["Ca_SR"], label=label)
        axes[0, 2].plot(res["t"], res["V"],     label=label)
        axes[1, 0].plot(res["t"], res["F"],     label=label)

    axes[0, 0].set_title(r"Cytosolic $Ca^{2+}$ (mM)")
    axes[0, 1].set_title(r"SR $Ca^{2+}$ (mM)")
    axes[0, 2].set_title("Membrane voltage (mV)")
    axes[1, 0].set_title("Contractile force")

    if "PKG" in res_e3:
        axes[1, 1].plot(res_e3["t"], res_e3["NO"],   label="[NO]")
        axes[1, 1].plot(res_e3["t"], res_e3["cGMP"], label="[cGMP]")
        axes[1, 1].set_title("NO / cGMP (mM)")
        axes[1, 1].legend()
        axes[1, 2].plot(res_e3["t"], res_e3["PKG"])
        axes[1, 2].set_title("PKG activity")

    for ax in axes.flat:
        ax.set_xlabel("Time (s)")

    axes[0, 0].legend()
    plt.tight_layout()
    plt.show()
