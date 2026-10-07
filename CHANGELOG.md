# Changelog

All notable changes to this project will be documented in this file.

## Unreleased
- PDK junction model is now parameterised by `PDK.k_Delta` (the intrinsic
  cryogenic Ambegaokar–Baratoff efficiency k_Δ = Ic·(Rn+Rx)/(πΔ_eff/2e)) and
  `PDK.jj_R0` (R*). `PDK.RI_factor` is a derived property (k_Δ·πΔ_eff/2e, with
  a setter for volt-based callers) and the f01(Rn) prefactor is computed by
  `PDK.A(T)` = k_Δ·Δ_eff/e²·tanh(Δ_eff/2k_BT) — never stored.
  `PDK.k_Delta_at(Rn)` gives the model's per-junction k_Δ·Rn/(Rn+R*);
  `PDK.k_Delta_measured(f01, Ec, Rn)` the fit-free per-junction value;
  `PDK.IcR_ideal()` the bound πΔ_eff/(2e).
- `qfoundry.utils`: `Ej_from_f01`, `IcR_ideal`, `IcR_measured`, `k_Delta` —
  per-junction Ambegaokar–Baratoff metrics from measured f01/Ec/raw Rn with
  no fitted parameters (k_Δ ≡ 2e·Ic·Rn/(πΔ_eff)).

## 0.1.0 - 2025-08-11
- Switch to pyproject.toml packaging
- Expose __version__ per PEP 440
- Add MkDocs documentation scaffold
- Update README; add references
