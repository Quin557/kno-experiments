# KNO vs AM-FNO Current Summary

## Completed Runs

| Benchmark | Run | Best test full rel L2 | Last test full rel L2 | Assessment |
|---|---|---:|---:|---|
| NS-2D v1e-4 | `kno_ns2d_v1e4_o32_m16_r8_ep500_seed42` | `6.2351e-02` | `6.2378e-02` | Valid KNO result; stable convergence. |
| CFD-1D | `kno_cfd1d_o32_m16_r8_ep500_seed42` | `4.2512e-01` | `9.9900e-01` | Completed but unsuitable; likely needs variable normalization/stabilization. |

## Baseline Reference

| Benchmark | AM-FNO paper | AM-FNO local reproduction | Current KNO |
|---|---:|---:|---:|
| NS-2D | `8.51e-02` | `2.4848e-02` on local v1e-4 split | `6.2351e-02` |
| CFD-1D | `1.47e-02` | `1.5164e-02` full / `1.4850e-02` step | Not valid yet |

## Recommendation

Continue with `NS-2D v1e-3` full run next. Pause CFD full runs until CFD variable normalization and summary automation are improved.
