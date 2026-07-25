# KNO vs AM-FNO Current Summary

## Completed Runs

| Benchmark | Run | Best test full rel L2 | Last test full rel L2 | Assessment |
|---|---|---:|---:|---|
| NS-2D v1e-4 | `kno_ns2d_v1e4_o32_m16_r8_ep500_seed42` | `6.2351e-02` | `6.2378e-02` | Valid KNO result; stable convergence. |
| NS-2D v1e-3 | `kno_ns2d_v1e3_o32_m16_r8_ep500_seed42` | `3.5535e-03` | `4.0387e-03` | Stable supplemental KNO result; not directly comparable with the v1e-4 AM-FNO local split. |
| CFD-1D | `kno_cfd1d_o32_m16_r8_ep500_seed42` | `4.2512e-01` | `9.9900e-01` | Completed but unsuitable; likely needs variable normalization/stabilization. |
| CFD-1D | `kno_cfd1d_norm_o32_m16_r8_ep500_seed42` | `3.5024e-02` | `3.5331e-02` | Stable after per-variable CFD normalization; use this as the current KNO CFD-1D result. |

## Baseline Reference

| Benchmark | AM-FNO paper | AM-FNO local reproduction | Current KNO |
|---|---:|---:|---:|
| NS-2D | `8.51e-02` | `2.4848e-02` on local v1e-4 split | `6.2351e-02` |
| CFD-1D | `1.47e-02` step | `1.5164e-02` full / `1.4850e-02` step | `3.5024e-02` full / `3.4501e-02` step |

## Recommendation

NS-2D v1e-3 is complete and healthy. Next, run `CFD-2D` with the same normalization path used by the successful CFD-1D run. Start with 100 epochs; if `test_full_rel_l2` is decreasing or stable rather than diverging, continue to the full 500-epoch CFD-2D run.
