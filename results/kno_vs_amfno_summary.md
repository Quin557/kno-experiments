# KNO vs AM-FNO Current Summary

## Completed Runs

| Benchmark | Run | Best test full rel L2 | Last test full rel L2 | Assessment |
|---|---|---:|---:|---|
| NS-2D v1e-4 | `kno_ns2d_v1e4_o32_m16_r8_ep500_seed42` | `6.2351e-02` | `6.2378e-02` | Valid KNO result; stable convergence. |
| NS-2D v1e-3 | `kno_ns2d_v1e3_o32_m16_r8_ep500_seed42` | `3.5535e-03` | `4.0387e-03` | Stable supplemental KNO result; not directly comparable with the v1e-4 AM-FNO local split. |
| CFD-1D | `kno_cfd1d_o32_m16_r8_ep500_seed42` | `4.2512e-01` | `9.9900e-01` | Completed but unsuitable; likely needs variable normalization/stabilization. |
| CFD-1D | `kno_cfd1d_norm_o32_m16_r8_ep500_seed42` | `3.5024e-02` | `3.5331e-02` | Stable after per-variable CFD normalization; use this as the current KNO CFD-1D result. |
| CFD-2D | `kno_cfd2d_norm_o32_m16_r8_ep500_seed42` | `4.4433e-02` | `4.4433e-02` | Stable and still improving at the final epoch, but far behind AM-FNO CFD-2D. |
| CFD-2D | `kno_cfd2d_norm_o64_m16_r8_ep100_seed42` | `6.3840e-02` | `1.7205e-01` | Wider model improves the best first-100-epoch value slightly over `o=32`, but is unstable and should not be continued directly. |

## Baseline Reference

| Benchmark | AM-FNO paper | AM-FNO local reproduction | Current KNO |
|---|---:|---:|---:|
| NS-2D | `8.51e-02` | `2.4848e-02` on local v1e-4 split | `6.2351e-02` |
| CFD-1D | `1.47e-02` step | `1.5164e-02` full / `1.4850e-02` step | `3.5024e-02` full / `3.4501e-02` step |
| CFD-2D | `2.16e-03` step | `2.7686e-03` full / `2.7442e-03` step | `4.4433e-02` full / `4.2271e-02` step |

## Recommendation

The `o=64` CFD-2D tuning run did not meet the continuation threshold: its best 100-epoch result is only modestly better than the `o=32` first-100-epoch result, while its final and late-epoch metrics are much worse. Do not continue this exact setting to 500 epochs. Either freeze `kno_cfd2d_norm_o32_m16_r8_ep500_seed42` as the main CFD-2D result, or run one final stabilization check with `o=64`, lower learning rate `5e-4`, and `max_grad_norm=0.5` for 100 epochs before writing the final report.
