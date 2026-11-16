# Lane nonlinear observation and learned SBC

Completed 2026-11-15. The visual patch projection and a smooth invertible state-to-observation neural proxy were jointly trained. The original semantic actor is composed with the explicit nonlinear inverse; its weights and the existing learned nonnegative SBC weights are frozen.

- Fresh rendered calibration: 199 independent rollouts, 80 steps each.
- Frozen trajectory CP radius: 0.08753797876881117 in normalized decoded coordinates; rank 190 at alpha 0.05.
- Independent holdout: 96/100 whole trajectories inside the set; 0/100 unsafe. Calibration unsafe: 0/199.
- New continuous-domain numerical SBC audit: passed for all three registered curvatures, zero point witnesses and zero unresolved cells.
- Initial B upper bound: 0.08582801041697322. Unsafe B lower bound: 23.653495792448606.

The raw latent error set is E(x)={h(x+delta)-h(x): ||delta||inf<=q}. It is NOT a constant raw-latent radius box. Explicit invertibility gives inverse(h(x)+e)=x+delta, which connects this model to the semantic actor's existing verifier at the NEW radius. All files are bound in protocol.json and perception_contract.json.

The four uncovered holdout trajectories were observed safe but are not individually covered by this conditional certificate. Split CP gives marginal coverage under the registered exchangeable rollout law; it does not give fixed-state conditional coverage or remove acceptance-selection issues. Numerical audit is not an outward-rounded IEEE proof. The guarantee does not transfer to changed renderers, inference devices/batch shapes, or longer horizons without further justification.

Training used cached features on CPU; fresh visual inference used CUDA device 0 with batch size 8. The server rendering and GPU replay audit matched; a local cross-environment rendering check did not. See runtime_environment.json and replay_audit.json.

This run establishes an executable nonlinear-proxy pipeline, not superiority over the linear model. For detailed steps see docs/details/车道非线性神经代理_联合训练与完整验证_20261115.md in the repository root.
