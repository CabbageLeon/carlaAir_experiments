

# Appendix

This appendix provides additional platform and evaluation details that support the main paper. Appendix A describes the CARLA-AIR runtime design, coordinate-frame unification, sensing support, software stack, and runtime consistency validation. Appendix B provides task settings, cooperation modes, prompt templates, metric definitions, baseline adaptations, and timing statistics.

## A CARLA-AIR Platform Details

This appendix provides implementation details for CARLA-AIR, including coordinate-frame unification, sensor and API support, software versions, source modifications, and runtime requirements. 

*(Note: Figure 4 illustrates the core integration mechanism: CARLA-AIR resolves the single-GameMode constraint by keeping CARLA as the authoritative world manager and composing the AirSim aerial subsystem as an actor-level component.)*

### A.1 Coordinate Frame Unification and Single-Tick Execution
All cross-agent states, observations, and cooperation metrics are expressed in a unified metric frame. CARLA uses a left-handed Unreal Engine frame (centimetres, Z-up), while AirSim adopts a NED frame (metres, Z-down). As illustrated in Figure 5, we apply a deterministic transformation at every simulation tick:

$$
p_{NED} = \frac{1}{100} \begin{pmatrix} p_x - o_x \\ p_y - o_y \\ -(p_z - o_z) \end{pmatrix}, \quad q_{NED} = \frac{(w, q_x, q_y, -q_z)}{\|(w, q_x, q_y, -q_z)\|}
$$

where $(o_x, o_y, o_z)$ is the AirSim origin in Unreal Engine coordinates. This transformation ensures that UAV states, UGV states, relative poses, and cooperation metrics are evaluated in the same metric frame.

**Algorithm 1: Single-tick execution in CARLA-AIR**
```text
Require: World state W_t, UGV action a_t^G, UAV action a_t^A
Ensure: Updated world W_{t+1}, synchronized observations o_{t+1}^G, o_{t+1}^A

1: Apply UGV command a_t^G through the CARLA control interface
2: Apply UAV command a_t^A through the AirSim control interface
3: W_{t+1} ← PHYSICSSTEP(W_t)
4: Render and sample sensors from W_{t+1}
5: o_{t+1}^G ← SAMPLESENSORS(UGV, W_{t+1})
6: o_{t+1}^A ← SAMPLESENSORS(UAV, W_{t+1})
7: return W_{t+1}, o_{t+1}^G, o_{t+1}^A
```
Rendering is forced to complete within the same tick via `FlushRenderingCommands()`, so all sensor outputs are computed from a single physics state before the tick returns.

### A.2 Sensors and Native APIs
Table 7 lists the sensor modalities supported by CARLA-AIR. All enabled sensors are sampled at the shared simulation tick, so cross-agent correspondence does not require interpolation or extrapolation. CARLA and AirSim retain their original Python APIs and ROS2 interfaces; both command streams are resolved inside the same Unreal Engine runtime, so existing client code continues to operate without modification.

**Table 7: Sensor modalities available in CARLA-AIR.** UGV and UAV columns indicate availability on each platform.

| # | Sensor | UGV | UAV |
|---|---|:---:|:---:|
| 1 | RGB camera (forward) | ✓ | ✓ |
| 2 | RGB camera (downward) | — | ✓ |
| 3 | RGB camera (wide-angle) | ✓ | ✓ |
| 4 | Depth camera | ✓ | ✓ |
| 5 | Semantic segmentation camera | ✓ | ✓ |
| 6 | Instance segmentation camera | ✓ | ✓ |
| 7 | Surface normal camera | ✓ | ✓ |
| 8 | LiDAR | ✓ | ✓ |
| 9 | Radar | ✓ | — |
| 10 | IMU | ✓ | ✓ |
| 11 | GNSS | ✓ | ✓ |
| 12 | Barometer | — | ✓ |
| 13 | Magnetometer | — | ✓ |
| 14 | Optical flow camera | ✓ | ✓ |
| 15 | Event camera | ✓ | ✓ |
| 16 | Collision sensor | ✓ | ✓ |
| 17 | Lane invasion sensor | ✓ | — |
| 18 | Obstacle distance sensor | ✓ | ✓ |

### A.3 Software and Hardware
Table 8 summarizes the software stack used in our experiments. All builds and experiments are run on a single workstation with an Intel Core Ultra 9 275HX (24C/24T) CPU, an NVIDIA RTX 5090 Laptop GPU (24 GB), and 128 GB RAM under Ubuntu 22.04. Runtime evaluation requires a GPU with sufficient memory for the selected VLA baseline.

**Table 8: Software stack used in our experiments.**

| Component | Version |
|---|---|
| Unreal Engine | 4.26.2 |
| CARLA | 0.9.16 |
| AirSim | 1.7.0 |
| Python | 3.8.18 |
| PyTorch | 2.1.2 (CUDA 11.8) |
| ROS 2 | Humble Hawksbill |
| OS | Ubuntu 22.04 LTS |

*Note on AirSim:* The original open-source AirSim project is no longer actively maintained by Microsoft. We build on Colosseum [32], a community-maintained fork that preserves the AirSim API surface and remains compatible with Unreal Engine.

### A.4 Runtime Consistency Validation
This section provides the experimental details underlying the runtime consistency check reported in Section 3.2 (Table 2), and extends the validation to additional sensor configurations, control frequencies, and agent counts.

**Setup.** The UGV drives along a predefined route in Town10 at constant speed, while the UAV is driven by a position P-controller to follow the UGV at a fixed relative offset; the underlying flight control uses AirSim’s built-in cascaded PID module (the AirSim default configuration [22]).

**Comparison configurations.** Under the bridge runtime, CARLA and AirSim run as separate processes and exchange messages via ROS2 at 10 Hz—the architecture commonly used by prior air-ground integrations [3]. Under the CARLA-AIR runtime, both simulators share a single simulation tick. Both runtimes execute the same 100 episodes with matched random seeds, identical spawn points, and the same follow controller; the only varied factor is the runtime architecture.

**Measurements.** Sensor offset is the per-tick timestamp difference $\|\tau_{UAV} - \tau_{UGV}\|$ (ms) between matched UAV and UGV sensor frames. Follow-error std is the per-episode mean of the Euclidean distance (in metres) between the UAV’s actual relative pose and its target relative pose during the steady-state segment of the episode, taken as the standard deviation across the 100 episodes; this is the $\sigma$ reported in Table 2. Wall-clock jitter (used in Table 9) is the standard deviation of OS frame-delivery timestamps after the physics tick returns, and arises from GPU flush overhead. The reported 5.1× noise reduction has a 95% confidence interval of [3.4×, 7.7×] (F-distribution, $n_1 = n_2 = 100$).

**Stress-test extension.** Table 9 extends this protocol to heavier sensor payloads, higher control rates, multi-agent settings, and dense traffic. CARLA-AIR maintains $\Delta t = 0$ ms by construction in all configurations, with $\sigma$ ratios in the 4.2×–6.3× range; remaining wall-clock jitter (4–12 ms P95) is substantially lower than bridge timing noise.

**Table 9: Runtime stress test.** Bridge timestamp offset (mean/P95, ms), wall-clock delivery jitter (P95, ms), and metric noise reduction ($\sigma$ ratio = bridge / CARLA-AIR cooperation-metric std, mean ± std over 5 independent 100-episode runs). CARLA-AIR maintains zero simulation-timestamp offset in all settings.

| Setting (agents, sensors, Hz) | Bridge offset mean/P95 (ms) | CARLA-AIR offset (ms) | WC jitter P95 (ms) | $\sigma$ ratio |
|---|---|---|---|---|
| Easy-RGB (1U+1G, RGB, 10 Hz) | 12.4 / 34 | 0.0 | 4 | 5.1 ± 0.4× |
| Multi-sensor (1U+1G, RGB+D+L, 10 Hz) | 15.2 / 41 | 0.0 | 7 | 4.8 ± 0.4× |
| High-rate (1U+1G, RGB, 30 Hz) | 18.7 / 52 | 0.0 | 9 | 6.3 ± 0.5× |
| Multi-agent (2U+2G, RGB+L, 10 Hz) | 22.1 / 64 | 0.0 | 12 | 4.2 ± 0.4× |
| Dense-traffic (1U+1G, RGB, 10 Hz) | 13.1 / 37 | 0.0 | 5 | 5.0 ± 0.4× |

### A.5 Source Modification Summary
The CARLA-AIR integration modifies only a small number of files relative to the CARLA upstream codebase:
* `CarlaUE4GameMode.h`: adds the AirSim flight actor declaration and composition pointer.
* `CarlaUE4GameMode.cpp`: instantiates the AirSim actor and synchronizes it with the CARLA world lifecycle.
* `CarlaUE4.Build.cs`: adds the AirSim module dependency.

The integration preserves the native CARLA and AirSim client-facing APIs.

---

## B Additional Diagnostic Evaluation Details

This appendix provides details omitted from the main diagnostic evaluation section, including task settings, cooperation modes, prompt templates, metric definitions, baseline adaptations, and evaluation protocol.

### B.1 Task Details
**Cooperative Moving-Platform Landing.** A UGV truck drives along an urban road while providing a flat rear cargo bed as the landing surface. The UAV receives the instruction: *"Follow the moving truck, align above its rear cargo bed, and land safely."* The task consists of tracking, alignment, and landing, with a 60 s episode time limit. It is successful only when the UAV lands on the rear cargo bed without collision, side impact, or hard landing.

**Cooperative Occlusion-Recovery Escort.** A UGV drives along an urban route and becomes temporarily occluded by bridges, buildings, or large artifacts. The UAV must escort the UGV and recover visual contact after the target becomes invisible. Each escort episode has a 90 s time limit. The C1 cue describes the UGV’s motion intent and expected reappearance direction, while the VLA policy still outputs only UAV actions.

### B.2 Cooperation Modes
* **C0: Independent execution.** The UAV and UGV do not communicate. The UGV follows a predefined speed profile or route, while the UAV relies only on onboard RGB observations and the task instruction.
* **C1: UGV-to-UAV semantic prompting.** The UGV provides a compact semantic cue to the UAV. For landing, the cue describes the relative direction to the cargo bed, coarse truck motion, and landing phase. For occlusion recovery, it describes the occlusion status, motion intent, and expected reappearance direction. The UAV still outputs only native UAV actions.
* **C2: Bidirectional UAV-to-UGV action coupling.** C2 is used only for Moving-Platform Landing. The UAV receives the same semantic cue as in C1. The magnitude of the UAV’s commanded forward velocity is passed directly to a fixed UGV longitudinal controller, with no intermediate phase decoder or learned mapping:

$$
v_{UGV}(t) = v_0 \cdot \text{clip}\left(\frac{\|v_{fwd}^{UAV}(t)\|}{v_{ref}}, 0.5, 1.5\right)
$$

where $v_0 = 4.0$ m/s is the nominal UGV speed and $v_{ref} = 2.0$ m/s is a reference scaling constant. The $\text{clip}(\cdot, 0.5, 1.5)$ operator bounds the multiplicative factor to $[0.5\times, 1.5\times]$ to prevent extreme values. The controller modulates only longitudinal speed; the UGV heading and route remain unchanged. The update frequency matches the UAV decision frequency. For baselines whose native output is not a continuous velocity vector (OpenFly, AerialVLN), $\|v_{fwd}^{UAV}(t)\|$ is obtained from the realized UAV forward velocity in the simulator at the same tick. For waypoint-output baselines (SPF, OpenUAV), it is computed as commanded waypoint displacement divided by the inference period. The updated UGV state is then fed back to the UAV prompt for the next step. No additional VLA output head or UGV steering command is introduced; the protocol is intentionally a naive form of bidirectional action coupling.

### B.3 Prompt Templates
**Table 10: Full prompt protocol.** All VLA baselines receive the same base instruction within each task. C1 uses semantic partner-state prompts. C2 uses the same UAV-side prompt as C1 and only enables the UGV-side longitudinal response to UAV action.

| Task | Mode | Interaction | UAV prompt example |
|---|---|---|---|
| Landing | C0 | No communication | Follow the moving truck, keep it in view, align above the flat rear cargo bed, and land safely on the cargo bed. |
| Landing | C1 | Semantic state cue | Follow the moving truck, keep it in view, align above the flat rear cargo bed, and land safely on the cargo bed.<br><br>*Assistant hint: the cargo bed is forward-left. The truck is moving slowly. Current phase: approach. Use the hint only to choose your next UAV action.* |
| Landing | C2 | Same UAV prompt as C1; UGV speed responds to UAV forward velocity | Follow the moving truck, keep it in view, align above the flat rear cargo bed, and land safely on the cargo bed.<br><br>*Assistant hint: the cargo bed is forward-left. The truck is moving slowly. Current phase: approach. Use the hint only to choose your next UAV action.* |
| Escort | C0 | No communication | Follow the moving truck and keep it in view. |
| Escort | C1 | Occlusion-recovery cue | Follow the moving truck and keep it in view.<br><br>*Assistant hint: the truck is temporarily occluded by the bridge. The truck continues forward and will reappear on the forward-right side. Current phase: occlusion recovery. Use the hint only to recover visual contact.* |
| Landing | C1-Oracle-Bearing | Oracle geometric cue | Follow the moving truck, keep it in view, align above the flat rear cargo bed, and land safely.<br><br>*State update: cargo bed at bearing 312°, range 6.2 m, elevation −8°. Phase: approach. Use the state update only to choose your next UAV action.* |

### B.4 Metric Details
* **Tracking Success Rate (TSR).** Measures whether the target truck remains inside the UAV camera view for at least $K = 3$ s of cumulative time before task termination. It evaluates the single-UAV tracking primitive rather than final task success.
* **Landing Success Rate (LSR).** Measures the fraction of episodes in which the UAV lands on the rear cargo bed within the 60 s time limit and remains stable after touchdown, defined as no further displacement exceeding 0.3 m within 2 s of first contact.
* **Cooperative Conversion Rate (CCR).** Measures whether single-UAV tracking becomes cooperative landing:
  $$ \text{CCR} = \frac{\text{LSR}}{\max(\text{TSR}, \epsilon)} $$
  We set $\epsilon = 0.05$; substituting $\epsilon \in \{0.01, 0.05, 0.10\}$ changes CCR by at most 0.01 across all reported conditions, as all baselines achieve TSR $\ge 0.55$ under C0. A high TSR with low CCR indicates that the UAV can track the moving platform but cannot convert tracking into landing.
* **Cooperation Gain (CG).** Measures the change in LSR relative to independent execution: $\text{CG}(C_k) = \text{LSR}(C_k) - \text{LSR}(C_0)$. Positive CG means that the cooperation mode improves over C0, while negative CG indicates degradation.
* **Occlusion-recovery metrics.** **Recovery Success Rate (RSR)** measures whether the UAV recovers visual contact with the UGV within 15 s of occlusion onset; success requires IoU $\ge 0.15$ between the UAV camera view and the UGV bounding box, sustained for at least 0.5 s. **Re-acquisition Time (RAT)** is measured from occlusion onset to the first frame satisfying the IoU threshold; it is capped at 15 s for episodes with no recovery. Each escort episode contains 1–3 occlusion events sampled from three geometry types (bridge underpass: 40%, building: 35%, large artifacts: 25%), with occlusion duration drawn uniformly from 4–12 s.
* **Timing metrics.** **Decision Frequency (DF)** is the realized control update rate of each aerial policy. **Effective Coordination Latency (ECL)** measures the delay from a newly generated UAV action or UGV state update to its use by the partner side in the next control step.
* **Statistical analysis.** All values are reported as mean ± std over 3 seeds × 50 episodes per condition. Across-baseline trends are tested with a sign test on per-seed-mean CGs ($n=5$ baselines). Single-baseline 95% confidence intervals are computed via hierarchical bootstrap clustered by seed (1000 resamples) to account for episode dependencies within seeds.

### B.5 Baseline Adaptation Details
Table 11 summarizes how each baseline is adapted to the CARLA-AIR evaluation interface. All baselines use officially released checkpoints without fine-tuning or task-specific training.

**Table 11: Baseline adaptation summary.** All methods use official checkpoints without fine-tuning. Heterogeneous low-level wrappers preserve each baseline’s native operating regime; comparisons reflect policy family characteristics rather than implementation-matched rankings. DF: realized decision frequency under evaluation conditions.

| Method | Original task | Native output | Action mapping | Controller | DF (Hz) |
|---|---|---|---|---|---|
| **AerialVLA** | UAV nav. + VLA | UAV velocity cmd | Direct passthrough; task prompt rewritten for landing/escort | AirSim cmd passthrough | 6.2 |
| **OpenFly** | Aerial VLN + VLA | Discrete UAV action | {Forward 3/6/9 m, Turn left/right, Up, Down, Stop} mapped to fixed-duration velocity bursts | Fixed-duration AirSim cmd | 3.1 |
| **OpenUAV** | UAV traj. generation | Dense traj. array | 1 s segment sampled at 10 Hz; replanned every 0.6 s | AirSim velocity controller | 1.6 |
| **SPF** | UAV way-point nav. | Single waypoint | One waypoint per inference; position controller tracks it | AirSim position ctrl ($K_p=0.8$) | 1.1 |
| **AerialVLN** | Language-cond. UAV nav. | Discrete action cmd | {forward, left, right, up, down, hover} mapped to 0.5 s velocity bursts at 1.5 m/s | Fixed-duration AirSim cmd | 9.0 |
| **Rule-Coop-State** | Designed for this eval | UAV + UGV cmd pair | Direct metric-state feedback (incl. UGV side) | State-feedback rule | 50+ |

* **AerialVLA:** An end-to-end aerial VLA policy outputting continuous UAV velocity commands directly from onboard visual observations and language instructions. Its native UAV-action interface is kept unchanged; cooperation is introduced only through the C0/C1/C2 protocols.
* **OpenFly:** A keyframe-aware aerial VLA model fine-tuned from OpenVLA-7B on the OpenFly aerial VLN dataset [5]. Its native output is a discrete action vocabulary, which we map to fixed-duration velocity bursts in CARLA-AIR following the same adaptation pattern used for AerialVLN.
* **OpenUAV:** An end-to-end aerial VLA model that generates a continuous UAV trajectory via MLLM-based planning. The first 1 s segment of the trajectory is passed to the AirSim velocity controller, and the trajectory is replanned every 0.6 s.
* **SPF:** Uses explicit spatial reasoning via a VLM planner and outputs a single waypoint per inference. The waypoint is tracked by a position controller.
* **AerialVLN:** A pre-LLM cross-modal aerial navigation baseline. Its discrete action vocabulary is mapped to fixed-duration velocity bursts in CARLA-AIR.
* **Rule-Coop-State:** Uses explicit metric state (UAV–cargo-bed relative pose, relative velocity, UGV speed, landing phase) and applies deterministic low-latency rules for UAV descent and UGV longitudinal speed adjustment.

### B.6 Prompt-Format Ablation
Table 12 shows the C1 prompt variants used in the prompt-format ablation. The base task instruction remains unchanged within each task; only the appended assistant hint is modified. C1-Sem is the default semantic partner-state cue used in the main evaluation. C1-Num uses structured numeric state fields. C1-Noisy corrupts direction, motion, or phase fields. C1-Oracle-Bearing replaces the semantic hint with ground-truth geometric bearing, range, and elevation while keeping the native UAV action interface unchanged.

**Table 12: Prompt variants for C1 ablation.** Examples show the assistant hint appended to the unchanged base task instruction. Landing variants provide cargo-bed state, while escort variants provide occlusion-recovery and expected reappearance state.

| Task | Variant | State format | Assistant hint example |
|---|---|---|---|
| Landing | C1-Sem | Semantic | *Assistant hint: the cargo bed is forward-left. The truck is moving slowly. Current phase: approach. Use the hint only to choose your next UAV action.* |
| Landing | C1-Num | Numeric | *State: truck speed = 2.0 m/s; truck heading = 15 deg; relative bearing to cargo bed = -30 deg; relative distance to cargo bed = 8.0 m; phase = approach.* |
| Landing | C1-Noisy | Corrupted semantic | *Assistant hint: the cargo bed is right. The truck is nearly stopped. Current phase: descend. Use the hint only to choose your next UAV action.* |
| Landing | C1-Oracle-Bearing | Oracle geometry | *State update: cargo bed at bearing 312°, range 6.2 m, elevation −8°. Phase: approach. Use the state update only to choose your next UAV action.* |
| Escort | C1-Sem | Semantic | *Assistant hint: the truck is temporarily occluded by the bridge. The truck continues forward and will reappear on the forward-right side. Current phase: occlusion recovery. Use the hint only to recover visual contact.* |
| Escort | C1-Num | Numeric | *State: occlusion = true; truck speed = 2.0 m/s; expected reappearance bearing = 35 deg; expected reappearance distance = 9.4 m; phase = occlusion recovery.* |
| Escort | C1-Noisy | Corrupted semantic | *Assistant hint: the truck is temporarily occluded. The truck will reappear on the rear-left side. Current phase: normal escort. Use the hint only to recover visual contact.* |
| Escort | C1-Oracle-Bearing | Oracle geometry | *State update: UGV at bearing 35°, range 9.4 m, elevation −12°. Phase: occlusion recovery. Use the state update only to recover visual contact.* |

**Table 13: Timing statistics.** DF: realized decision frequency (Hz). ECL (Effective Coordination Latency): delay from policy inference completion to partner-side controller consumption; excludes UAV actuator delay. Values are episode-level medians with IQR (P25–P75) over 50 episodes per method.

| Method | DF (Hz) ↑ | ECL mean (ms) ↓ | ECL P95 (ms) ↓ | ECL IQR (ms) |
|---|---|---|---|---|
| AerialVLA | 6.2 | 160 | 260 | 120–190 |
| OpenFly | 3.1 | 330 | 520 | 240–400 |
| OpenUAV | 1.6 | 620 | 950 | 450–740 |
| SPF | 1.1 | 850 | 1250 | 620–1050 |
| AerialVLN | 9.0 | 110 | 180 | 80–135 |
| Rule-Coop-State | 50+ | 20 | 35 | 15–25 |

### B.7 Evaluation Protocol
Unless otherwise specified, each method is evaluated under the same route, spawn, weather, and random-seed protocol within each diagnostic task. The UGV route and speed profile are fixed for C0 and C1. In C2, only the UGV longitudinal speed is adjusted by the fixed response controller; the route is unchanged. All VLA policies receive the same task instruction and the same mode-specific prompt template.

--- 