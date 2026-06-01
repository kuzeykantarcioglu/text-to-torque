# Final Report Notes

## Experiment Set

All runs used KimoLab/Kimodo-generated Unitree G1 references, MuJoCo Warp PPO
tracking, 4 s prompts resampled to 50 Hz, 1024 parallel environments, and 2000
training iterations unless otherwise noted.

Primary prompts:

- Walk: `A person walks forward`
- Wave: `A person waves with their right hand`
- Tap head: `A person taps themselves on the head`
- Squat: `A person squats down and stands up`

Termination settings:

- Loose: large termination thresholds, intended to preserve full-horizon
  exploration.
- Strict: default/strict tracking termination thresholds.
- Curriculum: walk only; 1000 iterations loose, then resume for 1000 iterations
  under strict thresholds.
- Gradual curriculum: squat seed 0; thresholds 100, 20, 10, 5, 2, 1, then
  default strict thresholds.
- Calibrated threshold: squat seed 0; fixed intermediate thresholds selected
  from the gradual-curriculum boundary and trained from scratch.

Extra runs added on 5/31:

- Squat seed 0 curriculum: 1000 loose iterations followed by 1000 strict
  iterations.
- Squat seed 1 loose and strict replications.
- Reference-only hard prompts: jump, roll forward, backflip, and cartwheel.

Extra runs added on 6/1:

- Squat seed 0 gradual curriculum: thresholds 100 -> 20 -> 10 -> 5 -> 2 -> 1
  -> strict over 3000 total iterations.
- Jump seed 0 loose PPO training for 2000 iterations.
- Squat seed 0 direct calibrated-threshold PPO runs with fixed thresholds 1 and
  2. These were interrupted around 1200 iterations, but logs/videos were synced.

## Main Results

Strict termination is not universally harmful. Gesture-like motions learn well
under both loose and strict settings:

- Wave strict: final reward 18.07, episode length 250.0.
- Tap strict: final reward 17.41, episode length 246.68.

Strict termination fails on the squat reference:

- Squat loose: final reward 13.66, episode length 250.0, termination total 0.0.
- Squat strict: final reward 0.84, episode length 13.52, termination total 144.88.

The squat strict run has lower reported body/joint position error than loose, but
that is not evidence of better tracking because the episode terminates almost
immediately and only evaluates the early part of the motion. Episode length and
termination count are the correct success indicators for this case.

Walk is learnable under both loose and strict termination:

- Walk loose: final reward 11.35, episode length 250.0.
- Walk strict: final reward 13.12, episode length 244.78.
- Walk curriculum: final reward 11.19, episode length 232.64.

The tested walk curriculum worked mechanically but did not outperform strict
training from scratch on final reward or episode length. This is still a useful
negative result: the benefit of relaxing terminations appears motion-dependent,
with the clearest benefit on the squat motion rather than on walk.

The extra squat runs make the conclusion more nuanced:

- Squat seed 1 loose: final reward 12.70, episode length 250.0.
- Squat seed 1 strict: final reward 13.96, episode length 246.82.
- Squat seed 0 curriculum loose stage: final reward 11.62, episode length 250.0.
- Squat seed 0 curriculum strict stage: final reward 0.77, episode length 12.61,
  termination total 144.33.

This means strict termination is seed/path dependent rather than universally
bad. For squat seed 0, strict and abrupt loose-to-strict curriculum fail. For
squat seed 1, strict eventually recovers by 2000 iterations, although it is much
slower early in training. Loose termination is robust across both squat seeds.

The gradual squat curriculum gives a sharper boundary:

- Threshold 100: final reward 4.51, episode length 250.0.
- Threshold 20: final reward 10.72, episode length 250.0.
- Threshold 10: final reward 12.39, episode length 250.0.
- Threshold 5: final reward 13.11, episode length 250.0.
- Threshold 2: final reward 13.04, episode length 250.0.
- Threshold 1: final reward 11.93, episode length 244.86.
- Strict defaults: final reward 0.71, episode length 12.33, termination total
  145.67.

This means the seed-0 squat policy is not simply unable to benefit from a
curriculum. It survives several intermediate thresholds and collapses only when
the final strict/default thresholds are restored.

The direct calibrated-threshold runs turn that diagnosis into an intervention:

- Direct threshold 2: final synced reward 12.41, episode length 250.0,
  termination total 0.0.
- Direct threshold 1: final synced reward 9.21, episode length 241.82,
  termination total 0.33.

These runs were interrupted before the planned 2000 iterations but already show
that finite intermediate thresholds can recover squat seed-0 tracking from
scratch. This is stronger than the curriculum-only story because it is a direct
training intervention, not just an evaluation sweep.

The jump loose PPO run is a useful positive hard-prompt result:

- Jump loose: final reward 15.63, episode length 250.0, body-position error
  0.046, joint-position error 0.727, termination total 0.0.

This prevents the final story from being "hard prompts fail." The better claim
is that difficulty depends on the generated reference and termination boundary;
jump is learnable under loose termination, while squat seed 0 is brittle under
strict termination.

## Reference Diagnostics

Pre-training diagnostics separate the motions in intuitive ways:

- Squat has the largest root height range: about 0.53 m.
- Walk has the largest root speed: about 2.10 m/s.
- Wave has high joint acceleration but still learns under strict termination,
  suggesting joint acceleration alone is not a sufficient failure predictor.

The strongest current predictor of strict PPO failure in this small suite is
large vertical/root motion combined with strict early termination, not simply
high joint acceleration.

## Figure Paths

- `figures/final_outcome_summary.png`: final reward, episode length, and
  termination count by motion/condition.
- `figures/final_reference_diagnostics.png`: root height, root speed, and joint
  acceleration diagnostics.
- `figures/final_squat_termination_curves.png`: loose vs strict squat learning
  curves.
- `figures/final_walk_curriculum_curves.png`: loose, strict, and curriculum walk
  learning curves.
- `figures/final_squat_extra_curves.png`: extra squat seed/curriculum curves.
- `figures/final_gradual_curriculum_curves.png`: gradual seed-0 squat
  curriculum curves.
- `figures/final_calibrated_threshold_curves.png`: loose/strict, gradual
  intermediate thresholds, and direct fixed-threshold squat curves.
- `figures/final_jump_training_curves.png`: loose jump PPO learning curves.
- `figures/final_video_contact_sheet.png`: thumbnails from final rollout videos.
- `figures/final_gradual_curriculum_sequence_panel.png`: rollout thumbnails for
  threshold-2, threshold-1, and strict stages of the gradual squat curriculum.
- `figures/final_calibrated_threshold_sequence_panel.png`: rollout thumbnails
  comparing strict, direct threshold 1, direct threshold 2, and loose squat.
- `figures/final_jump_sequence_panel.png`: rollout thumbnails for loose jump
  PPO.
- `figures/final_hard_reference_panel.png`: jump/roll/backflip/cartwheel
  reference snapshots.

## Suggested Final Report Claims

The project should frame the result as an empirical study of the failure boundary
between text-generated kinematic references and dynamically feasible humanoid
tracking. The strongest claim is:

> In this prompt suite, strict tracking terminations are acceptable for easy
> gesture motions and can learn some squat seeds, but they are less robust on
> motions with large root-height changes. Relaxed terminations recover
> full-horizon learning reliably on squat, while abrupt loose-to-strict
> curricula can collapse under strict thresholds. A gradual curriculum localizes
> the squat seed-0 failure boundary to the final default strict threshold, and
> direct calibrated-threshold training recovers full-horizon squat behavior from
> scratch. A loose jump run shows that some harder generated references are
> still learnable.

Avoid claiming that the curriculum broadly improves learning. The current data
supports "implemented and tested; negative for final strict performance, but
useful for diagnosing where strict termination becomes too brittle." The direct
calibrated-threshold run is the positive method result.

## AI Tools Disclosure Draft

We used AI coding assistance to help scaffold Modal orchestration scripts,
download utilities, diagnostics scripts, plotting scripts, and report notes. All
experiments were run by the team on Modal GPU jobs, and all reported quantitative
results come from saved KimoLab/MuJoCo Warp training logs and generated reference
motion artifacts.
