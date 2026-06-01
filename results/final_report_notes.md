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

Extra runs added on 5/31:

- Squat seed 0 curriculum: 1000 loose iterations followed by 1000 strict
  iterations.
- Squat seed 1 loose and strict replications.
- Reference-only hard prompts: jump, roll forward, backflip, and cartwheel.

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
- `figures/final_video_contact_sheet.png`: thumbnails from final rollout videos.
- `figures/final_hard_reference_panel.png`: jump/roll/backflip/cartwheel
  reference-only snapshots.

## Suggested Final Report Claims

The project should frame the result as an empirical study of the failure boundary
between text-generated kinematic references and dynamically feasible humanoid
tracking. The strongest claim is:

> In this prompt suite, strict tracking terminations are acceptable for easy
> gesture motions and can learn some squat seeds, but they are less robust on
> motions with large root-height changes. Relaxed terminations recover
> full-horizon learning reliably on squat, while abrupt loose-to-strict
> curricula can collapse under strict thresholds.

Avoid claiming that the curriculum broadly improves learning. The current data
supports "implemented and tested; negative on walk and failed on squat seed 0
under abrupt strict transfer" rather than a positive curriculum result.

## AI Tools Disclosure Draft

We used AI coding assistance to help scaffold Modal orchestration scripts,
download utilities, diagnostics scripts, plotting scripts, and report notes. All
experiments were run by the team on Modal GPU jobs, and all reported quantitative
results come from saved KimoLab/MuJoCo Warp training logs and generated reference
motion artifacts.
