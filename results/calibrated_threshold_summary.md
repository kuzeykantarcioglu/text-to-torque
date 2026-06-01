| condition | final_reward | final_episode_length | body_pos_error | joint_pos_error | anchor_pos_terminations | anchor_ori_terminations | ee_body_pos_terminations | termination_total | steps_per_second |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Loose baseline | 13.657 | 250.0 | 0.271 | 1.545 | 0.0 | 0.0 | 0.0 | 0.0 | 24638.0 |
| Strict baseline | 0.837 | 13.52 | 0.234 | 0.484 | 71.292 | 0.0 | 73.583 | 144.875 | 31223.0 |
| Gradual curriculum, threshold 2 | 13.04 | 250.0 | 0.293 | 1.708 | 0.0 | 0.0 | 0.0 | 0.0 | 24405.0 |
| Gradual curriculum, threshold 1 | 11.934 | 244.86 | 0.294 | 1.423 | 0.083 | 0.208 | 0.0 | 0.292 | 22781.0 |
| Direct calibrated threshold 2 | 12.405 | 250.0 | 0.263 | 1.723 | 0.0 | 0.0 | 0.0 | 0.0 | 18363.0 |
| Direct calibrated threshold 1 | 9.214 | 241.82 | 0.351 | 1.662 | 0.292 | 0.042 | 0.0 | 0.333 | 18179.0 |
