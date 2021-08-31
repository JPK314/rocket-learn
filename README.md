# rocket-learn
RLGym training tool

## TODO
- Add logging ✔ (wandb only)
  - Give both PPO and RolloutGenerator access ✔
- Add reward normalization (and distribution?)
- Model freedom
  - Multiple inputs
  - Allow shared layers (ex.: `PPOAgent(shared, actor, critic)`) ✔
  - Recurrent?
  - Continuous actions if we really want
- Redis features 
  - Full setup (architecture, params, config?) communicated via redis, can start worker with only IP
  - Version quality is most important measurement, need to log it ✔
  - Implement quality update
    - How to account for mix of agents in a single team? TrueSkill would probably be better
    - Should workers be the ones updating?
- Long-term plan is to set up a stream and let (at least some) people contribute with rollouts
  - Keep track of who is contributing ✔, make on-screen leaderboards
  - Exact setup should probably be in different repo
  - Rolv can keep it running on current PC, planning to get new one
  - Need to come to agreement on config (architecture, reward func, parameters etc.)
  - See `serious.py` for suggestions
