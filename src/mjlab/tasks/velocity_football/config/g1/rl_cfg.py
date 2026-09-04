"""RL configuration for Unitree G1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the RL runner configuration for G1 football training."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity_football",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_temporal_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the TemporalCNN runner configuration for masked ball vision."""
  cfg = unitree_g1_ppo_runner_cfg()
  temporal_model = (
    "mjlab.tasks.velocity_football.rl.temporal_cnn_model:TemporalCNNModel"
  )
  cnn_cfg = {
    "output_channels": (256, 128, 64),
    "kernel_size": 3,
    "activation": "elu",
    "global_pool": "avg",
  }
  cfg.actor.class_name = temporal_model
  cfg.actor.cnn_cfg = cnn_cfg
  cfg.critic.class_name = temporal_model
  cfg.critic.cnn_cfg = cnn_cfg.copy()
  cfg.obs_groups = {
    "actor": ("actor", "actor_history"),
    "critic": ("critic", "critic_history"),
  }
  return cfg


def unitree_g1_visual_mask_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Current-frame Actor control with the same temporal Critic as the test arm."""
  cfg = unitree_g1_temporal_ppo_runner_cfg()
  cfg.actor.class_name = "MLPModel"
  cfg.actor.cnn_cfg = None
  cfg.obs_groups["actor"] = ("actor",)
  return cfg


def unitree_g1_factorial_ppo_runner_cfg(
  *,
  use_b1_history: bool,
) -> RslRlOnPolicyRunnerCfg:
  """Create the frozen A0/A1 runner while keeping one shared temporal Critic."""
  cfg = unitree_g1_ppo_runner_cfg()
  temporal_model = (
    "mjlab.tasks.velocity_football.rl.temporal_cnn_model:TemporalCNNModel"
  )
  cfg.critic.class_name = temporal_model
  cfg.critic.cnn_cfg = {
    "output_channels": (256, 128, 64),
    "kernel_size": 3,
    "activation": "elu",
    "global_pool": "avg",
  }
  cfg.obs_groups = {
    "actor": ("actor",),
    "critic": ("critic", "critic_history"),
  }
  if use_b1_history:
    cfg.actor.class_name = temporal_model
    cfg.actor.cnn_cfg = {
      "output_channels": (64, 64, 64),
      "kernel_size": 3,
      "activation": "elu",
      "dilations": (1, 2, 4),
      "causal": True,
      "output_mode": "last",
    }
    cfg.obs_groups["actor"] = ("actor", "actor_history")
  return cfg


def unitree_g1_factorial_history30_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """A1R0 runner whose causal Actor covers the full 30-frame window."""
  cfg = unitree_g1_factorial_ppo_runner_cfg(use_b1_history=True)
  assert cfg.actor.cnn_cfg is not None
  cfg.actor.cnn_cfg = {
    **cfg.actor.cnn_cfg,
    "output_channels": (64, 64, 64, 64),
    "dilations": (1, 2, 4, 8),
    "activate_last": False,
  }
  return cfg


def unitree_g1_temporal_velocity_pretrain_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """TemporalCNN configuration for the matched walking pretraining stage."""
  cfg = unitree_g1_temporal_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_football_pretrain"
  return cfg


def unitree_g1_velocity_pretrain_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the RL runner configuration for football-compatible walking."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_football_pretrain"
  return cfg


def unitree_g1_klavier_replica_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Klavier Walk PPO config for the 512-256-128, 20k retraining run."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.actor.hidden_dims = (512, 256, 128)
  cfg.critic.hidden_dims = (512, 256, 128)
  cfg.algorithm.entropy_coef = 0.005
  cfg.algorithm.symmetry_cfg = None
  cfg.experiment_name = "g1_velocity_walk_klavier_replica"
  cfg.save_interval = 1000
  # The runner saves after iteration 20_000, matching the reference run's
  # 30_001 setting that produced model_30000.pt.
  cfg.max_iterations = 20_001
  cfg.run_name = "unitree_g1_flat_mlp512_noMirrorLoss_seed42_20k_wandb"
  cfg.upload_model = False
  return cfg


def unitree_g1_klavier_ball_temporal_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Scheme A PPO: transferred 490-D Walk MLP plus a 64-D causal BallCNN."""
  cfg = unitree_g1_factorial_ppo_runner_cfg(use_b1_history=True)
  # Match the Klavier Walk checkpoint MLP so its weights transfer exactly.
  # Keep the BallCNN unchanged.
  cfg.actor.hidden_dims = (1024, 512, 256)
  cfg.critic.hidden_dims = (1024, 512, 256)
  cfg.algorithm.entropy_coef = 0.01
  cfg.algorithm.symmetry_cfg = {
    "data_augmentation_func": (
      "mjlab.tasks.velocity_football.rl.klavier_symmetry:data_augmentation_func"
    ),
    "use_data_augmentation": False,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 1.0,
  }
  cfg.experiment_name = "g1_velocity_football_klavier_ball_temporal"
  cfg.save_interval = 200
  cfg.max_iterations = 50_001
  cfg.run_name = "schemeA_ballcnn64_longdropout10_from_walk20000_seed42_50k_wandb"
  cfg.upload_model = False
  return cfg


def unitree_g1_klavier_legacy512_walk_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Old-width Walk baseline used to initialize the controlled Teacher pair."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.actor.hidden_dims = (512, 256, 128)
  cfg.critic.hidden_dims = (512, 256, 128)
  cfg.algorithm.entropy_coef = 0.01
  cfg.algorithm.symmetry_cfg = None
  cfg.experiment_name = "g1_velocity_walk_klavier_legacy512"
  cfg.save_interval = 1000
  cfg.max_iterations = 20_001
  cfg.run_name = "KlavierLegacy512_LegacyPush_NoSym_seed42_20k_wandb"
  cfg.upload_model = False
  return cfg


def unitree_g1_legacy_xml_legacy512_walk_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Legacy512 Walk comparison using the old standard G1 XML."""
  cfg = unitree_g1_klavier_legacy512_walk_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_walk_legacy_xml_legacy512"
  cfg.run_name = "LegacyXML_Legacy512_LegacyPush_NoSym_seed42_20k_wandb"
  return cfg


def unitree_g1_klavier_legacy512_ball_temporal_ppo_runner_cfg() -> (
  RslRlOnPolicyRunnerCfg
):
  """512-256-128 football Teacher shared by the 0 cm / 5 cm noise pair."""
  cfg = unitree_g1_factorial_ppo_runner_cfg(use_b1_history=True)
  cfg.actor.hidden_dims = (512, 256, 128)
  cfg.critic.hidden_dims = (512, 256, 128)
  cfg.algorithm.entropy_coef = 0.01
  cfg.algorithm.symmetry_cfg = None
  cfg.experiment_name = "g1_velocity_football_klavier_legacy512_ball_temporal"
  cfg.save_interval = 1000
  cfg.max_iterations = 50_001
  cfg.upload_model = False
  return cfg


def unitree_g1_klavier_legacy512_motor_pd_ppo_runner_cfg() -> (
  RslRlOnPolicyRunnerCfg
):
  """Train a Legacy512 Teacher from Walk under explicit motor PD."""
  cfg = unitree_g1_klavier_legacy512_ball_temporal_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_football_klavier_legacy512_motor_pd"
  cfg.max_iterations = 30_000
  cfg.save_interval = 1000
  cfg.run_name = (
    "KlavierLegacy512_MotorPD_IdealPd_Envelope30_ActionAcc01_BallNoise0cm_"
    "FromWalk20k_seed42_30k_wandb"
  )
  return cfg
