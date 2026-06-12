from ml_collections import ConfigDict


def get_default_config():
    config = ConfigDict()

    # File Paths
    config.scene_file = 'regression/mjcf/scene_mjx_transparent.xml'
    config.datasets = ('regression/data/chirp-trajectories',)
    config.evaluation_dataset = 'regression/data/chirp-trajectories'

    # Physics Settings
    config.physics = ConfigDict()
    config.physics.timestep = 0.004
    config.physics.control_rate = 0.02
    config.physics.solver = 'newton'    # 'newton', 'cg', 'pgs'
    config.physics.iterations = 5
    config.physics.ls_iterations = 5
    config.physics.use_reverse_mode = False

    # Training Hyperparameters
    config.training = ConfigDict()
    config.training.seed = 0
    config.training.num_epochs = 20
    config.training.batches_per_epoch = 256
    config.training.minibatch_size = 25
    config.training.window_length = 25

    # Optimizer Settings
    config.optimizer = ConfigDict()
    config.optimizer.lr_init = 1e-5
    config.optimizer.lr_peak = 1e-2
    config.optimizer.lr_end = 1e-6
    config.optimizer.warmup_pct = 0.1
    config.optimizer.weight_decay = 1e-4
    config.optimizer.clip_norm = 1.0

    # Loss Function Weights
    config.loss = ConfigDict()
    config.loss.weights = ConfigDict({
        'position': 1.0,
        'velocity': 1.0,
        'actuator_force': 1.0,
    })

    # Loss Function Type: rmse, mse, mae, huber
    config.loss.type = 'mse'

    # Parameters to Regress:
    # Format: {param_name: {field: mjx_attr, column: optional_int, bounds: (min, max)}}
    config.regression = ConfigDict()

    config.regression.dof_frictionloss = ConfigDict({
        'field': 'dof_frictionloss', 'bounds': (1e-4, 1e2)
    })
    config.regression.dof_damping = ConfigDict({
        'field': 'dof_damping', 'bounds': (1e-4, 1e2)
    })
    config.regression.dof_armature = ConfigDict({
        'field': 'dof_armature', 'bounds': (1e-4, 1e2)
    })

    config.regression.log_cholesky_inertia = ConfigDict({
        'field': 'log_cholesky_inertia',
        'body_names': [
            'front_right_hip', 'front_right_thigh', 'front_right_calf',
            'front_left_hip', 'front_left_thigh', 'front_left_calf',
            'hind_right_hip', 'hind_right_thigh', 'hind_right_calf',
            'hind_left_hip', 'hind_left_thigh', 'hind_left_calf',
        ],
        'bounds': None
    })

    # Example of possible additional parameters to regress:
    # config.regression.qpos0 = ConfigDict({
    #     'field': 'qpos0', 'bounds': None
    # })
    # config.regression.actuator_dynprm = ConfigDict({
    #     'field': 'actuator_dynprm', 'column': 0, 'bounds': (1e-15, 1e2)
    # })

    # WandB
    config.wandb = ConfigDict()
    config.wandb.project = "Parameter-Regression-Unitree-Go2"
    config.wandb.group = None

    return config
