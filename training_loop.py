from absl import app

import jax

from train_from_config import train_from_config as train


def main(argv=None):
    # Train Config:
    train_config = {
        'vendor': {
            'tag': 'vendor-position',
            'curriculum': ['baseline', 'finetune'],
            'parameter_checkpoint': None,
            'domain_randomization': 'domain_randomize',
            'wandb_tags': ['vendor-position'],
        },
        'transparent': {
            'tag': 'transparent-position',
            'curriculum': ['baseline', 'finetune'],
            'parameter_checkpoint': None,
            'domain_randomization': 'domain_randomize',
            'wandb_tags': ['transparent-position'],
        },
        'regressed': {
            'tag': 'regressed-position',
            'curriculum': ['baseline', 'finetune'],
            'parameter_checkpoint': 'regression/checkpoints/hardy-sound-81',
            'domain_randomization': 'domain_randomize',
            'wandb_tags': ['regressed-position'],
        },
        'uniform': {
            'tag': 'vendor-position',
            'curriculum': ['baseline', 'finetune'],
            'parameter_checkpoint': None,
            'domain_randomization': 'uniform_domain_randomize',
            'wandb_tags': ['uniform-position'],
        },
     }

    for config_name, config in train_config.items():
        print(f'Running: {config_name}')
        jax.clear_caches()
        train(config)


if __name__ == '__main__':
    app.run(main)
