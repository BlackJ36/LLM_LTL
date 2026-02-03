from maple.launchers.launcher_util import run_experiment
from maple.launchers.robosuite_launcher import experiment
import maple.util.hyperparameter as hyp
import os.path as osp
import argparse
import json
import collections.abc
import copy

from maple.launchers.conf import LOCAL_LOG_DIR

base_variant = dict(
    algorithm_kwargs=dict(
        eval_only=True,
        num_epochs=5000,
        eval_epoch_freq=100,
    ),
    replay_buffer_size=int(1E2),
    vis_expl=False,
    dump_video_kwargs=dict(
        rows=1,
        columns=6,
        pad_length=5,
        pad_color=0,
    ),
    num_eval_rollouts=50,

    ckpt_epoch=75, #### evaluate itr_75 checkpoint ###
)

# VLM evaluation config
VLM_CONFIG = dict(
    enabled=False,
    api_base='http://172.19.1.40:8001',
    model_name='Qwen/Qwen3-VL-8B-Instruct',
    eval_frequency=5,  # VLM evaluates every 5 steps
    vlm_reward_scale=0.5,
    binary_weight=0.5,
    progress_weight=0.5,
    camera_name='frontview',
    camera_height=256,
    camera_width=256,
    image_history_size=4,
    save_vlm_results=True,
)

# Task descriptions for VLM
TASK_DESCRIPTIONS = {
    'Door': 'Open the door by pushing the handle',
    'Lift': 'Lift the cube off the table',
    'Stack': 'Stack the red cube on top of the green cube',
    'PickPlace': 'Pick up the object and place it in the bin',
    'NutAssembly': 'Pick up the nut and place it on the peg',
    'PegInHole': 'Insert the peg into the hole',
    'Cleanup': 'Clean up the objects by placing them in the correct locations',
}

env_params = dict(
    lift={
        'ckpt_path': [
            'lift/01-19-test/01-19-test_2026_01_19_17_07_28_0000--s-45612',  # itr_150, 最完整
            'lift/01-19-test/01-19-test_2026_01_19_17_07_02_0000--s-90494',  # itr_75
            'lift/01-19-6env-20ep/01-19-6env-20ep_2026_01_19_16_04_12_0000--s-72154',  # 6env
        ],
    },
    door={
        'ckpt_path': [
            'door/01-19-test/01-19-test_2026_01_19_20_49_18_0000--s-8654',  # itr_975, 最新
        ],
        'ckpt_epoch': [975],  # 使用最新checkpoint
    },
    door_vlm={
        'ckpt_path': [
            'vlm-maple-Door/01-30-vlm0.5/01-30-vlm0.5_2026_01_30_05_46_00_0000--s-0',  # VLM trained
        ],
        'ckpt_epoch': [200],  # itr_200
    },
    pnp={
        'ckpt_path': [
            ### Add paths here ###
        ],
    },
    wipe={
        'ckpt_path': [
            ### Add paths here ###
        ],
    },
    stack={
        'ckpt_path': [
            ### Add paths here ###
        ],
    },
    nut_round={
        'ckpt_path': [
            ### Add paths here ###
        ],
    },
    cleanup={
        'ckpt_path': [
            ### Add paths here ###
        ],
    },
    peg_ins={
        'ckpt_path': [
            ### Add paths here ###
        ],
    },
)

def process_variant(eval_variant, vlm_config=None):
    ckpt_path = eval_variant['ckpt_path']
    json_path = osp.join(LOCAL_LOG_DIR, ckpt_path, 'variant.json')
    with open(json_path) as f:
        ckpt_variant = json.load(f)
    deep_update(ckpt_variant, eval_variant)
    variant = copy.deepcopy(ckpt_variant)

    if args.debug:
        mpl = variant['algorithm_kwargs']['max_path_length']
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = mpl * 3
        variant['dump_video_kwargs']['rows'] = 1
        variant['dump_video_kwargs']['columns'] = 2
    else:
        mpl = variant['algorithm_kwargs']['max_path_length']
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = mpl * variant['num_eval_rollouts']

    variant['save_video_period'] = variant['algorithm_kwargs']['eval_epoch_freq']

    if args.no_video:
        variant['save_video'] = False

    variant['exp_label'] = args.label

    # Add VLM config if enabled
    if vlm_config and vlm_config.get('enabled', False):
        env_type = variant.get('env_variant', {}).get('env_type', args.env)
        # Handle env names like 'door_vlm' -> 'Door'
        env_type_clean = env_type.replace('_vlm', '').capitalize()
        task_desc = vlm_config.get('task_description') or TASK_DESCRIPTIONS.get(env_type_clean, 'Complete the task')

        variant['vlm_variant'] = {
            'enabled': True,
            'task_description': task_desc,
            'api_base': vlm_config['api_base'],
            'model_name': vlm_config['model_name'],
            'vlm_reward_scale': vlm_config['vlm_reward_scale'],
            'eval_frequency': vlm_config['eval_frequency'],
            'binary_weight': vlm_config['binary_weight'],
            'progress_weight': vlm_config['progress_weight'],
            'camera_name': vlm_config['camera_name'],
            'camera_height': vlm_config['camera_height'],
            'camera_width': vlm_config['camera_width'],
            'image_history_size': vlm_config['image_history_size'],
            'reward_mode': 'bonus_only',
            'warmup_steps': 0,  # No warmup for evaluation
            'use_async': False,  # Sync mode for evaluation
            'save_vlm_results': vlm_config.get('save_vlm_results', True),
        }
        print(f"[VLM] Enabled for evaluation")
        print(f"[VLM] Task: {task_desc}")
        print(f"[VLM] Eval frequency: every {vlm_config['eval_frequency']} steps")

    return variant

def deep_update(source, overrides):
    '''
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    Copied from: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    '''
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--label', type=str, default='test')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--no_video', action='store_true')
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--first_variant', action='store_true')
    # VLM arguments
    parser.add_argument('--vlm', action='store_true', help='Enable VLM evaluation')
    parser.add_argument('--vlm-api', type=str, default='http://172.19.1.40:8001')
    parser.add_argument('--vlm-freq', type=int, default=5, help='VLM eval frequency (steps)')
    parser.add_argument('--task-desc', type=str, default=None, help='Task description for VLM')
    args = parser.parse_args()

    # Update VLM config if enabled
    if args.vlm:
        VLM_CONFIG['enabled'] = True
        VLM_CONFIG['api_base'] = args.vlm_api
        VLM_CONFIG['eval_frequency'] = args.vlm_freq
        if args.task_desc:
            VLM_CONFIG['task_description'] = args.task_desc

    search_space = env_params[args.env]
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=base_variant,
    )
    for exp_id, eval_variant in enumerate(sweeper.iterate_hyperparameters()):
        variant = process_variant(eval_variant, VLM_CONFIG if args.vlm else None)

        # Use VLM launcher if VLM enabled
        if args.vlm:
            from llm_ltl.vlm.vlm_maple_launcher import experiment as vlm_experiment
            experiment_fn = vlm_experiment
        else:
            experiment_fn = experiment

        run_experiment(
            experiment_fn,
            exp_folder=args.env,
            exp_prefix=args.label,
            variant=variant,
            snapshot_mode='gap_and_last',
            snapshot_gap=200,
            exp_id=exp_id,
            use_gpu=(not args.no_gpu),
            gpu_id=args.gpu_id,
        )

        if args.first_variant:
            exit()