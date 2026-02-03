from functools import partial

import numpy as np
import copy

create_rollout_function = partial


def multitask_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        observation_key=None,
        desired_goal_key=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
):
    if full_o_postprocess_func:
        def wrapped_fun(env, agent, o):
            full_o_postprocess_func(env, agent, observation_key, o)
    else:
        wrapped_fun = None

    def obs_processor(o):
        return np.hstack((o[observation_key], o[desired_goal_key]))

    paths = rollout(
        env,
        agent,
        max_path_length=max_path_length,
        render=render,
        render_kwargs=render_kwargs,
        get_action_kwargs=get_action_kwargs,
        preprocess_obs_for_policy_fn=obs_processor,
        full_o_postprocess_func=wrapped_fun,
    )
    if not return_dict_obs:
        paths['observations'] = paths['observations'][observation_key]
    return paths


def contextual_rollout(
        env,
        agent,
        observation_key=None,
        context_keys_for_policy=None,
        obs_processor=None,
        **kwargs
):
    if context_keys_for_policy is None:
        context_keys_for_policy = ['context']

    if not obs_processor:
        def obs_processor(o):
            combined_obs = [o[observation_key]]
            for k in context_keys_for_policy:
                combined_obs.append(o[k])
            return np.concatenate(combined_obs, axis=0)
    paths = rollout(
        env,
        agent,
        preprocess_obs_for_policy_fn=obs_processor,
        **kwargs
    )
    return paths


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        reset_callback=None,
        addl_info_func=None,
        image_obs_in_info=False,
        last_step_is_terminal=False,
        terminals_all_false=False,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda env, agent, o: o
    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    addl_infos = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(env, agent, o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        if addl_info_func:
            addl_infos.append(addl_info_func(env, agent, o, a))

        next_o, r, d, env_info = env.step(copy.deepcopy(a), image_obs_in_info=image_obs_in_info)

        new_path_length = path_length + env_info.get('num_ac_calls', 1)

        if new_path_length > max_path_length:
            break
        path_length = new_path_length

        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        if terminals_all_false:
            terminals.append(False)
        else:
            terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)

    path_length_actions = np.sum(
        [info.get('num_ac_calls', 1) for info in env_infos]
    )

    reward_actions_sum = np.sum(
        [info.get('reward_actions', 0) for info in env_infos]
    )

    if last_step_is_terminal:
        terminals[-1] = True

    skill_names = []
    sc = env.env.skill_controller
    for i in range(len(actions)):
        ac = actions[i]
        skill_name = sc.get_skill_name_from_action(ac)
        skill_names.append(skill_name)
        success = env_infos[i].get('success', False)
        if success:
            break

    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        addl_infos=addl_infos,
        full_observations=raw_obs,
        full_next_observations=raw_obs,
        path_length=path_length,
        path_length_actions=path_length_actions,
        reward_actions_sum=reward_actions_sum,
        skill_names=skill_names,
        max_path_length=max_path_length,
    )


def deprecated_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def vectorized_rollout(
        vec_env,
        agent,
        max_path_length=np.inf,
        get_action_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        terminals_all_false=False,
        last_step_is_terminal=False,
):
    """
    Perform vectorized rollout across multiple parallel environments.

    Collects paths from all environments until each completes at least one
    full episode (done=True or max_path_length reached).

    Args:
        vec_env: Vectorized environment (SubprocVecEnv or DummyVecEnv)
        agent: Policy that implements get_actions(obs_batch)
        max_path_length: Maximum steps per episode
        get_action_kwargs: Additional kwargs for get_actions
        preprocess_obs_for_policy_fn: Optional function to preprocess observations
        terminals_all_false: If True, set all terminal flags to False
        last_step_is_terminal: If True, set last step terminal to True

    Returns:
        List of path dicts, one per completed episode
    """
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda env, agent, o: o

    num_envs = vec_env.num_envs

    # Per-environment path buffers
    path_buffers = [
        {
            'raw_obs': [],
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': [],
            'next_observations': [],
            'agent_infos': [],
            'env_infos': [],
            'path_length': 0,
            'path_length_actions': 0,
            'reward_actions_sum': 0,
        }
        for _ in range(num_envs)
    ]

    # Track which envs have completed at least one episode
    completed_paths = []
    envs_completed = [False] * num_envs

    # Get skill controllers if available (for skill_names tracking)
    try:
        skill_controllers = vec_env.get_skill_controllers()
    except:
        skill_controllers = [None] * num_envs

    # Reset all environments and agent
    agent.reset()
    obs_batch = vec_env.reset()

    while not all(envs_completed):
        # Preprocess observations for policy
        obs_for_agent = np.stack([
            preprocess_obs_for_policy_fn(None, agent, obs_batch[i])
            for i in range(num_envs)
        ])

        # Batch action inference (GPU accelerated)
        actions_batch = agent.get_actions(obs_for_agent, **get_action_kwargs)

        # Handle case where get_actions returns (actions, agent_infos)
        if isinstance(actions_batch, tuple):
            actions_batch, agent_info_list = actions_batch
        else:
            agent_info_list = [{} for _ in range(num_envs)]

        # Ensure actions_batch is numpy array
        if not isinstance(actions_batch, np.ndarray):
            actions_batch = np.array(actions_batch)

        # Step all environments
        next_obs_batch, rewards_batch, dones_batch, infos_batch = vec_env.step(
            copy.deepcopy(actions_batch)
        )

        # Update path buffers for each environment
        for i in range(num_envs):
            if envs_completed[i]:
                continue

            buf = path_buffers[i]

            # Get num_ac_calls from env_info for variable-length actions
            num_ac_calls = infos_batch[i].get('num_ac_calls', 1)
            new_path_length = buf['path_length'] + num_ac_calls

            # Check if this step would exceed max_path_length
            if new_path_length > max_path_length:
                # Finalize this path without adding this step
                if len(buf['observations']) > 0:
                    path = _finalize_path(
                        buf, skill_controllers[i],
                        terminals_all_false, last_step_is_terminal, max_path_length
                    )
                    completed_paths.append(path)
                envs_completed[i] = True
                continue

            # Store transition
            buf['raw_obs'].append(obs_batch[i])
            buf['observations'].append(obs_batch[i])
            buf['actions'].append(actions_batch[i])
            buf['rewards'].append(rewards_batch[i])
            buf['next_observations'].append(next_obs_batch[i])
            buf['agent_infos'].append(
                agent_info_list[i] if isinstance(agent_info_list, list) else {}
            )
            buf['env_infos'].append(infos_batch[i])

            if terminals_all_false:
                buf['terminals'].append(False)
            else:
                buf['terminals'].append(dones_batch[i])

            buf['path_length'] = new_path_length
            buf['path_length_actions'] += num_ac_calls
            buf['reward_actions_sum'] += infos_batch[i].get('reward_actions', 0)

            # Check if episode done
            if dones_batch[i] or new_path_length >= max_path_length:
                path = _finalize_path(
                    buf, skill_controllers[i],
                    terminals_all_false, last_step_is_terminal, max_path_length
                )
                completed_paths.append(path)
                envs_completed[i] = True

                # Reset this environment's buffer for potential next episode
                path_buffers[i] = {
                    'raw_obs': [],
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'terminals': [],
                    'next_observations': [],
                    'agent_infos': [],
                    'env_infos': [],
                    'path_length': 0,
                    'path_length_actions': 0,
                    'reward_actions_sum': 0,
                }

        # Update observations for next iteration
        obs_batch = next_obs_batch

        # Reset environments that finished but we still need to track
        for i in range(num_envs):
            if dones_batch[i] and not envs_completed[i]:
                obs_batch[i] = vec_env.reset_at(i)

    return completed_paths


def _finalize_path(buf, skill_controller, terminals_all_false, last_step_is_terminal, max_path_length):
    """Convert path buffer to final path dict format."""
    actions = np.array(buf['actions'])
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)

    observations = np.array(buf['observations'])
    next_observations = np.array(buf['next_observations'])

    rewards = np.array(buf['rewards'])
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)

    terminals = buf['terminals']
    if last_step_is_terminal and len(terminals) > 0:
        terminals[-1] = True

    # Get skill names if skill controller is available
    skill_names = []
    if skill_controller is not None:
        for i in range(len(actions)):
            ac = actions[i]
            try:
                skill_name = skill_controller.get_skill_name_from_action(ac)
                skill_names.append(skill_name)
            except:
                skill_names.append('unknown')
            # Check for success
            success = buf['env_infos'][i].get('success', False)
            if success:
                break

    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=buf['agent_infos'],
        env_infos=buf['env_infos'],
        addl_infos=[],
        full_observations=buf['raw_obs'],
        full_next_observations=buf['raw_obs'],
        path_length=buf['path_length'],
        path_length_actions=buf['path_length_actions'],
        reward_actions_sum=buf['reward_actions_sum'],
        skill_names=skill_names,
        max_path_length=max_path_length,
    )
