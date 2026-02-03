"""
Vectorized environment wrappers for parallel sampling.

Uses subprocess-based parallelization to run multiple environments simultaneously,
enabling batch action inference on GPU for improved training throughput.
"""
import multiprocessing as mp
from multiprocessing import connection, shared_memory
from typing import Callable, List, Tuple, Any, Optional
import numpy as np
import cloudpickle
import struct
import json


def _extract_obs(reset_result):
    """Extract observation from reset result (handles gym vs gymnasium API)."""
    if isinstance(reset_result, tuple):
        return reset_result[0]  # gymnasium returns (obs, info)
    return reset_result  # gym returns just obs


def _extract_step(step_result):
    """Extract step outputs (handles gym vs gymnasium API)."""
    if len(step_result) == 5:
        # gymnasium: (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, terminated or truncated, info
    else:
        # gym: (obs, reward, done, info)
        return step_result


def _send_raw(conn, msg_type: str, data: bytes):
    """Send raw bytes with header to avoid pickle.

    Protocol:
    - 4 bytes: message type length
    - N bytes: message type (utf-8)
    - 4 bytes: data length
    - M bytes: data
    """
    msg_type_bytes = msg_type.encode('utf-8')
    header = struct.pack('I', len(msg_type_bytes)) + msg_type_bytes + struct.pack('I', len(data))
    conn.send_bytes(header + data)


def _recv_raw(conn) -> Tuple[str, bytes]:
    """Receive raw bytes with header."""
    # Receive all data at once
    raw = conn.recv_bytes()

    # Parse header
    type_len = struct.unpack('I', raw[:4])[0]
    msg_type = raw[4:4+type_len].decode('utf-8')
    data_len = struct.unpack('I', raw[4+type_len:8+type_len])[0]
    data = raw[8+type_len:8+type_len+data_len]

    return msg_type, data


def _encode_obs(obs) -> bytes:
    """Encode observation as raw bytes."""
    arr = np.array(obs, copy=True, dtype=np.float64)
    arr = np.ascontiguousarray(arr)
    # Header: shape dimensions count, shape values, then data
    shape = arr.shape
    header = struct.pack('I', len(shape)) + struct.pack(f'{len(shape)}I', *shape)
    return header + arr.tobytes()


def _decode_obs(data: bytes) -> np.ndarray:
    """Decode raw bytes to observation array."""
    ndim = struct.unpack('I', data[:4])[0]
    shape = struct.unpack(f'{ndim}I', data[4:4+4*ndim])
    arr_data = data[4+4*ndim:]
    return np.frombuffer(arr_data, dtype=np.float64).reshape(shape)


def _encode_step_result(obs, reward: float, done: bool, info: dict) -> bytes:
    """Encode step result as raw bytes."""
    obs_bytes = _encode_obs(obs)
    # Sanitize info to JSON-serializable format
    info_clean = _sanitize_info_for_json(info)
    info_json = json.dumps(info_clean).encode('utf-8')

    # Pack: obs_len, obs_bytes, reward, done, info_len, info_bytes
    result = (
        struct.pack('I', len(obs_bytes)) + obs_bytes +
        struct.pack('d', reward) +  # double for reward
        struct.pack('?', done) +    # bool for done
        struct.pack('I', len(info_json)) + info_json
    )
    return result


def _decode_step_result(data: bytes) -> Tuple[np.ndarray, float, bool, dict]:
    """Decode step result from raw bytes."""
    offset = 0

    # Observation
    obs_len = struct.unpack('I', data[offset:offset+4])[0]
    offset += 4
    obs = _decode_obs(data[offset:offset+obs_len])
    offset += obs_len

    # Reward
    reward = struct.unpack('d', data[offset:offset+8])[0]
    offset += 8

    # Done
    done = struct.unpack('?', data[offset:offset+1])[0]
    offset += 1

    # Info
    info_len = struct.unpack('I', data[offset:offset+4])[0]
    offset += 4
    info_json = data[offset:offset+info_len].decode('utf-8')
    info = json.loads(info_json)

    return obs, reward, done, info


def _sanitize_info_for_json(info: dict) -> dict:
    """Convert info dict to JSON-serializable format."""
    result = {}
    for k, v in info.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            result[k] = float(v)
        elif isinstance(v, np.bool_):
            result[k] = bool(v)
        elif isinstance(v, (int, float, bool, str, type(None))):
            result[k] = v
        elif isinstance(v, dict):
            result[k] = _sanitize_info_for_json(v)
        elif isinstance(v, (list, tuple)):
            result[k] = [
                x.tolist() if isinstance(x, np.ndarray)
                else float(x) if isinstance(x, (np.floating, np.integer))
                else x
                for x in v
            ]
        else:
            try:
                result[k] = float(v)
            except (TypeError, ValueError):
                pass  # Skip non-serializable
    return result


def _worker(
    remote: connection.Connection,
    parent_remote: connection.Connection,
    env_fn_wrapper: bytes,
) -> None:
    """Worker process that runs a single environment.

    Uses raw bytes protocol to avoid mujoco_py pickle issues.

    Args:
        remote: Connection to receive commands from parent
        parent_remote: Parent's connection (closed in worker)
        env_fn_wrapper: Cloudpickle-serialized environment factory function
    """
    import torch
    torch.set_num_threads(1)

    parent_remote.close()

    try:
        env_fn = cloudpickle.loads(env_fn_wrapper)
        env = env_fn()
    except Exception as e:
        import traceback
        error_msg = f"Failed to create env: {e}\n{traceback.format_exc()}"
        _send_raw(remote, "error", error_msg.encode('utf-8'))
        remote.close()
        return

    try:
        while True:
            # Receive command using raw protocol
            msg_type, cmd_data = _recv_raw(remote)

            if msg_type == "step":
                # Decode action from bytes
                action = _decode_obs(cmd_data)  # Actions use same encoding as obs
                result = env.step(action)
                obs, reward, done, info = _extract_step(result)
                # Send step result using raw bytes
                result_bytes = _encode_step_result(obs, reward, done, info)
                _send_raw(remote, "step_result", result_bytes)

            elif msg_type == "reset":
                try:
                    result = env.reset()
                    obs = _extract_obs(result)
                    obs_bytes = _encode_obs(obs)
                    _send_raw(remote, "obs", obs_bytes)
                except Exception as e:
                    import traceback
                    error_msg = f"Reset failed: {e}\n{traceback.format_exc()}"
                    _send_raw(remote, "error", error_msg.encode('utf-8'))

            elif msg_type == "close":
                env.close()
                remote.close()
                break

            elif msg_type == "get_spaces":
                # Spaces need pickle, but this is called once at init before mujoco issues
                remote.send((env.observation_space, env.action_space))

            elif msg_type == "get_attr":
                attr_name = cmd_data.decode('utf-8')
                remote.send(getattr(env, attr_name, None))

            else:
                raise NotImplementedError(f"Unknown command: {msg_type}")

    except EOFError:
        pass
    except Exception as e:
        import traceback
        try:
            error_msg = f"Worker exception: {e}\n{traceback.format_exc()}"
            _send_raw(remote, "error", error_msg.encode('utf-8'))
        except:
            pass


class CloudpickleWrapper:
    """Wrapper that uses cloudpickle for serialization."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def __getstate__(self):
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob):
        self.fn = cloudpickle.loads(ob)

    def __call__(self):
        return self.fn()


class SubprocVecEnv:
    """Vectorized environment using subprocesses for parallelization.

    Each environment runs in its own subprocess, enabling true parallel
    execution. Actions are sent to all environments simultaneously, and
    results are collected in batch.

    Attributes:
        num_envs: Number of parallel environments
        observation_space: Observation space (from first env)
        action_space: Action space (from first env)
    """

    def __init__(
        self,
        env_fns: List[Callable],
        start_method: str = "spawn",
    ):
        """Initialize vectorized environment.

        Args:
            env_fns: List of callables that create environments
            start_method: Multiprocessing start method ('spawn', 'fork', 'forkserver')
        """
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False

        # Use spawn to avoid MuJoCo/OpenGL issues with fork
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[
            ctx.Pipe() for _ in range(self.num_envs)
        ])

        self.processes = []
        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, env_fns
        ):
            # Serialize env_fn with cloudpickle for lambda/closure support
            env_fn_bytes = cloudpickle.dumps(env_fn)
            process = ctx.Process(
                target=_worker,
                args=(work_remote, remote, env_fn_bytes),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
            work_remote.close()

        # Get spaces from first environment using raw protocol
        _send_raw(self.remotes[0], "get_spaces", b"")
        self.observation_space, self.action_space = self.remotes[0].recv()

        # Cache for skill controllers (lazily loaded)
        self._skill_controllers: Optional[List] = None

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Execute actions in all environments.

        Args:
            actions: Array of shape (num_envs, action_dim)

        Returns:
            observations: Array of shape (num_envs, obs_dim)
            rewards: Array of shape (num_envs,)
            dones: Array of shape (num_envs,)
            infos: List of info dicts, one per environment
        """
        self._assert_not_closed()

        # Send actions using raw protocol
        for remote, action in zip(self.remotes, actions):
            action_bytes = _encode_obs(action)
            _send_raw(remote, "step", action_bytes)

        # Receive results using raw protocol
        observations = []
        rewards = []
        dones = []
        infos = []

        for remote in self.remotes:
            msg_type, result_bytes = _recv_raw(remote)
            obs, reward, done, info = _decode_step_result(result_bytes)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(observations),
            np.array(rewards),
            np.array(dones),
            infos,
        )

    def step_async(self, actions: np.ndarray) -> None:
        """Send actions to all environments without waiting for results."""
        self._assert_not_closed()

        for remote, action in zip(self.remotes, actions):
            action_bytes = _encode_obs(action)
            _send_raw(remote, "step", action_bytes)
        self.waiting = True

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Wait for all environments to complete their steps."""
        self._assert_not_closed()
        self.waiting = False

        observations = []
        rewards = []
        dones = []
        infos = []

        for remote in self.remotes:
            msg_type, result_bytes = _recv_raw(remote)
            obs, reward, done, info = _decode_step_result(result_bytes)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(observations),
            np.array(rewards),
            np.array(dones),
            infos,
        )

    def reset(self) -> np.ndarray:
        """Reset all environments.

        Returns:
            observations: Array of shape (num_envs, obs_dim)
        """
        self._assert_not_closed()

        # Send reset commands using raw protocol
        for remote in self.remotes:
            _send_raw(remote, "reset", b"")

        # Receive observations using raw protocol
        observations = []
        for i, remote in enumerate(self.remotes):
            try:
                msg_type, obs_bytes = _recv_raw(remote)
                if msg_type == "error":
                    raise RuntimeError(f"Worker {i} error: {obs_bytes.decode('utf-8')}")
                obs = _decode_obs(obs_bytes)
                observations.append(obs)
            except Exception as e:
                # Check if process is still alive
                if not self.processes[i].is_alive():
                    raise RuntimeError(f"Worker {i} process died unexpectedly") from e
                raise

        return np.stack(observations)

    def reset_at(self, index: int) -> np.ndarray:
        """Reset a specific environment.

        Args:
            index: Index of environment to reset

        Returns:
            observation: Observation from reset environment
        """
        self._assert_not_closed()

        _send_raw(self.remotes[index], "reset", b"")
        msg_type, obs_bytes = _recv_raw(self.remotes[index])
        return _decode_obs(obs_bytes)

    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List:
        """Get attribute from environments.

        Args:
            attr_name: Name of attribute to get
            indices: Optional list of environment indices

        Returns:
            List of attribute values
        """
        self._assert_not_closed()

        if indices is None:
            indices = range(self.num_envs)

        for i in indices:
            _send_raw(self.remotes[i], "get_attr", attr_name.encode('utf-8'))

        return [self.remotes[i].recv() for i in indices]

    def get_skill_controllers(self) -> List:
        """Get skill controllers from all environments.

        Note: Skill controllers contain mujoco references that can't be pickled
        across processes, so we return None for SubprocVecEnv.
        """
        # Can't pickle mujoco-based skill controllers across processes
        return [None] * self.num_envs

    def close(self) -> None:
        """Close all environments and terminate subprocesses."""
        if self.closed:
            return

        if self.waiting:
            for remote in self.remotes:
                try:
                    _recv_raw(remote)
                except:
                    pass

        for remote in self.remotes:
            try:
                _send_raw(remote, "close", b"")
            except:
                pass

        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

        self.closed = True

    def _assert_not_closed(self) -> None:
        """Raise error if environment is closed."""
        if self.closed:
            raise RuntimeError("Attempting to use closed VecEnv")

    def __len__(self) -> int:
        return self.num_envs

    def __del__(self):
        if hasattr(self, 'closed') and not self.closed:
            self.close()


class DummyVecEnv:
    """Vectorized environment that runs environments sequentially.

    Useful for debugging and testing without subprocess overhead.
    """

    def __init__(self, env_fns: List[Callable]):
        """Initialize dummy vectorized environment.

        Args:
            env_fns: List of callables that create environments
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def _extract_obs(self, reset_result):
        """Extract observation from reset result (handles gym vs gymnasium API)."""
        if isinstance(reset_result, tuple):
            return reset_result[0]  # gymnasium returns (obs, info)
        return reset_result  # gym returns just obs

    def _extract_step(self, step_result):
        """Extract step outputs (handles gym vs gymnasium API)."""
        if len(step_result) == 5:
            # gymnasium: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = step_result
            return obs, reward, terminated or truncated, info
        else:
            # gym: (obs, reward, done, info)
            return step_result

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Execute actions in all environments sequentially."""
        results = [self._extract_step(env.step(action)) for env, action in zip(self.envs, actions)]
        observations, rewards, dones, infos = zip(*results)

        return (
            np.stack(observations),
            np.array(rewards),
            np.array(dones),
            list(infos),
        )

    def reset(self) -> np.ndarray:
        """Reset all environments."""
        observations = [self._extract_obs(env.reset()) for env in self.envs]
        return np.stack(observations)

    def reset_at(self, index: int) -> np.ndarray:
        """Reset a specific environment."""
        return self._extract_obs(self.envs[index].reset())

    def get_skill_controllers(self) -> List:
        """Get skill controllers from all environments."""
        controllers = []
        for env in self.envs:
            inner = getattr(env, 'env', env)
            controllers.append(getattr(inner, 'skill_controller', None))
        return controllers

    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()

    def __len__(self) -> int:
        return self.num_envs
