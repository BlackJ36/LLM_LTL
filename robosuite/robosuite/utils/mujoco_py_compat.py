"""
MuJoCo-py Compatibility Layer
兼容层：将新版mujoco API映射到mujoco_py风格的API

This module provides backward compatibility for code written for mujoco-py
when using the new DeepMind mujoco package.
"""
import mujoco
import numpy as np
from typing import Optional, Dict, Any


class MjSimState:
    """模拟mujoco_py的MjSimState"""
    def __init__(self, time, qpos, qvel, act, udd_state):
        self.time = time
        self.qpos = qpos.copy()
        self.qvel = qvel.copy()
        self.act = act.copy() if act is not None else np.array([])
        self.udd_state = udd_state


class MjSim:
    """
    模拟mujoco_py的MjSim类
    Wraps the new mujoco.MjModel and mujoco.MjData
    """
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)

        # 缓存常用属性
        self._nq = model.nq
        self._nv = model.nv
        self._nu = model.nu

    @property
    def nq(self):
        return self._nq

    @property
    def nv(self):
        return self._nv

    @property
    def nu(self):
        return self._nu

    def step(self):
        """执行一步仿真"""
        mujoco.mj_step(self.model, self.data)

    def forward(self):
        """前向运动学计算"""
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        """重置仿真状态"""
        mujoco.mj_resetData(self.model, self.data)

    def get_state(self) -> MjSimState:
        """获取当前状态"""
        return MjSimState(
            time=self.data.time,
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            act=self.data.act.copy() if self.data.act.size > 0 else None,
            udd_state={}
        )

    def set_state(self, state: MjSimState):
        """设置状态"""
        self.data.time = state.time
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel
        if state.act is not None and self.data.act.size > 0:
            self.data.act[:] = state.act

    def set_state_from_flattened(self, state_vec):
        """从扁平化状态向量设置状态"""
        self.data.qpos[:] = state_vec[:self._nq]
        self.data.qvel[:] = state_vec[self._nq:self._nq + self._nv]


class MjRenderContext:
    """模拟mujoco_py的渲染上下文"""
    def __init__(self, sim: MjSim, offscreen: bool = True):
        self.sim = sim
        self.offscreen = offscreen
        # 新版mujoco使用不同的渲染方式
        self._renderer = None

    def render(self, width: int, height: int, camera_id: int = -1):
        """渲染图像"""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.sim.model, height, width)

        self._renderer.update_scene(self.sim.data, camera_id)
        return self._renderer.render()


def load_model_from_path(xml_path: str) -> mujoco.MjModel:
    """从XML文件加载模型"""
    return mujoco.MjModel.from_xml_path(xml_path)


def load_model_from_xml(xml_string: str) -> mujoco.MjModel:
    """从XML字符串加载模型"""
    return mujoco.MjModel.from_xml_string(xml_string)


# 常用函数映射
def functions():
    """返回mujoco函数的命名空间（兼容mujoco_py.functions）"""
    class Functions:
        @staticmethod
        def mj_jac(model, data, jacp, jacr, point, body_id):
            mujoco.mj_jac(model, data, jacp, jacr, point, body_id)

        @staticmethod
        def mj_jacBody(model, data, jacp, jacr, body_id):
            mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

        @staticmethod
        def mj_jacSite(model, data, jacp, jacr, site_id):
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        @staticmethod
        def mj_name2id(model, type_name, name):
            type_id = getattr(mujoco.mjtObj, f'mjOBJ_{type_name.upper()}')
            return mujoco.mj_name2id(model, type_id, name)

        @staticmethod
        def mj_id2name(model, type_id, id):
            return mujoco.mj_id2name(model, type_id, id)

    return Functions()


# MujocoException兼容
class MujocoException(Exception):
    """兼容mujoco_py.MujocoException"""
    pass


# 上下文管理器
class ignore_mujoco_warnings:
    """兼容mujoco_py.ignore_mujoco_warnings"""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# 导出兼容接口
__all__ = [
    'MjSim',
    'MjSimState',
    'MjRenderContext',
    'load_model_from_path',
    'load_model_from_xml',
    'functions',
    'MujocoException',
    'ignore_mujoco_warnings',
]
