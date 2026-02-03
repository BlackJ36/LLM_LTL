"""
mujoco_py compatibility shim
当代码尝试import mujoco_py时，提供新版mujoco的兼容API
"""
import warnings
import mujoco
import numpy as np
from typing import Optional, Dict, Any

warnings.warn(
    "mujoco_py is deprecated. Using mujoco compatibility layer.",
    DeprecationWarning,
    stacklevel=2
)


class MjSimState:
    """模拟mujoco_py的MjSimState"""
    def __init__(self, time, qpos, qvel, act, udd_state):
        self.time = time
        self.qpos = qpos.copy() if qpos is not None else np.array([])
        self.qvel = qvel.copy() if qvel is not None else np.array([])
        self.act = act.copy() if act is not None else np.array([])
        self.udd_state = udd_state if udd_state is not None else {}


class MjDataWrapper:
    """包装mujoco.MjData以提供mujoco-py兼容API"""
    def __init__(self, data, model):
        self._data = data
        self._model = model
        # 缓存速度数组 (per-forward cache, keyed by time)
        self._cached_site_vel = None
        self._cached_site_vel_time = -1.0
        self._cached_body_vel = None
        self._cached_body_vel_time = -1.0

    def __getattr__(self, name):
        return getattr(self._data, name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        elif hasattr(self._data, name):
            setattr(self._data, name, value)
        else:
            super().__setattr__(name, value)

    @property
    def body_xpos(self):
        """body位置数组 (mujoco-py兼容, 别名xpos)"""
        return self._data.xpos

    @property
    def body_xmat(self):
        """body旋转矩阵数组 (mujoco-py兼容, 别名xmat)"""
        return self._data.xmat

    @property
    def body_xquat(self):
        """body四元数数组 (mujoco-py兼容, 别名xquat)"""
        return self._data.xquat

    def _compute_site_velocities(self):
        """计算并缓存所有site的速度 (线速度和角速度一起计算)"""
        current_time = self._data.time
        if self._cached_site_vel is not None and self._cached_site_vel_time == current_time:
            return self._cached_site_vel

        nsite = self._model.nsite
        xvelp = np.zeros((nsite, 3))
        xvelr = np.zeros((nsite, 3))
        vel = np.zeros(6)
        for i in range(nsite):
            mujoco.mj_objectVelocity(self._model, self._data,
                                     mujoco.mjtObj.mjOBJ_SITE, i, vel, 0)
            xvelr[i] = vel[0:3]  # 角速度在前3个元素
            xvelp[i] = vel[3:6]  # 线速度在后3个元素

        self._cached_site_vel = (xvelp, xvelr)
        self._cached_site_vel_time = current_time
        return self._cached_site_vel

    @property
    def site_xvelp(self):
        """计算所有site的线速度 (mujoco-py兼容, 带缓存)"""
        return self._compute_site_velocities()[0]

    @property
    def site_xvelr(self):
        """计算所有site的角速度 (mujoco-py兼容, 带缓存)"""
        return self._compute_site_velocities()[1]

    def _compute_body_velocities(self):
        """计算并缓存所有body的速度 (线速度和角速度一起计算)"""
        current_time = self._data.time
        if self._cached_body_vel is not None and self._cached_body_vel_time == current_time:
            return self._cached_body_vel

        nbody = self._model.nbody
        xvelp = np.zeros((nbody, 3))
        xvelr = np.zeros((nbody, 3))
        vel = np.zeros(6)
        for i in range(nbody):
            mujoco.mj_objectVelocity(self._model, self._data,
                                     mujoco.mjtObj.mjOBJ_BODY, i, vel, 0)
            xvelr[i] = vel[0:3]
            xvelp[i] = vel[3:6]

        self._cached_body_vel = (xvelp, xvelr)
        self._cached_body_vel_time = current_time
        return self._cached_body_vel

    @property
    def body_xvelp(self):
        """计算所有body的线速度 (mujoco-py兼容, 带缓存)"""
        return self._compute_body_velocities()[0]

    @property
    def body_xvelr(self):
        """计算所有body的角速度 (mujoco-py兼容, 带缓存)"""
        return self._compute_body_velocities()[1]

    def get_site_xvelp_single(self, site_id):
        """获取单个site的线速度 (优化版，避免计算所有sites)"""
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self._model, self._data,
                                 mujoco.mjtObj.mjOBJ_SITE, site_id, vel, 0)
        return vel[3:6].copy()  # 线速度在后3个元素

    def get_site_xvelr_single(self, site_id):
        """获取单个site的角速度 (优化版，避免计算所有sites)"""
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self._model, self._data,
                                 mujoco.mjtObj.mjOBJ_SITE, site_id, vel, 0)
        return vel[0:3].copy()  # 角速度在前3个元素

    def get_site_xvel_single(self, site_id):
        """获取单个site的完整速度 [angular, linear] (优化版)"""
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self._model, self._data,
                                 mujoco.mjtObj.mjOBJ_SITE, site_id, vel, 0)
        return vel[3:6].copy(), vel[0:3].copy()  # (linear, angular)

    def get_site_jac_both(self, site_name):
        """获取site的位置和旋转雅可比矩阵 (优化版，一次调用)"""
        site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"Site '{site_name}' not found")
        nv = self._model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(self._model, self._data, jacp, jacr, site_id)
        return jacp.flatten(), jacr.flatten()

    def get_site_jacp(self, site_name):
        """获取site的位置雅可比矩阵 (mujoco-py兼容)"""
        site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"Site '{site_name}' not found")
        nv = self._model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(self._model, self._data, jacp, jacr, site_id)
        return jacp.flatten()

    def get_site_jacr(self, site_name):
        """获取site的旋转雅可比矩阵 (mujoco-py兼容)"""
        site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"Site '{site_name}' not found")
        nv = self._model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(self._model, self._data, jacp, jacr, site_id)
        return jacr.flatten()

    def get_body_jacp(self, body_name):
        """获取body的位置雅可比矩阵 (mujoco-py兼容)"""
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found")
        nv = self._model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacBody(self._model, self._data, jacp, jacr, body_id)
        return jacp.flatten()

    def get_body_jacr(self, body_name):
        """获取body的旋转雅可比矩阵 (mujoco-py兼容)"""
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found")
        nv = self._model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacBody(self._model, self._data, jacp, jacr, body_id)
        return jacr.flatten()

    def get_body_xpos(self, body_name):
        """获取body的位置 (mujoco-py兼容)"""
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found")
        return self._data.xpos[body_id].copy()

    def get_body_xmat(self, body_name):
        """获取body的旋转矩阵 (mujoco-py兼容)"""
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found")
        return self._data.xmat[body_id].copy()

    def get_body_xquat(self, body_name):
        """获取body的四元数 (mujoco-py兼容)"""
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found")
        return self._data.xquat[body_id].copy()

    def get_site_xpos(self, site_name):
        """获取site的位置 (mujoco-py兼容)"""
        site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"Site '{site_name}' not found")
        return self._data.site_xpos[site_id].copy()

    def get_site_xmat(self, site_name):
        """获取site的旋转矩阵 (mujoco-py兼容)"""
        site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"Site '{site_name}' not found")
        return self._data.site_xmat[site_id].copy()

    def get_geom_xpos(self, geom_name):
        """获取geom的位置 (mujoco-py兼容)"""
        geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id == -1:
            raise ValueError(f"Geom '{geom_name}' not found")
        return self._data.geom_xpos[geom_id].copy()

    def get_geom_xmat(self, geom_name):
        """获取geom的旋转矩阵 (mujoco-py兼容)"""
        geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id == -1:
            raise ValueError(f"Geom '{geom_name}' not found")
        return self._data.geom_xmat[geom_id].copy()

    def get_joint_qpos(self, joint_name):
        """获取关节位置 (mujoco-py兼容)"""
        joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Joint '{joint_name}' not found")
        addr = self._model.jnt_qposadr[joint_id]
        # 根据关节类型确定qpos的长度
        jnt_type = self._model.jnt_type[joint_id]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            return self._data.qpos[addr:addr+7].copy()
        elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
            return self._data.qpos[addr:addr+4].copy()
        else:
            return self._data.qpos[addr]

    def get_joint_qvel(self, joint_name):
        """获取关节速度 (mujoco-py兼容)"""
        joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Joint '{joint_name}' not found")
        addr = self._model.jnt_dofadr[joint_id]
        jnt_type = self._model.jnt_type[joint_id]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            return self._data.qvel[addr:addr+6].copy()
        elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
            return self._data.qvel[addr:addr+3].copy()
        else:
            return self._data.qvel[addr]

    def set_joint_qpos(self, joint_name, value):
        """设置关节位置 (mujoco-py兼容)"""
        joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Joint '{joint_name}' not found")
        addr = self._model.jnt_qposadr[joint_id]
        jnt_type = self._model.jnt_type[joint_id]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            self._data.qpos[addr:addr+7] = value
        elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
            self._data.qpos[addr:addr+4] = value
        else:
            self._data.qpos[addr] = value

    def set_joint_qvel(self, joint_name, value):
        """设置关节速度 (mujoco-py兼容)"""
        joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Joint '{joint_name}' not found")
        addr = self._model.jnt_dofadr[joint_id]
        jnt_type = self._model.jnt_type[joint_id]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            self._data.qvel[addr:addr+6] = value
        elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
            self._data.qvel[addr:addr+3] = value
        else:
            self._data.qvel[addr] = value


class MjModelWrapper:
    """包装mujoco.MjModel以提供mujoco-py兼容API"""
    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    def get_joint_qpos_addr(self, joint_name):
        """获取关节在qpos中的地址"""
        joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Joint '{joint_name}' not found")
        return self._model.jnt_qposadr[joint_id]

    def get_joint_qvel_addr(self, joint_name):
        """获取关节在qvel中的地址"""
        joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Joint '{joint_name}' not found")
        return self._model.jnt_dofadr[joint_id]

    def joint_name2id(self, joint_name):
        """获取关节ID"""
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

    def body_name2id(self, body_name):
        """获取body ID"""
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    def site_name2id(self, site_name):
        """获取site ID"""
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    def geom_name2id(self, geom_name):
        """获取geom ID"""
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

    def actuator_name2id(self, actuator_name):
        """获取actuator ID"""
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)

    def sensor_name2id(self, sensor_name):
        """获取sensor ID"""
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)

    def geom_id2name(self, geom_id):
        """获取geom名称 (mujoco-py兼容)"""
        return mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)

    def body_id2name(self, body_id):
        """获取body名称 (mujoco-py兼容)"""
        return mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id)

    def joint_id2name(self, joint_id):
        """获取joint名称 (mujoco-py兼容)"""
        return mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

    def site_id2name(self, site_id):
        """获取site名称 (mujoco-py兼容)"""
        return mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SITE, site_id)

    def actuator_id2name(self, actuator_id):
        """获取actuator名称 (mujoco-py兼容)"""
        return mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)

    def camera_name2id(self, camera_name):
        """获取camera ID (mujoco-py兼容)"""
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)


class MjSim:
    """
    模拟mujoco_py的MjSim类
    Wraps the new mujoco.MjModel and mujoco.MjData
    """
    def __init__(self, model):
        if isinstance(model, str):
            # 如果传入的是XML路径
            raw_model = mujoco.MjModel.from_xml_path(model)
        elif isinstance(model, mujoco.MjModel):
            raw_model = model
        else:
            # 假设是XML字符串
            raw_model = mujoco.MjModel.from_xml_string(model)

        self._raw_model = raw_model
        self.model = MjModelWrapper(raw_model)
        self.data = MjDataWrapper(mujoco.MjData(raw_model), raw_model)

        # 缓存常用属性
        self._nq = raw_model.nq
        self._nv = raw_model.nv
        self._nu = raw_model.nu

        # 渲染上下文 (mujoco-py兼容)
        self._render_context_offscreen = None
        self._render_context_window = None

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
        mujoco.mj_step(self._raw_model, self.data._data)

    def forward(self):
        """前向运动学计算"""
        mujoco.mj_forward(self._raw_model, self.data._data)

    def reset(self):
        """重置仿真状态"""
        mujoco.mj_resetData(self._raw_model, self.data._data)

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

    def add_render_context(self, render_context):
        """添加渲染上下文 (mujoco-py兼容)"""
        if render_context.offscreen:
            self._render_context_offscreen = render_context
        else:
            self._render_context_window = render_context

    def render(self, camera_name=None, width=None, height=None, depth=False, mode='offscreen'):
        """渲染相机图像 (mujoco-py兼容)

        Args:
            camera_name: 相机名称
            width: 图像宽度
            height: 图像高度
            depth: 是否返回深度图
            mode: 渲染模式 ('offscreen' 或 'window')

        Returns:
            如果 depth=False: RGB 图像 (height, width, 3)
            如果 depth=True: (RGB 图像, 深度图)
        """
        import cv2

        # 获取相机ID
        if camera_name is not None:
            camera_id = mujoco.mj_name2id(self._raw_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        else:
            camera_id = -1

        # 设置默认尺寸
        if width is None:
            width = 640
        if height is None:
            height = 480

        # 使用较小的渲染尺寸以避免帧缓冲区限制，然后放大
        max_size = 480
        scale_factor = 1.0
        render_width = width
        render_height = height

        if width > max_size or height > max_size:
            scale_factor = max(width, height) / max_size
            render_width = int(width / scale_factor)
            render_height = int(height / scale_factor)

        # 使用缓存的渲染器或创建新的
        if not hasattr(self, '_renderer_cache'):
            self._renderer_cache = {}

        cache_key = (render_height, render_width)
        if cache_key not in self._renderer_cache:
            try:
                self._renderer_cache[cache_key] = mujoco.Renderer(
                    self._raw_model, render_height, render_width
                )
            except ValueError:
                # 如果还是太大，使用更小的尺寸
                render_height = min(render_height, 256)
                render_width = min(render_width, 256)
                scale_factor = max(width / render_width, height / render_height)
                cache_key = (render_height, render_width)
                self._renderer_cache[cache_key] = mujoco.Renderer(
                    self._raw_model, render_height, render_width
                )

        renderer = self._renderer_cache[cache_key]
        renderer.update_scene(self.data._data, camera=camera_id)

        # 渲染RGB图像
        rgb = renderer.render()

        # 如果需要放大
        if scale_factor > 1.0:
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)

        if depth:
            # 渲染深度图
            renderer.enable_depth_rendering(True)
            renderer.update_scene(self.data._data, camera=camera_id)
            depth_img = renderer.render()
            renderer.enable_depth_rendering(False)

            if scale_factor > 1.0:
                depth_img = cv2.resize(depth_img, (width, height), interpolation=cv2.INTER_LINEAR)

            return rgb, depth_img
        else:
            return rgb


class _VisualOption:
    """模拟mujoco_py的可视化选项"""
    def __init__(self):
        # geomgroup 控制哪些geom组可见
        self.geomgroup = np.ones(6, dtype=np.uint8)
        self.flags = np.zeros(32, dtype=np.uint8)


class MjRenderContext:
    """模拟mujoco_py的渲染上下文"""
    def __init__(self, sim: MjSim, offscreen: bool = True, device_id: int = -1):
        self.sim = sim
        self.offscreen = offscreen
        self.device_id = device_id
        self._renderer = None
        # 可视化选项 (mujoco-py兼容)
        self.vopt = _VisualOption()

    def render(self, width: int, height: int, camera_id: int = -1):
        """渲染图像"""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.sim._raw_model, height, width)

        self._renderer.update_scene(self.sim.data._data, camera=camera_id)
        return self._renderer.render()


# 兼容旧名称
MjRenderContextOffscreen = MjRenderContext
MjRenderContextWindow = MjRenderContext


class MjViewer:
    """模拟mujoco_py的MjViewer (窗口渲染)"""
    def __init__(self, sim: MjSim):
        self.sim = sim
        self._viewer = None

    def render(self):
        """渲染到窗口"""
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.sim._raw_model, self.sim.data._data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


class MjViewerBasic(MjViewer):
    """兼容MjViewerBasic"""
    pass


def load_model_from_path(xml_path: str):
    """从XML文件加载模型"""
    return mujoco.MjModel.from_xml_path(xml_path)


def load_model_from_xml(xml_string: str):
    """从XML字符串加载模型"""
    return mujoco.MjModel.from_xml_string(xml_string)


def load_model_from_mjb(path: str):
    """从mjb加载模型"""
    return mujoco.MjModel.from_binary_path(path)


class _Functions:
    """兼容mujoco_py.functions"""
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
        type_map = {
            'body': mujoco.mjtObj.mjOBJ_BODY,
            'joint': mujoco.mjtObj.mjOBJ_JOINT,
            'geom': mujoco.mjtObj.mjOBJ_GEOM,
            'site': mujoco.mjtObj.mjOBJ_SITE,
            'camera': mujoco.mjtObj.mjOBJ_CAMERA,
            'actuator': mujoco.mjtObj.mjOBJ_ACTUATOR,
            'sensor': mujoco.mjtObj.mjOBJ_SENSOR,
        }
        type_id = type_map.get(type_name.lower(), mujoco.mjtObj.mjOBJ_UNKNOWN)
        return mujoco.mj_name2id(model, type_id, name)

    @staticmethod
    def mj_id2name(model, type_id, id):
        return mujoco.mj_id2name(model, type_id, id)


functions = _Functions()


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


# Fake cymj module (某些代码可能直接访问cymj)
class _CymjModule:
    """Fake cymj module - provides low-level mujoco-py compatibility"""

    @staticmethod
    def _mj_fullM(model, mass_matrix, qM):
        """计算完整质量矩阵 (mujoco-py兼容)

        注意: mujoco-py传入的mass_matrix是1D数组(nv^2,)
              新mujoco需要2D数组(nv, nv)
        """
        # 获取原始model (可能是wrapper)
        if hasattr(model, '_model'):
            raw_model = model._model
        else:
            raw_model = model

        nv = raw_model.nv
        # 创建2D临时数组
        mass_matrix_2d = np.zeros((nv, nv), dtype=np.float64)
        mujoco.mj_fullM(raw_model, mass_matrix_2d, qM)
        # 将结果复制回原始的1D数组
        mass_matrix[:] = mass_matrix_2d.flatten()

    @staticmethod
    def _mj_jac(model, data, jacp, jacr, point, body_id):
        """计算雅可比矩阵 (mujoco-py兼容)"""
        if hasattr(model, '_model'):
            raw_model = model._model
        else:
            raw_model = model
        if hasattr(data, '_data'):
            raw_data = data._data
        else:
            raw_data = data
        mujoco.mj_jac(raw_model, raw_data, jacp, jacr, point, body_id)


cymj = _CymjModule()

# builder模块兼容
class _BuilderModule:
    """Fake builder module"""
    MujocoException = MujocoException


builder = _BuilderModule()

# 导出兼容接口
__all__ = [
    'MjSim',
    'MjSimState',
    'MjRenderContext',
    'MjRenderContextOffscreen',
    'MjRenderContextWindow',
    'MjViewer',
    'MjViewerBasic',
    'load_model_from_path',
    'load_model_from_xml',
    'load_model_from_mjb',
    'functions',
    'MujocoException',
    'ignore_mujoco_warnings',
    'cymj',
    'builder',
]
