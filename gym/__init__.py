"""
gym 兼容性模块

将 gymnasium API 重新导出为 gym 接口，用于支持旧版代码。
"""

# 直接从 gymnasium 重新导出所有内容
from gymnasium import *
from gymnasium import spaces
from gymnasium import Env

# 重新导出 core 模块内容
from gymnasium import core
