"""
Based on rllab's logger.

https://github.com/rll/rllab
"""
from enum import Enum
from contextlib import contextmanager
import numpy as np
import os
import os.path as osp
import sys
import datetime
import dateutil.tz
import csv
import json
import pickle
import errno
import torch

from maple.core.tabulate import tabulate
from collections import OrderedDict

# TensorBoard important metrics whitelist
# Only these metrics will be logged to TensorBoard to keep the dashboard clean
# All metrics are still saved to CSV for detailed analysis
#
# Metric naming convention:
#   - eval/  : evaluation rollouts (deterministic policy)
#   - expl/  : exploration rollouts (stochastic policy)
#   - trainer/: SAC training statistics
#
TB_IMPORTANT_METRICS = {
    # ============ 核心性能指标 ============
    # Returns = sum of rewards per episode, Mean/Std across episodes
    'eval/Returns Mean',       # 评估回报均值 - 最重要的性能指标
    'eval/Returns Std',        # 评估回报标准差 - 稳定性指标
    'expl/Returns Mean',       # 探索回报均值
    'expl/Returns Std',        # 探索回报标准差
    # Success rate at final step of episode
    'eval/env_infos/final/success Mean',  # 评估成功率 - 任务完成指标
    'expl/env_infos/final/success Mean',  # 探索成功率

    # ============ Affordance & Grasp 指标 ============
    # Affordance: 技能前置条件是否满足
    'eval/env_infos/final/aff_success Mean',  # 评估时affordance成功率
    'expl/env_infos/final/aff_success Mean',  # 探索时affordance成功率
    'eval/env_infos/final/aff_reward Mean',   # 评估时affordance奖励
    'expl/env_infos/final/aff_reward Mean',   # 探索时affordance奖励
    # Grasped: 是否成功抓取物体
    'eval/env_infos/final/grasped Mean',      # 评估时抓取成功率
    'expl/env_infos/final/grasped Mean',      # 探索时抓取成功率

    # ============ 训练损失 ============
    # QF Loss = MSE(Q_pred, r + γ * Q_target)
    'trainer/QF1 Loss',        # Q网络1的TD误差
    'trainer/QF2 Loss',        # Q网络2的TD误差
    # Policy Loss = E[α*log_π - Q]  (SAC目标: 最大化Q + 熵)
    'trainer/Policy Loss',     # 策略损失

    # ============ Alpha (熵系数) ============
    # Standard SAC: single alpha
    'trainer/Alpha',           # 熵系数 (标准SAC)
    'trainer/Alpha Loss',      # 熵系数损失 (标准SAC)
    # Hybrid SAC: separate alpha for skill (S) and parameter (P)
    'trainer/Alpha S',         # 技能选择的熵系数 (Hybrid SAC)
    'trainer/Alpha S Loss',    # 技能熵系数损失
    'trainer/Alpha P',         # 参数的熵系数 (Hybrid SAC)
    'trainer/Alpha P Loss',    # 参数熵系数损失

    # ============ Q 值监控（关键！）============
    # Q值应该稳定增长，不应发散或崩溃
    'trainer/Q1 Predictions Mean',  # Q1预测均值
    'trainer/Q2 Predictions Mean',  # Q2预测均值
    'trainer/Q Targets Mean',       # Q目标值均值 = r + γ * min(Q1', Q2')
    'trainer/Q1 Predictions Std',   # Q1预测标准差 - 过大表示不稳定
    'trainer/Q2 Predictions Std',   # Q2预测标准差
    'trainer/Q Targets Std',        # Q目标标准差

    # ============ Log π 监控 ============
    # log π 反映策略的熵，过小表示策略过于确定
    # Standard SAC
    'trainer/Log Pis Mean',    # log π 均值 (标准SAC)
    'trainer/Log Pis Std',     # log π 标准差
    # Hybrid SAC: separate for skill and parameter
    'trainer/Log Pis S Mean',  # 技能选择的 log π (Hybrid SAC)
    'trainer/Log Pis S Std',
    'trainer/Log Pis P Mean',  # 参数的 log π (Hybrid SAC)
    'trainer/Log Pis P Std',

    # ============ 训练进度 ============
    'Epoch',                   # 当前训练轮次
    'time/total (s)',          # 总训练时间（秒）
    'time/epoch (s)',          # 每轮时间（秒）
    'expl/num steps total',    # 总环境交互步数 - 常用作x轴
    'replay_buffer/size',      # 经验回放缓冲区大小

    # ============ 路径统计 ============
    'expl/Num Paths',          # 探索的轨迹数量
    'expl/path length Mean',   # 探索轨迹平均长度
    'eval/path length Mean',   # 评估轨迹平均长度

    # ============ VLM 奖励指标 (步级) ============
    # VLM 二元判断: 动作是否合理
    'expl/env_infos/vlm_binary_reasonable Mean',   # VLM认为合理的比例
    'expl/env_infos/vlm_binary_confidence Mean',   # VLM判断置信度
    # VLM 进度评估
    'expl/env_infos/vlm_progress Mean',            # VLM估计的任务进度 (0-1)
    # VLM 奖励分解
    'expl/env_infos/vlm_binary_reward Mean',       # 二元判断奖励
    'expl/env_infos/vlm_progress_reward Mean',     # 进度奖励
    'expl/env_infos/vlm_total_reward Mean',        # VLM总奖励 (未缩放)
    'expl/env_infos/vlm_reward_scaled Mean',       # VLM奖励 (已缩放)
    # 评估时的VLM指标 (步级)
    'eval/env_infos/vlm_binary_reasonable Mean',
    'eval/env_infos/vlm_progress Mean',
    'eval/env_infos/vlm_total_reward Mean',

    # ============ VLM 轨迹级指标 (final) ============
    # 探索轨迹的VLM统计
    'expl/env_infos/final/vlm_traj_reward_sum Mean',      # 轨迹VLM奖励总和
    'expl/env_infos/final/vlm_traj_reward_mean Mean',     # 轨迹VLM奖励均值
    'expl/env_infos/final/vlm_traj_reasonable_rate Mean', # 轨迹合理动作比例
    'expl/env_infos/final/vlm_traj_confidence_mean Mean', # 轨迹平均置信度
    'expl/env_infos/final/vlm_traj_progress_final Mean',  # 轨迹最终进度
    'expl/env_infos/final/vlm_traj_progress_max Mean',    # 轨迹最大进度
    'expl/env_infos/final/vlm_traj_progress_delta Mean',  # 轨迹进度变化
    'expl/env_infos/final/vlm_traj_eval_count Mean',      # 轨迹VLM评估次数
    # 评估轨迹的VLM统计
    'eval/env_infos/final/vlm_traj_reward_sum Mean',
    'eval/env_infos/final/vlm_traj_reward_mean Mean',
    'eval/env_infos/final/vlm_traj_reasonable_rate Mean',
    'eval/env_infos/final/vlm_traj_progress_final Mean',
    'eval/env_infos/final/vlm_traj_progress_max Mean',
    'eval/env_infos/final/vlm_traj_progress_delta Mean',

    # ============ VLM 优先级采样指标 ============
    # Replay buffer 优先级统计
    'replay_buffer/vlm_priority/mean_reasonable',  # 缓冲区中的平均合理率
    'replay_buffer/vlm_priority/mean_confidence',  # 平均置信度
    'replay_buffer/vlm_priority/mean_progress',    # 平均进度
    'replay_buffer/vlm_priority/mean_priority',    # 平均优先级
    'replay_buffer/vlm_priority/beta',             # 当前重要性采样系数
    # 训练时的重要性采样权重
    'trainer/IS/weights_mean',                     # IS权重均值
    'trainer/IS/weights_std',                      # IS权重标准差
    'trainer/VLM/batch_priority_mean',             # 批次VLM优先级均值
    'trainer/VLM/batch_reasonable_mean',           # 批次合理率均值
}

def add_prefix(log_dict: OrderedDict, prefix: str, divider=''):
    with_prefix = OrderedDict()
    for key, val in log_dict.items():
        with_prefix[prefix + divider + key] = val
    return with_prefix


def append_log(log_dict, to_add_dict, prefix=None):
    if prefix is not None:
        to_add_dict = add_prefix(to_add_dict, prefix=prefix)
    return log_dict.update(to_add_dict)


class TerminalTablePrinter(object):
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os
        rows, columns = os.popen('stty size', 'r').read().split()
        tabulars = self.tabulars[-(int(rows) - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Logger(object):
    def __init__(self):
        self._prefixes = []
        self._prefix_str = ''

        self._tabular_prefixes = []
        self._tabular_prefix_str = ''

        self._tabular = []
        self._tabular_keys = {}

        self._text_outputs = []
        self._tabular_outputs = []

        self._text_fds = {}
        self._tabular_fds = {}
        self._tabular_header_written = set()

        self._snapshot_dir = None
        self._snapshot_mode = 'all'
        self._snapshot_gap = 1

        self._log_tabular_only = False
        self._header_printed = False
        self.table_printer = TerminalTablePrinter()

        # TensorBoard support
        self._tb_writer = None
        self._tb_step = 0

    def reset(self):
        self.__init__()

    def _add_output(self, file_name, arr, fds, mode='a'):
        if file_name not in arr:
            mkdir_p(os.path.dirname(file_name))
            arr.append(file_name)
            fds[file_name] = open(file_name, mode)

    def _remove_output(self, file_name, arr, fds):
        if file_name in arr:
            fds[file_name].close()
            del fds[file_name]
            arr.remove(file_name)

    def push_prefix(self, prefix):
        self._prefixes.append(prefix)
        self._prefix_str = ''.join(self._prefixes)

    def add_text_output(self, file_name):
        self._add_output(file_name, self._text_outputs, self._text_fds,
                         mode='a')

    def remove_text_output(self, file_name):
        self._remove_output(file_name, self._text_outputs, self._text_fds)

    def add_tabular_output(self, file_name, relative_to_snapshot_dir=False):
        if relative_to_snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
        self._add_output(file_name, self._tabular_outputs, self._tabular_fds,
                         mode='w')
        self._tabular_keys[file_name] = None

    def remove_tabular_output(self, file_name, relative_to_snapshot_dir=False):
        if relative_to_snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
        if self._tabular_fds[file_name] in self._tabular_header_written:
            self._tabular_header_written.remove(self._tabular_fds[file_name])
        self._remove_output(file_name, self._tabular_outputs, self._tabular_fds)

    def set_snapshot_dir(self, dir_name):
        self._snapshot_dir = dir_name

    def get_snapshot_dir(self, ):
        return self._snapshot_dir

    def get_snapshot_mode(self, ):
        return self._snapshot_mode

    def set_snapshot_mode(self, mode):
        self._snapshot_mode = mode

    def get_snapshot_gap(self, ):
        return self._snapshot_gap

    def set_snapshot_gap(self, gap):
        self._snapshot_gap = gap

    def set_log_tabular_only(self, log_tabular_only):
        self._log_tabular_only = log_tabular_only

    def get_log_tabular_only(self, ):
        return self._log_tabular_only

    def set_tensorboard(self, log_dir):
        """Initialize TensorBoard SummaryWriter.

        Args:
            log_dir: Directory where TensorBoard event files will be written.
        """
        from torch.utils.tensorboard import SummaryWriter
        self._tb_writer = SummaryWriter(log_dir)

    def set_tb_step(self, step):
        """Set the global step for TensorBoard logging.

        Args:
            step: The current global step (e.g., total environment steps).
        """
        self._tb_step = step

    def close_tensorboard(self):
        """Close TensorBoard writer if it exists."""
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None

    def log(self, s, with_prefix=True, with_timestamp=True):
        out = s
        if with_prefix:
            out = self._prefix_str + out
        if with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            out = "%s | %s" % (timestamp, out)
        if not self._log_tabular_only:
            # Also log to stdout
            print(out)
            for fd in list(self._text_fds.values()):
                fd.write(out + '\n')
                fd.flush()
            sys.stdout.flush()

    def record_tabular(self, key, val):
        self._tabular.append((self._tabular_prefix_str + str(key), str(val)))

    def record_dict(self, d, prefix=None):
        if prefix is not None:
            self.push_tabular_prefix(prefix)
        for k, v in d.items():
            self.record_tabular(k, v)
        if prefix is not None:
            self.pop_tabular_prefix()

    def push_tabular_prefix(self, key):
        self._tabular_prefixes.append(key)
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def pop_tabular_prefix(self, ):
        del self._tabular_prefixes[-1]
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def save_extra_data(self, data, file_name='extra_data.pkl', mode='joblib'):
        """
        Data saved here will always override the last entry

        :param data: Something pickle'able.
        """
        file_name = osp.join(self._snapshot_dir, file_name)
        if mode == 'joblib':
            import joblib
            joblib.dump(data, file_name, compress=3)
        elif mode == 'pickle':
            pickle.dump(data, open(file_name, "wb"))
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        return file_name

    def get_table_dict(self, ):
        return dict(self._tabular)

    def get_table_key_set(self, ):
        return set(key for key, value in self._tabular)

    @contextmanager
    def prefix(self, key):
        self.push_prefix(key)
        try:
            yield
        finally:
            self.pop_prefix()

    @contextmanager
    def tabular_prefix(self, key):
        self.push_tabular_prefix(key)
        yield
        self.pop_tabular_prefix()

    def log_variant(self, log_file, variant_data):
        mkdir_p(os.path.dirname(log_file))
        with open(log_file, "w") as f:
            json.dump(variant_data, f, indent=2, sort_keys=True, cls=MyEncoder)

    def record_tabular_misc_stat(self, key, values, placement='back'):
        if placement == 'front':
            prefix = ""
            suffix = key
        else:
            prefix = key
            suffix = ""
        if len(values) > 0:
            self.record_tabular(prefix + "Average" + suffix, np.average(values))
            self.record_tabular(prefix + "Std" + suffix, np.std(values))
            self.record_tabular(prefix + "Median" + suffix, np.median(values))
            self.record_tabular(prefix + "Min" + suffix, np.min(values))
            self.record_tabular(prefix + "Max" + suffix, np.max(values))
        else:
            self.record_tabular(prefix + "Average" + suffix, np.nan)
            self.record_tabular(prefix + "Std" + suffix, np.nan)
            self.record_tabular(prefix + "Median" + suffix, np.nan)
            self.record_tabular(prefix + "Min" + suffix, np.nan)
            self.record_tabular(prefix + "Max" + suffix, np.nan)

    def dump_tabular(self, *args, **kwargs):
        wh = kwargs.pop("write_header", None)
        if len(self._tabular) > 0:
            if self._log_tabular_only:
                self.table_printer.print_tabular(self._tabular)
            else:
                for line in tabulate(self._tabular).split('\n'):
                    self.log(line, *args, **kwargs)
            tabular_dict = dict(self._tabular)
            # Also write to the csv files
            for filename, tabular_fd in list(self._tabular_fds.items()):
                # Only saves keys in first iteration to CSV!
                # (But every key is printed out in text)
                itr0_keys = self._tabular_keys.get(filename)
                if itr0_keys is None:
                    itr0_keys = list(sorted(tabular_dict.keys()))
                    self._tabular_keys[filename] = itr0_keys
                else:
                    prev_keys = set(itr0_keys)
                    curr_keys = set(tabular_dict.keys())
                    if curr_keys != prev_keys:
                        print("Warning: CSV key mismatch")
                        print("extra keys in 0th iter", prev_keys - curr_keys)
                        print("extra keys in cur iter", curr_keys - prev_keys)

                writer = csv.DictWriter(tabular_fd,
                                        fieldnames=itr0_keys,
                                        extrasaction="ignore",)
                if wh or (
                        wh is None and tabular_fd not in self._tabular_header_written):
                    writer.writeheader()
                    self._tabular_header_written.add(tabular_fd)
                writer.writerow(tabular_dict)
                tabular_fd.flush()

            # Write to TensorBoard if enabled (only important metrics)
            if self._tb_writer is not None:
                # Use 'expl/num steps total' as x-axis if available, else use _tb_step
                step = self._tb_step
                if 'expl/num steps total' in tabular_dict:
                    try:
                        step = int(float(tabular_dict['expl/num steps total']))
                    except (ValueError, TypeError):
                        pass

                for key, val in tabular_dict.items():
                    # Only log important metrics to reduce TensorBoard clutter
                    if key not in TB_IMPORTANT_METRICS:
                        continue
                    # Skip non-numeric values
                    try:
                        float_val = float(val)
                        # Skip NaN values
                        if not np.isnan(float_val):
                            self._tb_writer.add_scalar(key, float_val, step)
                    except (ValueError, TypeError):
                        pass
                self._tb_writer.flush()

            del self._tabular[:]

    def pop_prefix(self, ):
        del self._prefixes[-1]
        self._prefix_str = ''.join(self._prefixes)

    def save_itr_params(self, itr, params):
        if self._snapshot_dir:
            if self._snapshot_mode == 'all':
                file_name = osp.join(self._snapshot_dir, 'itr_%d.pkl' % itr)
                torch.save(params, file_name)
            elif self._snapshot_mode == 'last':
                # override previous params
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
                torch.save(params, file_name)
            elif self._snapshot_mode == "gap":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir, 'itr_%d.pkl' % itr)
                    torch.save(params, file_name)
            elif self._snapshot_mode == "gap_and_last":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir, 'itr_%d.pkl' % itr)
                    torch.save(params, file_name)
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
                torch.save(params, file_name)
            elif self._snapshot_mode == 'none':
                pass
            else:
                raise NotImplementedError


logger = Logger()

