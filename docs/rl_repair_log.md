# RL 抓取闭环修复记录

## 2026-09-30：项目与基线审计

- 项目流水线：GRAB/MANO → 形态匹配 → 接触匹配 → 热扩散机器人目标 → 浮动腕 IK → ManiSkill 残差 PPO。当前对象为 Wuji Hand2 + apple_lift。
- 论文第五节要求物体 ADD 与接触奖励相乘；安全终止与完整轨迹最终 ADD≤5cm、有接触共同决定成功。
- 现有 stable 训练到 2,752,512 步，成功率仍为零，主要在参考约 220 步触发 ADD EMA 终止。不能将旧提交标题中的 stable 当成成功证据。
- 先运行真实物理完整轨迹诊断：记录实际/参考高度、最大抬升、ADD、接触力及历史失败。后续以完整轨迹且从起始状态执行为验收依据。
- 仓库没有 Git remote；关键节点可以本地提交，推送需要实际远端地址。用户原有 outputs/morphology_initialization.json 删除保持原状。

### 已定位的实现缺陷

1. 重置物体只设位姿，未清零线速度/角速度；反复回合或搜索时携带上个回合的动量。修复为同时重置物体和腕根速度。
2. ManiSkill 的 `_after_control_step` 在 GPU fetch 之前调用，原实现用旧 GPU 缓存更新 ADD EMA。现主动 fetch 后计算 EMA。
3. 原始 651 帧包含七次抓放，结尾参考手已离开物体，而成功判据要求最终接触。需要明确选择抓取子任务终点，不能用原始结尾训练并期待最终接触成功。
4. 单独搜索 20 个手指偏置仍不抬升；联合搜索毫米级腕位置校正后，首次候选已出现约 6.2cm 的真实物理抬升、最大接触力约 16N。仍需修复重置后重新验证。

### 抓取闭环与无域随机化训练

- 新参考校正：采用 CEM 仿真搜索的 20 个关节预偏置，加世界坐标腕偏移 `[+15.609, -11.656, -14.706] mm`，按原接触 lookahead gate 激活。原始 MMO 文件保持原状；环境内参考腕数组显式 clone，避免训练/渲染共享 reference 被重复叠加校正。
- 验收子任务：原始 30Hz 参考前 77 帧（0–76），然后重复最后位姿 30 帧，模拟 1 秒悬空保持，共 107 帧。它验证第一个抓取与抬升，**不代表原始 651 帧七次抓放轨迹全部成功**。
- PPO：256 环境、128 步 rollout、1,048,576 timesteps，学习率 1e-4，初始 log std -3。奖励依旧为论文 ADD×接触奖励，保持原安全阈值与 5cm 最终 ADD 判据；额外验收要求保持结束时真实高度相对初始化增加≥4cm。
- 修正成功统计：ManiSkill BaseEnv 会把 success 当作 terminated；本环境显式以 fail 为失败终止、参考末帧为 truncation，避免成功回合被诊断记录成失败。
- 保存无 DR 模型：`outputs/rl/lift_fixed_v1/checkpoint-final.pt`。
- 独立 seed=101，256 回合：256/256 通过论文判据且保持后≥4cm；平均峰值抬升 6.65cm、平均最终抬升 5.12cm、最小最终抬升 4.97cm、平均最终 ADD 1.53cm、全程最大接触力 14.36N、最小桌面间距 7.18mm。

### 随后的域随机化阶段

- 从无 DR 检查点 warm start，再训练 1,048,576 timesteps；优化器在新阶段重新初始化。
- 启用桌面 XY 各±5cm、yaw±15°、各轴外力±0.05N；质量、物体/手摩擦、PD 增益分别在基准的 [0.8,1.2] 范围随机化。
- 物理参数在每个场景建立时采样，在回合间保持；初始位置/yaw、外力重新采样。评估使用不同 seed 重建场景，验证新物理样本，而非复用训练的静态物理样本。
- 手部模型没有手臂，腕轨迹以物体初始位置/偏航为锚同步变换。这是带特权物体锚点的手部抓取鲁棒性验证，尚未验证固定手臂起点下独立物体位置扰动的可达性，也未实现论文全部各向异性尺度、质心、外力矩及随机起始时刻实验；不宣称完整论文域随机化复现。
- 保存 DR 模型：`outputs/rl/lift_dr_v1/checkpoint-final.pt`。
- 独立 seed=303，256 回合：256/256 通过论文判据且保持后≥4cm；平均峰值抬升 6.50cm、平均最终抬升 4.95cm、最小最终抬升 4.70cm、平均最终 ADD 1.52cm、全程最大接触力 22.07N、最小桌面间距 7.21mm。
- 机器可读摘要：`docs/rl_validation_results.json`；逐帧高度/接触/ADD 及每环境结果在 `outputs/rl/diagnostics/*final_eval.json`。

### 验证与成功 GIF

- 全部测试：34 passed；真实 GPU 检查验证了物体动量重置、局部环境重置、当前步 ADD EMA 和参考数组不被污染。
- 成功 GIF 使用保存检查点在 GPU PhysX 重新执行，不对动态物体施加位姿跟随。每帧标注实际高度、抬升量、ADD、接触手指数及终止原因；对应 JSON 记录真实成功判据。
- `outputs/rl/diagnostics/lift_success.gif` 与 `lift_dr_success.gif`。

### 复现命令（在 mmo_sharpa 下执行）

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m morphim_mmo.cli.rl.train \
  --num-envs 256 --total-timesteps 1048576 --gif-interval 0 \
  --output-dir outputs/rl/new_lift_run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m morphim_mmo.cli.rl.train \
  --config configs/rl/wuji_hand2_apple_dr.yaml \
  --checkpoint outputs/rl/new_lift_run/checkpoint-final.pt \
  --num-envs 256 --total-timesteps 1048576 --gif-interval 0 \
  --output-dir outputs/rl/new_dr_run
.venv/bin/python scripts/diagnose_lift.py \
  --config configs/rl/wuji_hand2_apple_dr.yaml \
  --checkpoint outputs/rl/lift_dr_v1/checkpoint-final.pt --num-envs 256 --seed 303 \
  --output outputs/rl/diagnostics/new_dr_eval.json
.venv/bin/python scripts/render_lift.py \
  --config configs/rl/wuji_hand2_apple_dr.yaml \
  --checkpoint outputs/rl/lift_dr_v1/checkpoint-final.pt --seed 303 \
  --output outputs/rl/diagnostics/new_dr_success.gif
```

所有数值参数为本地 Wuji + apple 的复现调参结果，不是论文公开参数。模型、原始逐帧指标、GIF 保留本地 outputs，按仓库忽略规则不加入 Git；配置、代码、摘要与本文档加入 Git。仓库无远端配置，无法执行远端推送。
