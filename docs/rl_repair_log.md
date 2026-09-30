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

## 2026-09-30：中指/无名指接触优化

- 用户指出成功 GIF 中中指没有参与抓取。真实接触力复测确认，旧策略在保持阶段只使用拇指、食指、小指；中指与无名指接触力均为 0N，而参考要求拇指、食指、中指、无名指。
- 根因：论文式接触奖励只匹配手指数，不匹配具体手指身份；旧三指方案的数量奖励仍为 0.8，替换参考手指不受额外惩罚。
- 新增显式 `contact_mode: per_finger`，以布尔接触状态的逐指 Hamming mismatch 替代数量 mismatch；新增 `require_reference_contact`，要求末帧接触状态逐指完全匹配。保留默认 count 模式与旧配置，以便区分论文原始形式与本地扩展。
- 用 512 个 GPU 物理环境、35 轮 CEM 联合搜索 20 个手指校正和 3 个腕位置校正；评分同时要求 ADD、参考四指接触、保持后真实抬升、桌面安全与接触力限制。
- 最终搜索候选真实峰值抬升 6.46cm、保持后抬升 5.09cm、ADD 5.83mm、最大接触力 17.07N；保持 30 帧接触覆盖率 `[1,1,1,1,0]`。
- 新配置 `wuji_hand2_apple_fingers.yaml` 与 `wuji_hand2_apple_fingers_dr.yaml`；旧成功模型及 GIF 保留。
- 环境新增每根手指实际接触力，评估输出逐指接触覆盖率；GIF 标注 T/I/M/R/P（拇指/食指/中指/无名指/小指），便于核实实际抓取。
- 新的独立验收除抬升≥4cm外，要求保持阶段参考四指各≥90% 帧有接触、小指≤10% 帧接触；不再仅凭数量或末帧成功宣称接触恢复。

### 四指策略阶段验收

- 无 DR 模型 `outputs/rl/lift_fingers_v2/checkpoint-final.pt` 已重新训练 1,048,576 步。独立 seed=101、256 回合：抬升成功与严格保持阶段逐指匹配均 256/256；四指保持覆盖率 `[1,1,1,1,0]`。
- 近景成功 GIF：`outputs/rl/diagnostics/lift_four_fingers_success.gif`。该 GIF 的中指末帧实际接触力 5.54N、无名指 15.13N，峰值抬升 6.42cm，保持后抬升 5.00cm，最终 ADD 6.63mm；对应 JSON 保存逐指接触覆盖率与力。
- 全部测试 35 passed，包括同样接触数量但替换具体手指时奖励必须下降的回归测试。
- 新四指姿态直接用于 DR 的零残差独立验收尚未通过：256 回合中严格通过仅 50，部分回合出现接触冲击超限。继续从无 DR 四指模型训练 DR，并用每候选 8 组随机物理场景进行鲁棒 CEM 搜索。
- 240Hz 仿真及较软 PD 的三个对照没有改进当前 DR 零残差成功率，保留 120Hz 和原 PD 参数；不靠放宽安全阈值提升成功率。

### 域随机化鲁棒姿态优化

- 四指 DR v2 从无 DR v2 warm start 训练 1,048,576 步，最后训练回合成功率约 99.1%；仍须独立逐指保持验收，不以训练统计作为最终结论。
- 鲁棒 CEM 每组 8 个不同随机物理环境使用同一候选修正，共 64 组/512 环境；以组内平均和最差评分各 50% 合成目标，35 轮搜索。这样避免单个候选偶然匹配某个质量或摩擦样本。
- 新配置 `wuji_hand2_apple_fingers_robust_dr.yaml`。其零残差独立 seed=707、256 回合已全部通过抬升与严格四指保持判据。
- 进一步从四指 DR v2 的第 20 轮检查点训练鲁棒 DR v3 1,048,576 步；保留此前全部检查点、配置与指标，独立输出目录 `outputs/rl/lift_fingers_robust_dr_v3`。

### 最终四指闭环验收

- 四指 DR v2 的独立 seed=303 验收也达到 256/256 严格通过，作为保留的中间模型。
- 最终鲁棒 DR v3：seed=707 与 seed=999，每组 256 回合，合计 **512/512** 同时满足完整抬升与保持、安全、ADD≤5cm、保持后抬升≥4cm、参考四指各≥90% 帧接触且小指≤10% 帧接触。机器可读结果与检查点 SHA256 已加入 `docs/rl_finger_validation_results.json`。
- 新 GIF `outputs/rl/diagnostics/lift_four_fingers_dr_success.gif`：seed=999，从保存的最终模型重新执行；中指末帧实际接触力 9.80N、无名指 17.02N，峰值抬升 6.67cm、最终抬升 5.22cm、最终 ADD 3.96mm；保持接触覆盖率 `[1,1,1,1,0]`。
- 默认训练配置改为无 DR 四指配置，诊断/渲染默认也改为四指配置；论文原始 count 模式仍可显式指定旧配置。README 更新四指模型的评估与 GIF 命令。
- 验收范围仍为第一次抬升和 1 秒保持；未将本次结果扩大解释为完整七次抓放或论文全部域随机化复现。
- 旧三指与新四指模型/GIF 均保留，本次改善没有降低接触力或桌面安全判据，也没有直接移动动态苹果以伪造接触。
- Git 关键节点本地提交 `ee221ec`（逐指奖励和姿态）、`412f22b`（逐指保持验收与近景 GIF）、`f394b0f`（鲁棒随机物理姿态）。远端仍未配置，无法推送。

```bash
# 新版无域随机化四指抬升 GIF
.venv/bin/python scripts/render_lift.py \
  --config configs/rl/wuji_hand2_apple_fingers.yaml \
  --checkpoint outputs/rl/lift_fingers_v2/checkpoint-final.pt --seed 101 \
  --output outputs/rl/diagnostics/lift_four_fingers_success.gif
# 新版域随机化四指抬升 GIF
.venv/bin/python scripts/render_lift.py \
  --config configs/rl/wuji_hand2_apple_fingers_robust_dr.yaml \
  --checkpoint outputs/rl/lift_fingers_robust_dr_v3/checkpoint-final.pt --seed 999 \
  --output outputs/rl/diagnostics/lift_four_fingers_dr_success.gif
```

## 2026-09-30：关节驱动、自碰撞与小指姿态审计

用户质疑小指翘起与仿真合理性。此前的抓取验收没有验证这些约束；本次审计发现旧成功轨迹不满足完整物理合理性要求，不能用于宣称硬件可执行。

### 旧设置的实际问题

- 逐物理子步读取 qpos/qvel：小指 MCP flex 峰值 11.49rad/s，超过 URDF 8.203rad/s；中指 MCP abd 峰值 12.58rad/s，超过 8.11rad/s。位置指令虽被限位夹紧，但并未设置引擎速度上限。
- 全关节 force_limit=5Nm；对 0.2Nm 外展轴为 25 倍，对 0.3Nm 指间轴为 16.67 倍，对 2Nm MCP flex 为 2.5 倍。
- 统一 Kp=200、Kd=10；资产 MJCF 的逐关节 Kp 约 0.18–0.69、Kd 约 0.008–0.031。旧参数并非厂家模型的驱动设置。
- 全手 collision ignore 共用同一位，自碰撞全部关闭。FCL 对 URDF 凸碰撞网格逐子步检测，非相邻食指/中指最大穿透约 5.03mm，部分小指/无名指也有穿透。
- 手部重力由 ManiSkill 的默认 passive-force balancing 自动关闭。未使用厂家 MJCF 的关节 armature。
- contact_offset 默认 20mm，远大于本任务的毫米级几何尺度；不能把接触力阈值与精确几何接触简单等同。
- 实际 SAPIEN joint order 与按手指分组的 URDF/reference order不同。旧偏置由搜索以运行时顺序得到，未按名称记录；本次新增 `grasp_bias_by_joint`，明确名称与值。小指 MCP flex 原始参考 0.826rad 被偏置减至 0.138rad，故明显伸展/翘起；这不是抓取所必需。

### 修复路径

- 新 `actuator_profile: asset` 读取固定版本 URDF effort/velocity，以及厂家 MJCF 的 Kp/Kd/armature；逐关节配置驱动，力矩上限不超过 URDF，恢复手部重力。域随机化对资产基准增益缩放，不再意外重设回旧统一 200/10/5。
- 开启非相邻刚体自碰撞。固定焊接的 distal/tip/sensor frame 按同一物理刚体处理，忽略真实相邻刚体与 MJCF 明确排除的腕部装配对；保留同指非相邻及跨指检测。通过 clique 位掩码覆盖排除图，每个环境均独立设置与校验，避免全手共享禁碰位。
- 使用硬件速度上限限制每控制步目标变化，并在每物理子步插值。发现 ManiSkill 原插值仅写 GPU buffer、BaseEnv 却在插值之前上传一次；新增控制器每子步主动上传插值目标。
- 接触裕量改为 1mm，480Hz 物理步进；30 次位置、4 次速度约束迭代。腕抓取校正渐变 15 步，不再一步跳变。
- 小指预偏置归零，跟踪原始 MMO 姿态；非接触小指策略残差限制为±0.05rad，停止用无约束搜索把它推向极端姿态。
- 新物理配置 `configs/rl/wuji_hand2_apple_physical.yaml`，旧检查点/配置保留用于审计对照。
- 新增非相邻自接触力指标与安全终止。重新调参必须在这些物理约束保持启用的条件下进行。

### 审计范围与证据

- 旧完整逐子步结果：`outputs/rl/diagnostics/physics_audit_legacy.json`；Git 保存摘要 `docs/rl_physics_audit_legacy.json`。
- 工具 `scripts/audit_hand_physics.py` 检查角度、实际速度、目标角度、逐轴驱动配置、重力、碰撞掩码、FCL 凸网格穿透；PD demand 明确标记为显式公式估计，不冒充 PhysX 实际隐式驱动力矩测量。
- URDF 31 个 link frame 中只有 26 个带碰撞几何。齿轮、腱绳、间隙、柔性内部组件没有建模，不能宣称它们的内部碰撞已被仿真验证。
- 官方参考：[ManiSkill 自定义机器人说明](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_robots.html)建议用具体排除对替代禁用全部自碰撞；[SAPIEN PD 驱动说明](https://sapien.ucsd.edu/docs/2.2/tutorial/robotics/pid.html)说明驱动刚度、阻尼与上限的含义。厂家参数的直接来源是本地 pinned `assets/wuji-description/hand2/hand2_beta2/body/{urdf/right.urdf,mjcf/right.xml}`。

### 物理修复后的参数与验证补充

- 仅使用厂家 1x PD 的初始恢复配置不能直接复用旧预偏置完成抓取；不回退自碰撞、重力或 effort/velocity 上限。
- 新调参配置 `wuji_hand2_apple_physical_tuned.yaml` 以厂家逐关节 PD 为基准，软件增益缩放 4x，仍严格保留各轴 URDF 力矩/速度上限与厂家 armature。这属于控制器调参，不是声称厂家实际伺服器恰好使用 4x 参数。
- 小指原始 MMO 初始/过程姿态也可能与邻指相冲突，改为固定、温和弯曲的休息目标：MCP flex=0.4rad、abd=0.05rad、PIP=0.4rad、DIP=0.2rad；策略允许±0.05rad修正。小指不再被自由 CEM 搜索，且自碰撞保持启用。
- GPU 实机加载检查：全部 4 个独立场景验证每关节 effort/velocity/armature、手部重力、逐对碰撞过滤以及小指动作边界；全部通过。
- 安全搜索拒绝曾经触发过失败的轨迹，即便后续还能抬升也不当作成功。新增逐根指尖到对象网格的连续距离评分，减少稀疏接触搜索中的无效候选。
