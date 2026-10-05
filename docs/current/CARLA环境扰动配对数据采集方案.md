# CARLA环境扰动配对数据采集方案与脚本说明

## 当前最终基准：以用户指定的spvc_carla.py为准

本节替代下方此前关于genGANData、固定截图矩形和sensor的操作建议。已直接读取CARLA机器 `/home/ab/carla/artical-F122/Aebs/connect/spvc_carla.py`（147行，SHA256 `e7057c7715fcd582d19ee7fedc86eabee25ca403e9969cd4c61f6a7cadf159b1`），未修改这个用户原文件。

新增入口 `Aebs.connect.collect_spvc_expanded`。保留原脚本：`available_maps[3]`加载地图，`spawn_points[4]`放车，Tesla、默认红色/晴天，spectator沿实际车身朝向后退，z_offset=2，等待0.3秒，mss主显示器monitors[1]全屏截图，BGRA转BGR，OpenCV缩放640×640。地图名以实际服务端返回为准，不按原注释猜Town01，也不强制Town10。没有固定桌面裁剪、没有相机传感器、没有调FOV、没有多道路或姿态扩展。

仅扩充数据：两段np.linspace每段默认1000点（共2000），可选4天气×3颜色（共24000）。默认仍是原晴天红车；支持200点/段恢复原400条距离。原脚本10米重复项保留，两项属于同一配对组。附加metadata不改变图片：天气、颜色、时间、实际车辆坐标、distance group、0.25米bin、实际地图/显示器分辨率与完成统计。labels.csv仍为filename,distance_m两列。输出新目录，不覆盖旧数据。

在CARLA可见的桌面终端，激活carla_env，保持CARLA全屏。先可用200点/段检查原构图，再正式采集：

```bash
cd /home/ab/carla/artical-F122
python -m Aebs.connect.collect_spvc_expanded \
  --reload-original-map --points-per-segment 1000 \
  --weathers original overcast low_sun rain \
  --colors 255,0,0 40,80,160 220,220,220 \
  --output Aebs/carla_data_spvc_expanded/full01
```

此命令重载地图，清空当前仿真场景，必须专用实例；不要与其他采集同时运行。纯SSH不保证截到正确桌面。开始有8秒切换全屏时间，采集期间不要让其他窗口遮挡CARLA。天气切换会额外等待稳定，但每次移动后的0.3秒不变。原异步截图没有图像与世界状态严格同帧保证，本次不自行改采集协议。

训练前按距离区间分组（不只按单张随机分），同距离的天气/颜色全留在同一个split；bin仅是可用分组标签，脚本没有自动划分或宣称各组独立。这批扩展同一场景的距离和外观，不是多场景泛化数据，也不是误差契约证明。24k张不等于24k独立状态。

交付验证：新增4项离线测试，全部采集测试共23项通过；dry-run核对24000计划。已同步两台服务器，没有重载地图、没有实际截图或实验。用户的spvc_carla.py保留不动；不要再运行此前collect_original_expanded作为当前采集入口。

## 当前入口更正：恢复原实验桌面截图方式

用户要求只扩充数据，不改变原实验采集方式。此前 `collect_latent_alignment` 改成sensor.camera.rgb、固定FOV90、多spawn锚点，并不等价于原genGANData的spectator+桌面裁剪。即便距离标签相同，画面中车辆大小仍可能不同。此前Town10预览完成只说明该新流程可运行，不证明其与原实验一致；暂停把这类数据并入原实验，不删除已有文件。下方传感器版本说明仅保留历史参考，不是当前推荐入口。

新入口 `Aebs.connect.collect_original_expanded` 保留：原地图选择available_maps[5]、目标坐标(42.846504,-193.132416,0.275307)、yaw1.5度、相机相对高度2米、沿车辆朝向向后移动、spectator视角、mss截屏(top50,left40,width2520,height1550)、OpenCV BGRA转BGR及640×640缩放、移动后等待0.3秒、原天气及红色Tesla默认值。不关闭车辆物理，不新增sensor、不改FOV、不自动选新道路。

默认仅将每段距离采样从200增至1000，即400张增至2000张，仍为5～10和10～16两个含端点等距序列，保留10米重复项。增加近邻距离并不等同于增加独立场景，也不保证latent泛化改善。用 `--points-per-segment 200` 可恢复原400条距离计划。

原版的地图重载会清空运行场景，新脚本要求显式 `--reload-original-map` 授权；使用专用CARLA实例。地图数组第5项取决于服务端，config记录实际地图，不把Town10或原坐标可用性当成已确认事实。如果该点生成失败应核对原地图，不擅自换场景。首次8秒等待仍用于调整窗口。必须在CARLA可见的桌面终端、原窗口大小/位置/分辨率/缩放下执行；纯SSH可能没有DISPLAY，或截取了错误桌面。不要为“车看起来更近”修改距离标签；先复现原视口和截图区域。原图中近处只见部分车身不再作为必须换镜头的理由。

```bash
cd /home/ab/carla/artical-F122
python -m Aebs.connect.collect_original_expanded --dry-run
python -m Aebs.connect.collect_original_expanded \
  --reload-original-map --points-per-segment 1000 \
  --output Aebs/carla_data_original_expanded/run01
```

先确认基准图与原实验构图一致，再考虑只扩充外观：`--weathers original overcast low_sun rain --colors 255,0,0 40,80,160 220,220,220`，共24,000张（2000×4×3），不改相机/截图链条；可选天气变更后等待8秒，避免截到过渡画面。默认不启用这些外观变化。此版沿用原异步截图，不声称传感器同帧或严格物理同步保证。

输出新目录中的编号PNG、兼容原两列格式labels.csv、附带距离配对组/天气/颜色的samples.jsonl、config.json、summary.json。不覆盖旧400张，也不自动创建随机train/test划分；同距离的所有外观与重复10米项必须留在同一数据划分。多外观时需调整后续加载/分组，不能直接逐行随机划分或将所有新增数据当测试集。

验证：4项原版采样/参数单元测试新增，整个采集测试集共19项通过；没有实际执行桌面截图或重载CARLA。原genGANData.py文件未修改。

更新：2026-10-04。本文件对应新增采集工具，不修改现有DINO、PPO或SBC训练入口。

## 最新：用于减少两条latent误差的密集配对采集（2026-10-04）

### Town10实际检查后的下一步

预览故障记录：首轮town10_preview01在第一张出现相机/目标距离不匹配；远端summary显示saved_samples=0、cleanup_errors=[]，没有错误标签图片入库。旧实现spawn后立即通过actor.get_transform定位相机，尚无新tick，存在读取客户端旧快照的时序风险；当前日志缺少坐标，尚不能断言这就是唯一根因。

修复：新目标生成并关闭物理后先推进一帧，清空至该帧的传感器数据，从同帧world snapshot读取目标transform，然后再摆相机。仍保留1厘米距离检查；再次失败会写pose_mismatch.json，包含请求/实际距离、定位时目标坐标、最终目标快照、请求/图像相机位姿。新增2项同步测试，总计15项通过。未自动启动重采；用户需使用新的town10_preview02目录重跑，保留preview01故障记录。

用户回传已确认CARLA 0.9.15、地图 `Carla/Maps/Town10HD_Opt`、155个spawn点（0～154），连接无报错。这仅证明地图查询正常，不证明场景可采。

新增 `latent_alignment_scenes.town10.candidate.json`：基于坐标分布选取0、6、15、26、30、89作为训练候选，19、36作为验证候选，21、149作为测试候选。尚未检查道路后方连续性、遮挡和背景重叠，划分也需审查，`reviewed_for_capture` 仍为false。

新增 `--preview`：每候选点仅在5、8、16米各采一张晴天、红车、基准相机图，共30张。允许未审核配置，但不允许和smoke/分批混用；写入用途标记、不导出训练CSV，也禁止后续用export把预览变成训练数据。预览仍要求空闲专用CARLA实例，不删除已有actor、不重载地图。

在CARLA机器的carla_env中执行：

```bash
cd /home/ab/carla/artical-F122
python -m Aebs.connect.collect_latent_alignment \
  --scenes Aebs/connect/latent_alignment_scenes.town10.candidate.json \
  --expected-map Town10HD_Opt --preview \
  --output Aebs/carla_latent_alignment/town10_preview01
```

检查输出rgb目录内30张图：目标是否完整可见、是否漂浮/穿模、相机是否位于道路、不同split是否背景高度重叠。反馈summary和图片后再确定正式配置。预览未实际运行；本地新增预览测试后共13项离线测试通过。不要只因为预览采完就将正式数据质量视为通过。

本节是当前运行入口。下方第1～10节保留旧版采集器说明；不要混淆两版默认规模。

### 已做与未做

- [x] 阅读原始 `genGANData.py` 和现有 `collect_semantic_pairs.py`，复用同帧传感器、实测距离、文件审计和异常清理。
- [x] 新增 `Aebs/connect/collect_latent_alignment.py`，默认密集距离、多外观、按配对组分批、分split CSV导出。
- [x] 新增 `latent_alignment_scenes.example.json`（6训练、2验证、2测试场景占位），实际采集前必须人工审核。
- [x] 本地12项离线测试通过；全量计划预览164,160张RGB。
- [ ] 在运行CARLA的机器选择真实道路锚点并试采；本次没有连接CARLA或生成新图像，也没有同步服务器。
- [ ] 实采、人工检查、审计、提取冻结DINO特征、用配对组训练表示。

### 采什么，为什么

| 维度 | 默认配置 | 意义 |
| --- | --- | --- |
| 距离 | 5～16米每0.1米；5.5～7.5米额外每0.025米，共171个去重距离 | 全范围覆盖并加密当前任务近距离区域；不是仅采旧验证失败点 |
| 普通天气 | 晴天、阴天、低太阳、另一太阳方向、小雨、雨，共6组 | 同状态不同光照/湿润环境 |
| 相机 | 高度2米基准、1.7米、2.3米、基准高度左右各2度yaw，共5组 | 单因素安装变化，距离标签保持纵向定义 |
| 颜色 | 红、蓝、浅灰，共3种 | 增加目标外观；固定Tesla车型避免首批混入目标几何变化 |
| 场景 | 6训练+2验证+2测试，背景必须分离 | 按场景隔离，不能把邻近视角随机分进不同split |
| 压力测试 | 雾、夜，只在测试场景 | 不混进训练或用来选checkpoint |
| 图像 | 原始640×640 RGB；可选同步原始深度 | 不再桌面截图，不提前压成32×32灰度 |

普通场景每个距离有6×5×3=90张配对图片。训练92,340张、验证30,780张、普通测试30,780张、雾/夜测试10,260张，合计164,160张RGB；加深度则PNG数量翻倍。图片不是164,160个独立物理状态：只有1,710个场景—距离组。按组取batch/计算指标，避免把外观变体当成独立样本夸大效果。

所谓“尽可能全”是这份有限采集表的完整覆盖，不是现实世界穷尽覆盖。没有动态交通、遮挡率控制、路面材质替换、运动模糊或动力学扰动；静态相机不提供真实速度标签。PPO中的速度输入仍需由任务状态单独提供，不能把静态截图复制成多个速度后声称采集了动态数据。雨量参数也不保证真实镜头水滴效果。

距离仍然是相机到目标actor原点的纵向距离，不是保险杠间距。所有新旧数据合并前要核对相机FOV、裁剪、距离定义；新传感器图片和旧桌面截图存在成像差异，不能宣称同分布。默认高度2米贴近原脚本，但不保证像素级匹配。

### 怎么运行

需要已经启动、能渲染、没有其他车辆/传感器、没有其他tick客户端的专用CARLA实例。Python环境中的carla包应与服务端版本匹配。SSH能连接不代表CARLA已启动；20020是SSH端口，默认2000才是CARLA端口。以下命令在代码根目录 `artical-F122` 执行；若carla安装在vt，可在python前加 `/opt/miniconda3/bin/conda run --no-capture-output -n vt`。

先看计划，无需安装CARLA：

```bash
python -m Aebs.connect.collect_latent_alignment --dry-run
```

检查当前地图和可选锚点（不重载地图）：

```bash
python -m Aebs.connect.collect_latent_alignment --inspect --host 127.0.0.1 --port 2000
```

复制 `Aebs/connect/latent_alignment_scenes.example.json` 为 `Aebs/connect/latent_alignment_scenes.local.json`。替换spawn_index，确认道路近水平、后方16米无遮挡且相机不穿地，场景之间背景不重叠。保留固定split，设置 `reviewed_for_capture` 为 `true`。示例0～9只是占位，脚本默认拒绝未经确认的正式采集，不能仅为跳过检查而设true。

试采所有场景8米处的全部外观（默认960张RGB）：

```bash
python -m Aebs.connect.collect_latent_alignment \
  --scenes Aebs/connect/latent_alignment_scenes.local.json \
  --expected-map YOUR_MAP_NAME --smoke \
  --output Aebs/carla_latent_alignment/smoke01
```

将 `YOUR_MAP_NAME` 替换为inspect打印的地图名末段，例如实际显示的是Town03才填Town03。检查每个场景、每种天气/姿态，尤其夜间是否目标仍可见。8米试采不能替代5米和16米的视野检查；先用距离参数做两端小批检查。若曝光未稳定，增加 `--settle-ticks 30`，正式采集前固定参数。

正式采集建议12批，一批一目录，以下先跑第0批：

```bash
python -m Aebs.connect.collect_latent_alignment \
  --scenes Aebs/connect/latent_alignment_scenes.local.json \
  --expected-map YOUR_MAP_NAME --shard-count 12 --shard-index 0 \
  --output Aebs/carla_latent_alignment/full01_part00
```

之后依次将index改为1～11，output改为对应part01～part11，其他参数和场景JSON完全不变。每批可先追加 `--dry-run` 查看数量。各批按scene+distance配对组确定性分配，组内所有外观保持完整，各批并集等于全计划；各批数目可能不同。不要在同一个CARLA实例并行跑多个批次。可以不分批跑全量，但不推荐首次就跑16万张。

输出目录必须不存在；不会覆盖旧图片。已完成批次无需重采。中断的那一批没有逐样本续采功能：用新的输出目录重跑该批，只保留一份完成的批次进入训练，不把失败的partial与重跑结果混合。`--smoke`图片也不要与正式数据重复导入。

磁盘和耗时以试采实测为准：用试采平均PNG大小乘全量图片数，并留出余量；用summary运行时间按样本数估算仅为参考。不要认为无需十几万张就无法进展，可以先减少训练场景、确认效果再补齐，验证/测试场景划分必须预先固定。

### 输出与训练衔接

每批输出包含 `config.json`、`samples.jsonl`、`summary.json`、`rgb/`、可选 `depth/`，完成后额外生成：

- `coverage_plan.json`：该批计划数量、分组数量和split计数。
- `collector_sha256.txt`：新脚本哈希；config的args还包含新脚本/场景配置/计划哈希。底层采集器哈希也保留。
- `labels_train.csv`、`labels_validation.csv`、`labels_test.csv`、`labels_test_ood.csv`：当前批存在的split才导出，保留filename、distance_m、group_id、scene_id、split、sample_id。路径相对该批根目录。

CSV的distance_m是同帧实测actor纵向距离。完整天气、相机参数、原始几何量、同步帧号和图片哈希仍在samples.jsonl中。配对训练使用group_id，不能对CSV逐行随机重新划分。

离线检查并重新生成split CSV：

```bash
python -m Aebs.connect.collect_latent_alignment \
  --export Aebs/carla_latent_alignment/full01_part00
```

它检查文件尺寸、哈希、帧对应和完成数量，但不检查目标可见性或遮挡。正式全量还应检查12个不同shard index全部齐全，地图/配置一致；单批audit不代表整个12批已经完成。

**当前交付是采集入口，不是训练入口迁移。** 现有DINO训练从旧H5读取，不能直接把这批RGB路径传给旧入口。下一步应按manifest和固定split提取RGB DINO特征，保留group_id用于同状态外观一致性损失。不要运行旧downSample.py把所有split混进一个H5，也不要自动对test提特征后用于调参。当前没有增加CP、没有修改PPO/SBC。

### 测试与回传

离线测试：`python -m unittest tests.test_collect_semantic_pairs tests.test_collect_latent_alignment -v`。本地12项通过，包含距离端点、无效参数、分批不重不漏、配对组完整、ID稳定、压力天气隔离、传感器同步及文件哈希审计。没有真实CARLA集成测试，传感器渲染和地图场景必须实测。

先回传inspect地图名、场景JSON、smoke的summary.json、同组不同外观示例。确认图像与标签后再全量采集，最终提供各批summary和分组覆盖统计。此阶段只判断数据质量和对齐误差改善，不产生安全保证。

采集同步与帧元数据依据：[CARLA同步时间步](https://carla.readthedocs.io/en/0.9.15/adv_synchrony_timestep/)、[传感器参考](https://carla.readthedocs.io/en/latest/ref_sensors/)。

## 1. 当前数据够不够

现有400张数据可以支持固定场景的最小原型，但不足以证明环境变化下的稳定性。代码库中的 `Aebs/connect/genGANData.py` 固定晴天、固定车辆和方向，移动观察点得到5～16米的图像；通过 `mss` 截取桌面窗口，只记录文件名和距离。`downSample.py` 随后转成32×32灰度图。

因此缺少的是**相同物理状态、不同视觉环境的配对数据**，不只是图片总数。单纯增加同一晴天视角的相邻距离图像，不能有效检验天气、光照和相机安装偏差。

建议PDF第3节提出同步采集 `(x, η, o)`，第13节讨论显式环境变量，第15～16节要求同状态不同环境的表示接近，同时保持不同安全状态可区分。本脚本实现这些建议所需的第一批静态图像数据。当前不添加CP。

## 2. 这次实现与未来数据的边界

第一批采集固定障碍车辆，让虚拟相机在车辆后方不同距离观测。改变天气、太阳、相机相对姿态和目标颜色，保存图像及同步真值。

- `x`：障碍物与相机参考点的几何关系，包括纵向距离及实际世界坐标。
- `η`：天气、太阳、相机高度/横向偏移/俯仰/偏航、目标颜色和场景身份。
- `o`：640×480 RGB PNG，保留彩色信息，不先压成32×32灰度。
- 同一 `group_id`：同场景、同参考距离下的多环境配对样本。

**这不是运动轨迹采集。**相机是独立传感器，没有自车；`ego_speed_mps=null`，目标车辆关闭物理模拟并记录实际速度。不能给静态图像随意贴上0.5、1、3 m/s等标签后声称采集了这些车速。车速作为PPO的独立传感输入，可以在后续仿真中与静态图像结合，但那仍是仿真输入组合。

静态配对可用于 `L_invariant`、安全语义监督和 `qψ(x,η)` 拟合，不能用于真实时序损失 `L_temporal`，因为相邻保存图像之间可能是相机瞬移。

视觉降雨、雾和亮度影响观测；路面摩擦、制动器延迟影响动力学。后者需要另采 `(x_t,u_t,x_{t+1})`，本脚本没有修改车辆摩擦或制动参数，也不把天气参数当作过程残差。

## 3. 已实现的采集维度

| 维度 | 默认设置 | 目的 |
| --- | --- | --- |
| 距离 | 5～16米，0.25米步长，共45档 | 包括5～6米附近的任务关键区域 |
| 训练/验证/普通测试天气 | 晴天、阴天、低太阳角、雨天，共4档 | 学习和评估常见外观变化 |
| 留出环境 | 雾天、夜间，共2档，仅测试场景采集 | 检查未训练环境中的表现 |
| 相机姿态 | 标准、左偏/高位、右偏/低位，共3档 | 检验小幅安装偏差 |
| 障碍物颜色 | 红色、蓝色，共2档 | 外观变化，目标几何保持相同 |
| 障碍模型 | `vehicle.tesla.model3` | 第一批固定几何，减少距离标签混淆 |
| 场景 | 示例3个锚点：训练、验证、测试各1个 | 用场景隔离数据 |
| 相机 | 640×480，水平FOV=90° | 保存RGB原图，后续统一做DINO预处理 |
| 可选深度 | 同帧CARLA原始深度PNG | 辅助检查几何和遮挡，不作为PPO默认输入 |

姿态参数：

| 姿态 | 相对目标坐标高度 | 横向偏移 | pitch | 相对yaw |
| --- | ---: | ---: | ---: | ---: |
| nominal | 1.6 m | 0 m | 0° | 0° |
| left_up | 1.7 m | −0.1 m | −2° | +2° |
| right_down | 1.5 m | +0.1 m | +2° | −2° |

高度是相对于目标actor原点的高度，不是经测量的离地高度；实际世界坐标也会保存。当前三个姿态是组合变化，只能评估组合鲁棒性；要单独归因俯仰或高度，应后续加单因素配置。

脚本没有实现路面材质替换、可控遮挡车辆、动态交通、动态运动模糊或真实镜头雨滴。目标颜色不等同于道路纹理。夜间也不会自动补充自车车灯，画面可能很暗，需人工检查并单列压力测试，不能据此要求模型在完全不可见的条件下恢复距离。

## 4. 数据量与推荐规模

示例3场景：

- 训练：1场景×45距离×4天气×3姿态×2颜色=1,080张；
- 验证：1,080张；
- 普通测试：1,080张；
- 雾/夜测试：1场景×45距离×2天气×3姿态×2颜色=540张；
- 合计 **3,780张RGB**；加 `--depth` 后是7,560个PNG文件。

这是流程检查规模，不是充分证明泛化的规模。第一批正式数据建议用6个训练、2个验证、2个测试场景，共 **11,880张RGB**。应选择背景明显不同、互不重叠的直路，而不是把相邻spawn点视为独立场景。真实耗时和压缩后磁盘空间由小批次试采估计；不预先承诺固定耗时。

同一个物理距离可以出现在不同场景的训练和测试中，这是场景泛化测试；它不证明未见距离插值或外推。若以后研究距离泛化，另外按距离区间预先分组，并让该区间的所有天气、姿态和颜色全部进入同一划分。

雾和夜只出现在测试场景，不允许拿来选checkpoint。普通测试和环境测试共用测试场景，故可以比较相同场景中的天气影响，但不能把它们当作完全独立的统计样本。

## 5. 距离定义必须统一

原采集脚本使用“相机参考位置到前车actor原点沿前车朝向的距离”，并不是前后保险杠间距。新脚本明确保存三种几何量：

| 字段 | 含义 |
| --- | --- |
| `distance_reference_m` | 请求的沿目标朝向纵向距离，用于配对分组 |
| `distance_center_longitudinal_m` | 图像同帧的实际相机到目标actor原点纵向投影；这里center名称指actor原点 |
| `distance_center_horizontal_m` | 相机到目标actor原点的水平欧氏距离，包含相机横向偏移 |
| `distance_rear_bbox_longitudinal_m` | 相机到目标包围盒最近后侧支撑平面的纵向距离 |

前两种纵向距离应相差不超过1 cm，否则脚本停止。最后一种仍不是两车保险杠间距，因为当前没有自车模型。以后若AEBS动力学改用真实保险杠间距，必须同时调整采集标签、动力学距离定义与安全阈值，不能悄悄替换标签后沿用5～6米阈值。

选择平直、近水平道路，确保相机后移16米仍在正确车道上。脚本检查锚点pitch/roll，却不能自动确认整段路况，也没有自动生成遮挡率标签。

## 6. 文件与样本格式

新增代码（相对 `artical-F122/`）：

- `Aebs/connect/collect_semantic_pairs.py`：采集、只读地图查询、采集计划预览、离线审计。
- `Aebs/connect/semantic_pair_scenes.example.json`：示例场景配置，必须替换为当前地图上检查过的锚点。
- `tests/test_collect_semantic_pairs.py`：无CARLA依赖的计划划分、帧同步及文件审计测试。

输出：

```text
run01/
  config.json       参数、地图、CARLA版本、脚本哈希、天气/姿态配置
  samples.jsonl     每张RGB对应一条完整标签；深度路径放在同一条里
  summary.json      完成/中断状态、计划数量、已写入数量、恢复错误
  rgb/*.png         原始RGB
  depth/*.png       可选，同帧原始打包深度
```

每条记录包含sample/group/scene ID、split、参考距离与实测距离、目标和相机世界位姿、天气数值、颜色、FOV、分辨率、世界帧号、时间戳、传感器帧号和文件SHA-256。

深度PNG保存CARLA原始打包编码，而不是肉眼看的灰度深度图。解码为米通常使用 `1000*(R+256G+65536B)/(256^3-1)`；读取文件时注意RGB/BGR通道顺序，具体语义见官方传感器文档。

不自动生成旧 `Downsampled.h5`，也不覆盖旧400张数据。现有训练器尚未读取该JSONL/RGB格式。后续接入需要按manifest选择split，用冻结DINO重新提取RGB特征，并为新数据拟合projection/代理；旧灰度模型的指标不能直接套到新RGB数据。

## 7. 如何运行

在运行CARLA的机器上执行，进入代码根目录。Python环境需要与服务端匹配的CARLA Python API；仅使用SSH主机不代表已启动CARLA。脚本用2000端口连接CARLA，SSH端口20020是另一回事。

```bash
cd /root/work_based_on_spvc/artical-F122
python -m Aebs.connect.collect_semantic_pairs --help
```

若CARLA包安装在 `vt`，将命令中的 `python` 替换为 `/opt/miniconda3/bin/conda run --no-capture-output -n vt python`。不要盲目安装与服务端不匹配的最新版CARLA包。

### 7.1 查看计划，不连接CARLA

```bash
python -m Aebs.connect.collect_semantic_pairs --dry-run
```

默认输出3,780张及四种split数量，无需CARLA、Torch或GPU。

### 7.2 查看当前地图和锚点

```bash
python -m Aebs.connect.collect_semantic_pairs --host 127.0.0.1 --port 2000 --inspect
```

会打印地图名、CARLA版本和spawn索引/坐标。将示例JSON另存为自己的场景配置（例如 `semantic_pair_scenes.local.json`），选直路且背景相互隔离的索引，修改 `id`、`spawn_index`、`split`。示例0/1/2只是占位，不能保证在你的地图上适用。

采集要求专用、空闲、开启渲染的CARLA实例。脚本拒绝已有车辆、行人或传感器的世界，也拒绝已有同步模式，避免混入动态交通或多个客户端同时tick。请关闭其他采集/训练客户端。它不会自行重载地图或删除别的actor。

### 7.3 首先采8米处的配对小批次

把下面 `Town10HD_Opt` 替换为inspect打印出的地图末段名称；该参数只检查地图，不加载地图。

```bash
python -m Aebs.connect.collect_semantic_pairs \
  --scenes Aebs/connect/semantic_pair_scenes.local.json \
  --expected-map Town10HD_Opt \
  --distance-min 8 --distance-max 8 --depth \
  --output Aebs/carla_semantic_pairs/smoke_8m
```

示例3场景产生84张RGB+84张深度。先看同组晴天/雨天/夜间图片：目标是否在画面中、是否穿模、相机是否位于路面上、夜间是否仍可见。同步tick每步0.05秒，每次换位后默认10帧稳定，再保留下一帧；如曝光未稳定，增加 `--settle-ticks 30`，并在正式采集前固定下来。

### 7.4 正式采集

```bash
python -m Aebs.connect.collect_semantic_pairs \
  --scenes Aebs/connect/semantic_pair_scenes.local.json \
  --expected-map Town10HD_Opt --depth \
  --output Aebs/carla_semantic_pairs/run01
```

输出目录必须不存在，防止新旧数据混写。采集中断后保留partial文件和summary；当前不实现续采，重新采集使用新目录。正常结束及可捕获异常会清理脚本创建的actor并恢复原天气和world设置。强制kill或服务器崩溃时不保证恢复，需要重启专用仿真实例。天气粒子/渲染和CARLA版本会影响像素，固定计划不等于逐像素确定复现。

### 7.5 离线审计

```bash
python -m Aebs.connect.collect_semantic_pairs \
  --audit Aebs/carla_semantic_pairs/run01
```

检查样本ID、配对组划分、图像帧号、文件存在、PNG头/尺寸和SHA-256，以及采集是否完整。它不判断画面是否合理或目标是否被遮挡；视觉检查仍必需。

## 8. 数据如何辅助后续训练

1. 在训练split上提取并缓存冻结DINO的RGB特征；验证/测试分别缓存，保留sample/group/split信息。
2. 用同一 `group_id` 下不同天气/姿态/颜色组成正配对，训练安全latent的一致性。场景不同的同距离图像可以用于扩展比较，但不随意跨split训练。
3. 用不同距离的样本维持状态可区分性，同时用距离辅助监督保留安全信息。
4. 可以分别评估 `qψ(d)` 和带环境描述的 `qψ(d,η)`；后者要明确η只在离线可知，并不假设部署能精确读到天气等参数。
5. PPO训练只读取训练图像。报告普通未见场景和雾/夜压力组的结果，分别统计成功、不安全、提前停车、超时及动作变化。
6. 数据中的视觉多样性不会自动给SBC端到端安全保证；应检验latent/动作残差，再讨论如何进入原SPVC分析。

## 9. 下一阶段再采什么

| 阶段 | 应增加的数据 | 为什么与本批分开 |
| --- | --- | --- |
| 扩展静态外观 | 更多地图、道路背景、目标模型，单因素姿态，已知遮挡物 | 检验跨场景与遮挡；目标几何改变要重新定义/记录距离 |
| 动态感知 | 相机挂真实自车，0.5～3 m/s行驶；同步记录速度、动作、下帧和episode ID | 支持时间一致性、运动模糊和真实在线闭环；按episode划分 |
| 动力学扰动 | 固定初始状态，改变摩擦、质量、延迟，记录实际transition | 用来估计过程残差；不能从静态天气图片推断 |

先完成本批静态配对，再补动态轨迹。不要把整段驾驶视频逐帧随机分进训练/测试；相邻帧的泄漏会让评估虚高。

## 10. 验收与需要回传的内容

本次代码交付仅做本地离线检查，没有启动或连接你的CARLA，也没有生成新的真实CARLA图像。

已完成检查：7项离线单元测试全部通过；默认计划预览为3,780张RGB，加深度为7,560个PNG；8米配对试采计划为84组RGB/深度。无效距离参数会被拒绝。测试覆盖场景划分、同组距离一致、雾/夜隔离、旧帧丢弃、未来帧拒绝、帧超时和文件篡改检测。CARLA实际渲染、actor生成、画面质量及异常后的世界恢复仍需在目标版本上实测。

运行后提供：

1. `--inspect` 的地图名、版本和你选中的锚点；
2. `--dry-run` 输出与最终 `summary.json`；
3. `--audit` 输出；
4. 同一 `group_id` 的晴、雨、雾/夜及不同相机姿态示例图；
5. `samples.jsonl` 中对应的几条记录。

先确认距离定义和图像质量，再进入训练，避免采满后发现标签不一致。第一批3780张/正式11880张只是可执行的起点，不是预先保证足够的数据量。

参考：用户提供的 `semantic+latent方案.pdf` 第3、8～9、13、15～16节；[CARLA传感器采集](https://carla.readthedocs.io/en/latest/tuto_G_retrieve_data/)、[同步时间步](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/)、[相机与深度编码](https://carla.readthedocs.io/en/0.9.10/ref_sensors/)。
