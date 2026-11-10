from pathlib import Path
import html

ROOT=Path(__file__).resolve().parent
PAGES=[('index.html','总览'),('aebs.html','AEBS'),('lane.html','车道'),('parking.html','停车场'),('robot.html','机器人'),('gallery.html','图片库'),('process.html','实验过程'),('compare.html','方法对比'),('data.html','数据与报告')]

def nav(active):
    links=''.join(f'<a href="{u}" class="{"active" if u==active else ""}">{html.escape(n)}</a>' for u,n in PAGES)
    return f'<nav><div class="nav-inner"><div class="brand">SPVC 实验总览<span>2026-11-09</span></div>{links}</div></nav>'

def page(name,title,active,body,subtitle=''):
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · SPVC</title><link rel="stylesheet" href="assets/style.css"></head>
<body>{nav(active)}<main class="wrap"><header class="hero"><h1>{html.escape(title)}</h1>{f'<p>{subtitle}</p>' if subtitle else ''}</header>{body}</main>
<footer><div class="footer-inner">SPVC 实验静态总览 · 结果与限制以各结果包和 docs/current 为准 · 不复制原始大数据</div></footer></body></html>'''
    (ROOT/name).write_text(doc,encoding='utf-8')

def table(headers, rows, cls=''):
    h=''.join(f'<th>{x}</th>' for x in headers)
    body=''.join('<tr>'+''.join(f'<td>{c}</td>' for c in r)+'</tr>' for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'

def pipes(items):
    out=[]
    for i,(b,s) in enumerate(items):
        out.append(f'<div class="pipe"><b>{b}</b><span>{s}</span></div>')
        if i<len(items)-1: out.append('<div class="arrow">→</div>')
    return '<div class="pipeline">'+''.join(out)+'</div>'

def fig(src,cap,alt=''):
    return f'<figure><img src="{src}" alt="{html.escape(alt)}"><figcaption>{cap}</figcaption></figure>'

def badge(t,kind): return f'<span class="badge {kind}">{t}</span>'

# ---------------- index ----------------
index_body=f'''
<div class="badges">{badge('AEBS：条件证书','ok')}{badge('车道：证书通过','ok')}{badge('停车场：受保护闭环证书','ok')}{badge('机器人：任务跑通 / 证书未通过','bad')}</div>
<p>这个站点把当前 SPVC 改进实验放在一起：每个实验的配置、逐步输入/输出、结果、保证边界和精选图片。原始 24k/3.6k 图片不复制，只放缩略图和完整包链接。</p>
<div class="grid">
<div class="card"><h3>AEBS · 真实 RGB 紧急制动</h3><p>24,000 张 640×640 CARLA RGB；DINO 距离观测；冻结回放 CP；新非负 SBC。</p>
<div class="num">0.4184 <small>m 轨迹 CP</small></div><p>292,024 区间 · 0 见证/0 未解决 · 4,800/4,800 行为</p><img src="assets/img/aebs_real_rgb.jpg" alt="AEBS 真实 RGB 输入示例"></div>
<div class="card"><h3>车道 · 2D 视觉保持</h3><p>192×192 渲染图；patch-DINO (ê,θ̂)；轨迹 CP；学习式非负 SBC。</p>
<div class="num">0.09009 <small>归一化 CP</small></div><p>199/100 轨迹 · 95/100 全程覆盖 · SBC 0/0</p><img src="assets/img/lane_inputs.jpg" alt="车道输入图像"></div>
<div class="card"><h3>停车场 · 三维视觉绕障</h3><p>4D 车辆、3D 视觉观测、在线后备；15 个起点组启动 CP。</p>
<div class="num">30/30 <small>图像仿真</small></div><p>启动审计 233/240 · 增广可达域神经 SBC</p><img src="assets/img/parking_sbc.jpg" alt="停车场 SBC 结果图"></div>
<div class="card"><h3>移动机器人 · 三维视觉</h3><p>3D 位姿、双动作、120 步；DINO 三维观测。</p>
<div class="num">24/24 <small>图像闭环</small></div><p>CP 已算；SBC 未通过；CP 盒内 28/32 反例</p><img src="assets/img/robot_traj.jpg" alt="移动机器人轨迹"></div>
</div>
<div class="section"><h2>前端误差对比（AEBS，同一 4,920 张 test）</h2>
{table(['前端','test MAE ↓','RMSE','静态 CP95 ↓','test 覆盖 ↑'],[
 ['原 SPVC cGAN + state_net（旧 400 图检查点）','2.747 m','3.320 m','5.221 m','87.64%'],
 ['32×32 灰度 DINO（历史运行）','0.1017 m','0.1842 m','0.2656 m','90.47%'],
 ['32×32 灰度 DINO（本轮重训，可重载）','0.1079 m','0.2010 m','0.2797 m','90.75%'],
 ['<b>真实 RGB DINO（当前）</b>','<b>0.0874 m</b>','<b>0.1211 m</b>','<b>0.2307 m</b>','<b>93.33%</b>']])}
<p class="small">这是<b>同一 4,920 test 的静态图片级</b>比较；CP 不是轨迹级 0.418359 m。cGAN 是旧 400 图检查点，非同预算；灰度 A1 同时改了颜色和源分辨率（32×32→224）；rain 分组灰度 MAE 反而低于 RGB。原 10/29 的 400 图灰度包不在本表。</p>
<p><a href="compare.html"><b>打开“方法对比”页</b></a>：前端误差、CP/SBC 和四个实验的完整口径表。</p>
</div>
<div class="section"><h2>详细过程与方法对比</h2><p><a href="process.html"><b>实验过程页</b></a> 按“输入 → 做了什么 → 输出/产物 → 为什么/检查”逐步展开 AEBS、车道、停车场、移动机器人；不是只列结果数字。</p><p><a href="compare.html"><b>方法对比页</b></a> 把同一 test 的前端误差、原 10/29 灰度实现、CP/SBC 敏感性和四个实验放在同一口径下比较。更完整的原理说明见 <a href="../docs/current/SPVC当前框架_逐步流程与安全保证_20261109.md">完整框架文档</a>。</p></div>
<div class="section"><h2>怎么读</h2><div class="grid three">
<div class="card"><h3><a href="aebs.html">AEBS</a></h3><p>真实 RGB、CP 单位修复、轨迹 CP、新 SBC、状态相关对照。</p></div>
<div class="card"><h3><a href="lane.html">车道</a></h3><p>2D latent、轨迹 CP、SBC 审计、状态相关负结果。</p></div>
<div class="card"><h3><a href="robot.html">机器人</a></h3><p>任务成功与证书失败的边界。</p></div>
</div></div>
<div class="section"><h2>统一安全标准</h2><p>非负 SBC `B≥0`；初始 `B≤1`；危险 `B≥20`；允许观测集合内 `B(next)≤B(current)`。满足时给出停止过程风险上界 `B0/20≤5%`。这是条件模型保证，不是新图像/实车/无条件 95%。</p></div>
<div class="warnbox"><b>最新修复：</b>AEBS 旧 0.2307 m 的 tube 通过结论因单位漏乘已撤回；当前主结果是 0.418359 m 轨迹 CP + 新 SBC。历史文档保留在 <a href="../docs/archive/2026-11-09_current_cleanup/README.md">归档</a>。</div>
'''

# ---------------- AEBS ----------------
aebs_body=f'''
<div class="badges">{badge('条件证书：通过','ok')}{badge('真实 RGB 640×640','info')}{badge('冻结图片回放','warn')}{badge('数值区间，非形式化浮点','warn')}</div>
<div class="grid">
<div class="card"><div class="num">0.4184 <small>m 轨迹 CP</small></div><p>199 校准 / 100 留出；留出 99/100 全程覆盖</p></div>
<div class="card"><div class="num">292,024 <small>验证区间</small></div><p>0 正漂移见证 / 0 未解决；初始 0，危险下界 25.2136</p></div>
<div class="card"><div class="num">4,800/4,800 <small>行为</small></div><p>12 组天气/颜色 ×400；全库与 test 库均成功</p></div>
<div class="card"><div class="num">0.0874 <small>m test MAE</small></div><p>DINO 距离头；max 0.525 m；不是实车/在线渲染</p></div>
</div>

<div class="section"><h2>1. 任务与配置</h2>
<div class="kv">
<div>物理状态</div><div><code>x=(d,v)</code>：距离 m、速度 m/s，2D</div>
<div>动作</div><div><code>a∈[-3,3]</code>，正值为制动，1D</div>
<div>动力学</div><div><code>d' = d - 0.05 v</code>，<code>v' = clip(v - 0.05 a, 0, 3)</code>，Δt=0.05 s</div>
<div>初态 / 危险</div><div><code>[15,16]×[2.5,3]</code> / <code>[5,6]×[0.5,3]</code></div>
<div>数据</div><div>24,000 张 CARLA 640×640 RGB；4 天气 ×3 颜色；距离 5–16 m；train 14,172 / validation 4,908 / test 4,920</div>
<div>前端</div><div>冻结 DINOv2-S/14 CLS 384 → 384-128-128-1 → 归一化距离</div>
<div>控制器</div><div>原已认证 AEBS PPO；输入 <code>[d̂/S, v]</code>，S=3.18272066</div>
<div>SBC</div><div>非负 <code>[2,16,8,1]</code>，Tanh 隐藏层 + ReLU 输出；v2 新训练</div>
<div>CP</div><div>冻结回放轨迹的米制最大误差；常量 0.418359 m</div>
<div>验证</div><div>连续域 tube 292,024 区间 + 400 图冻结回放；float64 数值区间</div>
</div></div>

<div class="section"><h2>2. 一步一步：输入 / 处理 / 输出</h2>
{table(['步','输入','处理','输出 / 当前结果'],[
 ['1. 数据','24k RGB PNG + labels.csv','按 0.25 m 距离 bin 分组划分；同组不跨集合','train 14,172 / val 4,908 / test 4,920'],
 ['2. 前端','真实 RGB','冻结 DINO CLS，训练距离头；不在线渲染','test MAE 0.0874 m，max 0.525 m'],
 ['3. 静态 CP','校验图片的 |d̂-d|','对 q 做 split CP；单位修正为 r=S·q','静态半径 0.2307 m，test 覆盖 93.33%'],
 ['4. 回放','图片库 + PPO + 400 初态','12 组天气/颜色分别固定；最近邻查图','全库 4,800/4,800；test 4,800/4,800 成功'],
 ['5. 轨迹 CP','199 条回放轨迹','每条取 max_t 米制误差；冻结 q 后再采 100 条','半径 0.418359 m；留出 99/100 覆盖'],
 ['6. 新 SBC','原 actor、前端、轨迹 CP、反例','固定 actor/前端，热启动训练非负 SBC','初始 0；危险下界 25.213636；v2 通过'],
 ['7. tube 验证','SBC v2、全部 live 状态、CP 球','自适应区间 + 仿射 actor 界；不缩小半径','292,024 区间；0 见证/0 未解决；深度 29'],
 ['8. 行为/对照','冻结模型、12 组图片','全库/test 回放；cGAN/灰度/RGB 对比','4,800/4,800；见下表'],
])}</div>

<div class="section"><h2>3. 当前结果与对照</h2>
<h3>主证书（常量 CP）</h3>
{table(['项目','数值'],[
 ['轨迹 CP 半径','0.4183585684 m（第 190/199）'],
 ['校准 / 留出覆盖','190/199；99/100'],
 ['初始上界 / 危险下界','0 / 25.2136360598'],
 ['SBC 验证','292,024 区间；0 实际反例；0 未解决；深度 29'],
 ['行为','全库 4,800/4,800；仅 test 4,800/4,800；OOD 12 组全部成功'],
])}
<h3>旧 0.2307 m 结论为什么撤回</h3>
<div class="badbox">旧合同把归一化 q=0.072493 直接当米；实际验证只有 ±0.07249 m。正确半径 0.230725 m 下，旧 SBC 有 <b>212 个正漂移见证 / 150,033 未解决</b>，旧 tube 通过结论撤回。旧报告保留，不作为当前证据。</div>
<h3>状态相关 CP 对照（参考 10/29）</h3>
{table(['项目','状态相关','常量主结果'],[
 ['平均半径','0.3217 m','0.4184 m'],
 ['最大半径','0.7189 m（d≈16）','0.4184 m'],
 ['校准 / 留出覆盖','190/199；92/100','190/199；99/100'],
 ['SBC 验证','41,070 区间；0/0；深度 11','292,024 区间；0/0；深度 29'],
])}
<p><b>注意：这里的 `C` 仍是固定 `[1,0]`，只学习误差尺度 `b0(d)`；联合学习视觉投影和 `C` 是另一个独立实验，见<a href="compare.html">方法对比</a>。</b> 状态相关把危险区附近半径压小、远距离放大；SBC 远距离在零平台，所以验证更省。但留出覆盖 92/100 低于常量版，<b>不替换主发布包</b>。<a href="../artical-F122/results/real_aebs_state_cp_20261109_units_v2/README.md">状态相关包</a></p>
<h3>前端误差/CP 对比（同一 4,920 test）</h3>
{table(['前端','test MAE','RMSE','静态 CP95','test 覆盖'],[
 ['cGAN+state_net（旧 400 图检查点）','2.747 m','3.320 m','5.221 m','87.64%'],
 ['32×32 灰度 DINO（历史）','0.1017 m','0.1842 m','0.2656 m','90.47%'],
 ['32×32 灰度 DINO（重训，可重载）','0.1079 m','0.2010 m','0.2797 m','90.75%'],
 ['<b>真实 RGB DINO（当前）</b>','<b>0.0874 m</b>','<b>0.1211 m</b>','<b>0.2307 m</b>','<b>93.33%</b>']])}
<p class="small">这是静态图片级 CP，不是轨迹级 0.418359 m；cGAN 旧检查点非同预算；灰度同时改颜色和源分辨率；rain 分组灰度 MAE 反而更低。原 10/29 的 400 图灰度包是独立历史基线，不能与本表横比。</p>
<p><a href="compare.html">完整方法对比与口径 →</a>。原生 cGAN 闭环 q=1.9156 m/92，新 DINO q=0.4184 m/99；5.221 m 是迁移口径，不是原生 q。</p>
</div>

<div class="section"><h2>3.1 原来的 10/29 灰度实现（历史基线）</h2>
<table><thead><tr><th>项目</th><th>原 10/29 灰度包</th></tr></thead><tbody>
<tr><td>数据</td><td>400 张 32×32 灰度图；240 train / 80 validation / 80 test</td></tr>
<tr><td>观测</td><td>冻结 DINOv2 ViT-S/14 CLS 384 → 128 → 128 → 1；投影最佳 epoch 243</td></tr>
<tr><td>test 误差</td><td>MAE 0.129266 m；RMSE 0.165444 m；max 0.414889 m（仅 80 test 图）</td></tr>
<tr><td>CP</td><td>状态相关 Bernstein b0(d)；q=2.522199；均值半径约 0.4316 m；40 条校准</td></tr>
<tr><td>SBC</td><td>同包非负 [2,16,8,1]；初始 0；危险下界 24.965932</td></tr>
<tr><td>连续域</td><td>CP tube 4,971,398 区间；0 反例/0 未解决；冻结回放 105,608 区间 0/0</td></tr>
<tr><td>行为</td><td>名义 / 全图 / 留出各 1441/1441 成功；32 项软件测试</td></tr>
</tbody></table>
<div class="warnbox">这个包只有 400 张 32×32 图和 80 张 test，使用状态相关 CP，SBC 与策略配套训练；<b>不能把它和 24k test 的 A1/A2 横比</b>。它证明早期灰度链路能闭合，不是当前前端排名的证据。<a href="../artical-F122/results/vision_sls_spvc_sbc_pass_20261029/README.md">原灰度包</a></div>
</div>

<div class="section"><h2>4. 输入图像与结果图</h2>
{fig('assets/img/aebs_real_rgb.jpg','真实 RGB 输入示例：6 个距离/天气组合；上方为标签。')}
{fig('assets/img/aebs_compare.jpg','cGAN / 32×32 灰度 / 真实 RGB 的距离误差、CP 半径与 test 覆盖。')}
</div>

<div class="section"><h2>5. 保证与边界</h2>
<div class="note"><b>保证：</b>在冻结回放分布和 CP 球内观测条件下，非负 SBC 逐步不增；初始 B≤1、危险 B≥20；支持停止过程风险上界 B0/20≤5%。</div>
<ul><li>不覆盖新图像、实车、在线 CARLA 或任意固定初态的条件覆盖。</li><li>CP test 覆盖 93.33%，低于 95%；tube 论证只覆盖包络内。</li><li>数值区间不是完整浮点形式化证明；不证明目标到达。</li><li>旧 0.2307 m tube 通过已撤回；当前以 0.418359 m + 新 SBC 为准。</li></ul>
<p><a href="../docs/current/AEBS真实RGB图像_完整实验_20261108.md">完整报告</a> · <a href="../artical-F122/results/real_aebs_repair_20261109_release/README.md">修复发布包</a> · <a href="../artical-F122/results/real_aebs_repair_20261109_release/verification_report.json">验证 JSON</a></p>
</div>
'''

# ---------------- lane ----------------
lane_body=f'''
<div class="badges">{badge('证书通过','ok')}{badge('2D latent (e,θ)','info')}{badge('名义渲染分布','warn')}{badge('数值区间','warn')}</div>
<div class="grid">
<div class="card"><div class="num">0.09009 <small>归一化轨迹 CP</small></div><p>199 校准 / 100 留出；95/100 全程覆盖</p></div>
<div class="card"><div class="num">0/0 <small>SBC 审计</small></div><p>初始 0.0858 / 危险 23.6535；三个曲率</p></div>
<div class="card"><div class="num">12/12 <small>图像闭环</small></div><p>patch-DINO；CP 覆盖 98.1–100%；OOD 9/9</p></div>
<div class="card"><div class="num">0.0211 <small>m e-MAE</small></div><p>θ MAE 0.0059 rad；2D 观测</p></div>
</div>
<div class="section"><h2>1. 任务与配置</h2>
<div class="kv">
<div>物理状态</div><div><code>(e,θ)</code>：横向偏差、航向偏差，2D</div>
<div>动作</div><div>前轮转角 <code>φ∈[-0.45,0.45]</code>，1D</div>
<div>动力学</div><div><code>e'=e+vΔt sinθ</code>，<code>θ'=θ+(vΔt/L)tanφ-vΔtκ</code>；v=4，L=2.6，Δt=0.1</div>
<div>初态 / 安全</div><div><code>|e|≤0.25, |θ|≤0.06</code> / <code>|e|≤0.85, |θ|≤0.30</code>；H=80</div>
<div>数据</div><div>60 个名义场景 ×60 状态 = 3,600 张 192×192 渲染图；曲率 κ∈{-0.01,0,0.01}</div>
<div>前端</div><div>冻结 DINOv2-S/14 16×16 patch → soft-argmax → <code>(ê,θ̂)</code></div>
<div>控制器</div><div>原图像 PPO；输入 <code>(ê,θ̂,κ)</code>；本包未重训策略</div>
<div>SBC</div><div>AEBS 同类非负 <code>[2,16,8,1]</code>，Tanh/ReLU；学习式</div>
<div>CP</div><div>80 步 ×2 坐标的轨迹最大误差；0.0900931580</div>
<div>验证</div><div>仿射界 + 普通 IBP；三个曲率；0 实际见证/0 未解决</div>
</div></div>

<div class="section"><h2>2. 一步一步：输入 / 处理 / 输出</h2>
{table(['步','输入','处理','输出 / 结果'],[
 ['1. 场景/图像','60 场景 ×60 状态','渲染 192×192；按场景划分','train 2,160 / val 540 / cal 540 / test 360'],
 ['2. 观测','patch-DINO','16×16 tokens → 空间 soft-argmax → 2D','e MAE 0.0211 m，θ 0.0059 rad'],
 ['3. 策略','观测 (ê,θ̂,κ)','原 PPO 80k 步；观测误差扰动训练','固定 actor；未重训'],
 ['4. 单帧 CP','校验残差','max-norm split CP','0.072 参考；证书用 0.08'],
 ['5. 轨迹 CP','199 条轨迹','每条 80×2 最大误差；冻结 q 后再采 100','0.090093；留出 95/100'],
 ['6. SBC','真实状态、误差盒','AEBS 同类学习式 SBC；两轮训练','初始 0.0858 / 危险 23.6535'],
 ['7. 连续域','SBC、CP 盒、三个曲率','自适应 40×40 起，深度 18；IBP 复核','0 见证/0 未解决'],
 ['8. 图像闭环','patch-DINO、PPO、渲染器','6 个测试场景 ×2 回合','12/12 成功；CP 覆盖 98.1–100%'],
])}</div>

<div class="section"><h2>3. 结果与状态相关对照</h2>
<table><thead><tr><th>项目</th><th>常量盒（主结果）</th><th>状态相关 v2（修复版）</th></tr></thead><tbody>
<tr><td>拟合 / 校准</td><td>—</td><td>99 / 100 条，索引不重叠</td></tr>
<tr><td>阈值 / 秩</td><td>0.090093，190/199</td><td>q=1.7205338，第 96/100 秩；无 cap</td></tr>
<tr><td>校准覆盖</td><td>190/199</td><td>96/100</td></tr>
<tr><td>旧留出回归覆盖</td><td>95/100</td><td><b>93/100</b>（旧数据回归，不是新盲测）</td></tr>
<tr><td>平均半径</td><td>0.090093</td><td>0.0883927</td></tr>
<tr><td>观测最大半径</td><td>0.090093</td><td>0.1093692；全状态域解析上界 0.2358151</td></tr>
<tr><td>SBC</td><td>0/0，原审计</td><td><b>13,000 区间，0 见证/0 未解决，三曲率</b></td></tr>
<tr><td>同一 100 条校准常量对照</td><td>—</td><td>半径 0.0886653；旧留出 94/100</td></tr>
</tbody></table>
<div class="note"><b>修复版 v2 的关键：</b>用前 99 条拟合、后 100 条校准，有限样本秩 96；留出分数只读留出数据自身残差；没有 `min(...,0.0992)` 截断。实际状态相关集合直接验证通过，不是用全域最大半径 0.235815 的扩大盒；本地/服务器各 31 项测试通过；这里的 `C` 仍固定。联合学习 C 是另一个独立实验，见<a href="compare.html">方法对比</a>。<br><b>不能说过头：</b>93/100 是旧数据回归，不是新增盲测；状态相关包络也没有明显优于常量盒（同一校准常量 0.0886653、留出 94/100）；`C` 仍固定，不是联合学习新前端。</div>
<div class="badbox"><b>旧 v1 仍撤回：</b>代码/报告不一致、留出分数误用校准残差、同批拟合+校准且 cap 收缩集合；旧 0.0992 盒上的 SBC 通过不能证明 CP 覆盖。见 <a href="../artical-F122/results/lane_state_cp_20261109_v1/AUDIT.md">v1 审计</a>。</div>
</div>

<div class="section"><h2>4. 输入图像与结果</h2>
{fig('assets/img/lane_inputs.jpg','车道输入图像：6 个不同 (e,θ,κ) 的 192×192 渲染图。')}
{fig('assets/img/lane_closed.jpg','车道闭环轨迹示例。')}
{fig('assets/img/lane_coverage.jpg','轨迹级 CP：校准/留出轨迹最大误差与完整覆盖数。')}
</div>

<div class="section"><h2>5. 保证与边界</h2>
<div class="note"><b>保证：</b>若每个时刻观测误差都在 CP 盒内，非负 SBC 逐后继不增；初始低、危险高；状态不首次越界。轨迹 CP 是边缘覆盖，不是逐状态条件覆盖。</div>
<ul><li>只验证三个注册曲率 κ；不是连续曲率。</li><li>图像是名义渲染器，不是实车/新相机；CP 只覆盖注册名义分布。</li><li>数值区间，不是形式化浮点；不证明目标到达。</li><li>状态相关盒试验留出覆盖 93/100，低于常量版，不作为主结果。</li></ul>
<p><a href="../docs/current/车道保持_轨迹级CP校准与SBC衔接_20261107.md">完整报告</a> · <a href="../artical-F122/results/lane_trajectory_cp_20261107_v1/README.md">轨迹 CP 包</a> · <a href="../artical-F122/results/lane_state_cp_20261109_v1/README.md">状态相关试验包</a></p>
</div>
'''

# ---------------- parking ----------------
parking_body=f'''
<div class="badges">{badge('受保护闭环证书','ok')}{badge('4D 状态 / 3D 观测','info')}{badge('在线后备必需','warn')}{badge('≠ 原无过滤器 SPVC','warn')}</div>
<div class="grid">
<div class="card"><div class="num">30/30 <small>图像仿真</small></div><p>原点/扰动两组；无碰撞、无越界</p></div>
<div class="card"><div class="num">233/240 <small>启动 CP 审计</small></div><p>15 个起点组；α=0.05，审计不参与调半径</p></div>
<div class="card"><div class="num">≤0.305 <small>初始 SBC</small></div><p>危险 ≥20；非终端单步变化 ≤−0.005</p></div>
<div class="card"><div class="num">59/59 <small>本地测试</small></div><p>服务器 CPU 同样 59/59；GPU 一次 OOM 后 CPU 通过</p></div>
</div>
<div class="section"><h2>1. 任务与配置</h2>
<div class="kv">
<div>任务</div><div>停车场四维车辆绕障并停入目标；15 个注册起点</div>
<div>物理状态</div><div>4D 车辆状态；视觉观测 3D；增广验证 8D（状态 + 观测器/参考）</div>
<div>控制</div><div>参考规划 + 延迟观测器 + LQR + 残差 PPO 建议 + <b>在线区间后备检查</b></div>
<div>启动 CP</div><div>每个起点组：16 拟合 / 32 校准 / 16 审计；组内 ±0.02</div>
<div>时域</div><div>30 步，dt=0.15，零过程扰动，原 RK4 模型</div>
<div>SBC</div><div>几何 ReLU 网络 + 有界 Tanh/Sigmoid 学习头 + 线性时钟；结构化神经 SBC</div>
<div>证明域</div><div>新增后备后的增广闭环可达不变域；不是原无过滤器 8D 全域</div>
<div>图像</div><div>固定俯视渲染；启动后不再用图像删除安全假设</div>
</div></div>
<div class="section"><h2>2. 一步一步</h2>
{table(['步','输入','处理','输出'],[
 ['1. 启动图像','每个起点组新渲染 65 张','拟合/校准/审计 + 中心检查','启动误差集合 b_j,r_j'],
 ['2. 启动 CP','32 个校准样本/组','每组一次联合 split CP','q_j；审计 233/240'],
 ['3. 初始信念','启动观测 + 误差集合','保留能解释观测的起点组；不把真实组给控制器','增广初始状态集合'],
 ['4. 后备控制','参考轨迹 + 延迟观测器 + PPO','LQR/PPO 建议经在线区间后备检查','安全动作；无后备则证书不适用'],
 ['5. SBC','增广状态、时钟、参考编号','几何+学习神经 SBC 训练与区间验证','初始 ≤0.305；危险 ≥20；漂移 ≤−0.005'],
 ['6. 图像实验','15 起点 ×2 组','原任务 30 步图像闭环','30/30 成功；0 碰撞/0 越界'],
])}</div>
<div class="section"><h2>3. 结果图</h2>
{fig('assets/img/parking_sbc.jpg','停车场结构化神经 SBC / 安全集结果图。')}
{fig('assets/img/parking_traj.jpg','规划反馈闭环轨迹对比。')}
</div>
<div class="section"><h2>4. 保证与边界</h2>
<div class="note"><b>保证：</b>在启动 CP 覆盖事件和注册模型/后备条件下，增广闭环不碰撞、不越界；匹配启动分布时支持单次新实验至少 95% 的模型安全概率。</div>
<ul><li>必须使用在线后备；仅加载 PPO/SBC 权重不提供本包闭环保证。</li><li>不是原无过滤器 SPVC 的任意八维安全状态域。</li><li>初始抖动 ±0.02、30 步、零过程扰动；不是实车。</li><li>任务到达和安全性分开；20/19 对抗回归中有一个极端误差 trial 未到达。</li></ul>
<p><a href="../docs/current/停车场神经SBC_构造证明与实验_20261103.md">完整专题</a> · <a href="../artical-F122/results/vision_sls_parking_20261103_neural_sbc_certificate_v1/README.md">结果包</a> · <a href="../artical-F122/results/vision_sls_parking_20261103_neural_sbc_certificate_v1/certificate.json">证书 JSON</a></p>
</div>
'''

# ---------------- robot ----------------
robot_body=f'''
<div class="badges">{badge('图像任务跑通','ok')}{badge('完整 SBC 证书未通过','bad')}{badge('CP 盒内有实际反例','bad')}{badge('3D 状态 / 2D 动作','info')}</div>
<div class="grid">
<div class="card"><div class="num">24/24 <small>开发图像闭环</small></div><p>安全到达并停稳；距障碍排除边界最小约 0.153 m</p></div>
<div class="card"><div class="num">39/20 <small>校准/留出轨迹</small></div><p>CP 半宽约 (0.0488,0.0488,0.3254)；留出覆盖 18/20</p></div>
<div class="card"><div class="num">20/20 <small>留出行为</small></div><p>安全到达并停稳；覆盖 90% 不等于 95% 测试通过</p></div>
<div class="card"><div class="num">28/32 <small>CP 盒内反例</small></div><p>21 条机器人/障碍物圆重叠；完整证书不成立</p></div>
</div>
<div class="section"><h2>1. 任务与配置</h2>
<div class="kv">
<div>状态</div><div><code>(x,y,θ)</code>，3D；位置与朝向</div>
<div>动作</div><div>线速度 + 角速度，2D；120 步，12 s</div>
<div>动力学</div><div>离散独轮车；dt=0.1；无障碍过程噪声</div>
<div>视觉</div><div>224×224 渲染图 → DINO ViT-S/14 patch → 空间注意力 → 3D 位姿</div>
<div>训练</div><div>2,400 张训练 / 400 验证 / 400 开发测试</div>
<div>控制器</div><div>PPO 直接输出双动作；无在线后备/过滤器</div>
<div>SBC</div><div>AEBS 同类非负网络；离线证书训练</div>
<div>CP</div><div>每条 120 步 ×3 坐标的最大残差；39 校准 / 20 留出</div>
</div></div>
<div class="section"><h2>2. 一步一步</h2>
{table(['步','输入','处理','输出 / 结果'],[
 ['1. 场景','6×4 m 地图，中央障碍','渲染 224×224；机器人/障碍/目标/地图标记','2,400+400+400 图像'],
 ['2. 观测','DINO patch','卷积 + 双空间注意力 → 3D 位姿','x MAE 2.55 cm、y 1.86 cm、θ 4.49°'],
 ['3. PPO','3D 观测','30 万步；宽初态 + 绕障距离奖励','名义 300/300；±0.03 停稳 275/300'],
 ['4. 图像闭环','渲染图 → DINO → PPO','24 条开发轨迹','24/24 安全到达并停稳'],
 ['5. 轨迹 CP','39 校准 + 20 留出','max_t 三维归一化误差；冻结阈值','半宽约 (0.0488,0.0488,0.3254)；留出 18/20'],
 ['6. SBC','真实状态、CP 盒','非负网络训练 / 区间审查','初始 0.179、危险 20.807，但不增条件未通过'],
 ['7. 反例','CP 盒内扰动搜索','可微搜索 + float64 独立重放','28/32 违反；21 条圆重叠'],
])}</div>
<div class="section"><h2>3. 为什么不能继续只训 SBC</h2>
<div class="badbox">固定当前控制器、动力学和 CP 盒，已经找到盒内实际越界反例。若假设初始 B≤1、危险 B≥20、每步不增，沿该允许轨迹会推出 B≤1 且 B≥20，矛盾。因此<b>不是训练轮数不够</b>；需要改控制器/误差表示/加后备。</div>
<p>下一步：改进朝向识别；用状态相关/椭球/带相关性的误差集合；把反例用于控制器训练；改完后重新采集轨迹 CP。</p>
</div>
<div class="section"><h2>4. 图像与轨迹</h2>
{fig('assets/img/robot_examples.jpg','三维视觉输入示例：机器人、障碍物、目标与地图标记。')}
{fig('assets/img/robot_traj.jpg','开发图像闭环轨迹。')}
{fig('assets/img/robot_attack.jpg','CP 盒内反例 / 攻击轨迹示例。')}
</div>
<div class="section"><h2>5. 保证与边界</h2>
<ul><li>已得到：三维任务、图像闭环开发成功、轨迹 CP、初始/危险区分离的神经候选。</li><li>未得到：完整 CP + 神经 SBC 概率安全保证。</li><li>CP 盒内反例否定的是“对盒内所有误差”的充分条件，不否定 CP 覆盖定理，也不否定其他条件噪声模型。</li><li>数值区间，不是形式化浮点；不是实车。</li></ul>
<p><a href="../docs/current/移动机器人_三维视觉实验_20261108.md">完整专题</a> · <a href="../artical-F122/results/mobile_robot_20261108_server_attack_cp/report.json">反例 JSON</a> · <a href="../artical-F122/results/mobile_robot_20261108_server_trajectory_cp_r1/report.json">轨迹 CP JSON</a></p>
</div>
'''

# ---------------- gallery ----------------
gallery_body=f'''
<p>只放精选缩略图；原始 24k/3.6k 图片在服务器/结果包中，不复制进站点。</p>
<div class="grid">
<div class="card"><img src="assets/img/aebs_real_rgb.jpg" alt="AEBS real"><h3>AEBS 真实 RGB</h3><p>6 个距离/天气组合；640×640 输入。</p></div>
<div class="card"><img src="assets/img/aebs_gray.jpg" alt="AEBS gray"><h3>AEBS 32×32 灰度</h3><p>旧前端对照；同一任务。</p></div>
<div class="card"><img src="assets/img/aebs_compare.jpg" alt="AEBS compare"><h3>前端误差对比</h3><p>cGAN / 灰度 / 真实 RGB 的 MAE、CP、覆盖。</p></div>
<div class="card"><img src="assets/img/lane_inputs.jpg" alt="Lane inputs"><h3>车道输入</h3><p>不同 (e,θ,κ) 的 192×192 渲染。</p></div>
<div class="card"><img src="assets/img/lane_closed.jpg" alt="Lane closed"><h3>车道闭环</h3><p>patch-DINO + PPO 的轨迹。</p></div>
<div class="card"><img src="assets/img/lane_coverage.jpg" alt="Lane CP"><h3>车道轨迹 CP</h3><p>校准/留出最大误差与完整覆盖数。</p></div>
<div class="card"><img src="assets/img/parking_sbc.jpg" alt="Parking SBC"><h3>停车场 SBC</h3><p>结构化神经 SBC / 安全集。</p></div>
<div class="card"><img src="assets/img/parking_traj.jpg" alt="Parking trajectory"><h3>停车场轨迹</h3><p>规划反馈闭环对比。</p></div>
<div class="card"><img src="assets/img/robot_examples.jpg" alt="Robot inputs"><h3>机器人输入</h3><p>三维视觉场景。</p></div>
<div class="card"><img src="assets/img/robot_traj.jpg" alt="Robot traj"><h3>机器人轨迹</h3><p>开发图像闭环。</p></div>
<div class="card"><img src="assets/img/robot_attack.jpg" alt="Robot attack"><h3>机器人反例</h3><p>CP 盒内攻击/越界示例。</p></div>
</div>
'''

# ---------------- process ----------------
process_body=f"""
<p>按“输入 → 做了什么 → 输出/产物 → 为什么/检查”展开。这里只写当前版本，历史失败版本在数据与报告页链接里。</p><p><a href="../docs/current/SPVC当前框架_逐步流程与安全保证_20261109.md"><b>完整框架文档（推荐先读）</b></a>：更详细地解释 CP、SBC、验证器和保证边界。</p>

<div class="section"><h2>AEBS RGB：从 24k 真实图到条件证书</h2>
{table(['步','输入','做了什么','输出 / 产物','检查 / 为什么'],[
 ['0. 协议','任务需求','固定状态 <code>(d,v)</code>、动作 <code>a∈[-3,3]</code>、<code>d′ = d - 0.05v</code>、<code>v′ = clip(v - 0.05a, 0, 3)</code>，初态 <code>[15,16]×[2.5,3]</code>、危险 <code>[5,6]×[0.5,3]</code>','协议与阈值','先固定任务，后看结果；不按结果改阈值'],
 ['1. 数据','CARLA 保存的 640×640 RGB 和 labels.csv','按 0.25 m 距离 bin 分组；同一 bin 不跨 train/val/test；只用 train 拟合前端','24,000 张；train 14,172 / val 4,908 / test 4,920','避免同场景泄漏；test 不参与选择'],
 ['2. 前端','真实 RGB','冻结 DINOv2-S/14；输入 224；CLS 384 → 384-128-128-1；目标 <code>d/S</code>，S=3.18272066','observation.pt；observations.npz','test MAE 0.0874 m，max 0.525 m'],
 ['3. 单帧 CP','validation 的 <code>d̂-d</code>','归一化残差；有限样本分位 q=0.072493；物理半径 <code>r=S·q=0.230725</code>','静态合同','旧合同误把 q 当米，实际只验证 ±0.07249；已撤回'],
 ['4. 回放','图片库、原 PPO、400 初态','每条轨迹固定一个天气/颜色；在该组内按距离最近邻查图；不生成新图','全库/Test 各 12×400 条','全库 4,800/4,800；test 4,800/4,800 成功；查询误差最大 0.003/0.878 m'],
 ['5. 轨迹 CP','199 条回放轨迹','每条取 <code>max_t |S·z_t-d_t|</code>，包含前端误差 + 最近邻查图误差；先冻 q 再采 100 条','轨迹合同 0.4183585684 m','校准 190/199；旧留出 99/100；旧 0.2307 m tube 结论撤回'],
 ['6. PPO','真实图像距离观测 + 实测速度','使用原已认证 AEBS PPO；输入 <code>[d̂/S,v]</code>，输出制动；本包不重训 actor','policy.pt','前端替换与控制器重训分离；行为评估用冻结 actor'],
 ['7. SBC','原 actor、真实前端、轨迹 CP、旧候选反例','非负 <code>[2,16,8,1]</code>，Tanh 隐藏/ReLU 输出；固定 actor/前端，热启动 v1，再在轨迹半径上训练 v2；每轮 4,000 更新，反例反馈','sbc.pt','初始上界 0；危险下界 25.213636；不是直接沿用旧 SBC'],
 ['8. tube 验证','SBC v2、全部 live 状态、CP 球','自适应状态盒；actor 用仿射界（8 段），后继速度 64 段；每个盒检查所有允许后继；不缩小半径','verification_report.json','292,024 区间；0 见证/0 未解决；深度 29'],
 ['9. 行为','冻结模型、12 组图片','全库/test 回放；每组 400 初态；记录 success/unsafe/timeout','grouped_behavior.json','全库 4,800/4,800；test 4,800/4,800'],
 ['10. 状态相关对照','同一 actor/SBC、Bernstein b(d)','5,000 train 残差拟合 b(d)；199 条轨迹取 q；半径 q·b(d)；直接验证','状态相关包','平均 0.3217、max 0.7189；41,070 区间 0/0；留出 92/100，低于常量版；不替换主包'],
])}</div>

<div class="warnbox"><b>历史灰度实现：</b>原 10/29 的 400 张 32×32 灰度包（DINO + 状态相关 CP + 配套 SBC，4,971,398 区间 0/0、1441/1441）是独立早期链路；不能与上面的 24k 同测试表横比。详见<a href="compare.html">方法对比页</a>。</div>
<div class="section"><h2>车道保持：从渲染图到轨迹 CP + SBC</h2>
{table(['步','输入','做了什么','输出 / 产物','检查 / 为什么'],[
 ['0. 协议','任务需求','固定 <code>(e,θ)</code>、前轮转角 φ、v=4、L=2.6、dt=0.1、H=80；初态 ±0.25/±0.06；安全 ±0.85/±0.30；κ∈{-0.01,0,0.01}','协议','先固定任务和曲率集合'],
 ['1. 场景/数据','渲染器','60 个名义场景 ×60 个状态；192×192 图；按场景划分 2,160/540/540/360','3,600 张图；scenes.json','场景组不泄漏'],
 ['2. 观测','patch tokens','冻结 DINOv2-S/14 16×16 tokens → 卷积 → 空间 soft-argmax → (ê,θ̂)；只用 train 训练','patch_projection.pt','test e MAE 0.0211 m，θ 0.0059 rad'],
 ['3. PPO','观测 (ê,θ̂,κ)','原图像 PPO 80k 步；观测误差扰动训练；本包不重训策略','actor.pt','策略固定；行为与证书分开'],
 ['4. 单帧 CP','validation 残差','max-norm split CP；得到 0.072；视觉证书用 0.08 留余量','单帧合同','只作参考；不拿单帧覆盖代替轨迹覆盖'],
 ['5. 轨迹 CP','199 条图像闭环轨迹','每条 80 步 ×2 坐标取最大归一化误差；先冻 q 再采 100 条留出','0.0900931580','199 校准 190/199；100 留出 95/100；帧覆盖 99.9375%'],
 ['6. SBC','真实状态、CP 盒、三个曲率','AEBS 同类非负 [2,16,8,1]；Tanh/ReLU；随机初始化训练；初始/危险/后继不增','sbc.pt','初始 0.085828；危险 23.6535；0 见证/0 未解决'],
 ['7. 连续域审计','SBC、CP 盒、actor','自适应 40×40 起，深度 18；仿射界 + 普通 IBP 复核；三个曲率','certificate.json / bridge.json','0 实际见证；0 未解决；数值区间不是形式化浮点'],
 ['8. 状态相关 v2','原 199 条轨迹','前 99 条拟合 degree-1 Bernstein b(e,θ)；后 100 条校准；有限样本秩 96；无 cap；留出分数只读留出自身 errors；实际 r(x) 集合直接验证','v2 合同 + 证书','13,000 区间 0/0；校准 96/100；旧留出回归 93/100；本地/服务器各 31 项测试；C 固定；v1 仍撤回'],
 ['9. 图像闭环','patch-DINO、PPO、渲染器','每个状态现渲染图；6 个测试场景 ×2 回合','评估 JSON','12/12 成功；CP 覆盖 98.1–100%'],
])}</div>

<div class="section"><h2>停车场：受保护闭环 + 启动 CP</h2>
{table(['步','输入','做了什么','输出 / 产物','检查 / 为什么'],[
 ['0. 场景','4D 车辆、15 个起点','固定地图、障碍、目标、RK4、dt=0.15、H=30','协议','证明域只针对注册起点和模型'],
 ['1. 启动图像','每个起点组','组内 ±0.02 初态；重新渲染 16 拟合/32 校准/16 审计 + 中心检查','975 张图','拟合/校准/审计分离；审计不调半径'],
 ['2. 启动 CP','32 校准/组','每组一次联合 split CP；得到 b_j,r_j','startup_model.json','审计覆盖 233/240'],
 ['3. 初始信念','启动观测 + 误差集合','保留能解释观测的起点组；不把真实组给控制器','增广初始状态集合','控制器不能偷看真实起点'],
 ['4. 后备控制','参考、延迟观测器、PPO 建议','LQR/PPO 建议先经过在线区间后备检查；不通过则选安全动作','闭环控制器','本包强制在线后备；无后备证书不适用'],
 ['5. SBC','增广状态、时钟、参考编号','几何 ReLU + 有界学习头 + 线性时钟；连续盒外包 + 对抗回归','neural_sbc.pt','初始 ≤0.305；危险 ≥20；非终端漂移 ≤−0.005'],
 ['6. 图像实验','15 起点 ×2 组','30 步图像闭环','validation.json','30/30 成功；0 碰撞/0 越界；59/59 测试'],
])}</div>

<div class="section"><h2>移动机器人：任务成功、证书失败</h2>
{table(['步','输入','做了什么','输出 / 产物','检查 / 为什么'],[
 ['0. 场景','6×4 m 地图','机器人、中央障碍、目标、固定俯视相机；120 步、dt=0.1','协议','状态 3D，动作 2D'],
 ['1. 数据','随机位姿','渲染 224×224；训练 2,400 / 验证 400 / 开发测试 400','图像集','开发测试不冒充最终 CP 数据'],
 ['2. 观测','DINO patch','patch 特征 + 卷积 + 双空间注意力 → 3D 位姿','perception.pt','x 2.55 cm、y 1.86 cm、θ 4.49°'],
 ['3. PPO','3D 观测','30 万步；宽初态 + 绕障距离奖励；双动作','actor.pt','名义 300/300；±0.03 停稳 275/300'],
 ['4. 图像闭环','渲染图 → DINO → PPO','24 条开发轨迹','轨迹图','24/24 安全到达并停稳'],
 ['5. 轨迹 CP','39 校准 + 20 留出','每条 120 步 ×3 坐标最大归一化误差；冻结 q','半宽 (0.0488,0.0488,0.3254)','留出覆盖 18/20；20/20 安全'],
 ['6. SBC','真实状态、CP 盒','非负网络训练；区间审查','SBC 候选','初始 0.179 / 危险 20.807，但不增条件未通过；7,360 未解决'],
 ['7. 反例','CP 盒内扰动搜索','可微搜索 + float64 独立重放','attack report','28/32 违反；21 条圆重叠；固定控制器+盒下不能只靠继续训 SBC'],
])}</div>

<div class="warnbox"><b>读法：</b>每个实验的“通过”都只在其声明的模型、数据分布和数值验证范围内成立；步骤表里的输出不是无条件安全率。完整证据在“数据与报告”页链接的结果包中。</div>
"""

# ---------------- compare ----------------
compare_body=f"""
<p>不同实验的任务、数据和证明域不同，不能用一个分数横比。本页把比较分成三层：<b>同一 test 的前端误差</b>、<b>同一 actor/SBC 的包络敏感性</b>、<b>四个实验的最终状态</b>。</p>
<div class="warnbox"><b>读表规则：</b>AEBS 前端表的 CP 是静态图片级 95% 半径，不是轨迹级 0.418359 m；cGAN 是旧 400 图检查点；灰度 A1 同时改了颜色和源分辨率；原 10/29 灰度包是 400 图早期实现，不在 24k 同一 test 表内。所有实验都不能用平均误差直接推出安全证书。</div>

<div class="section"><h2>1. AEBS 前端误差：同一 4,920 张 test 的正确版本</h2>
{table(['方法','训练/数据','test MAE ↓','RMSE','P95 abs','max abs','静态 CP95 ↓','test 覆盖'],[
 ['cGAN + state_net','旧 400 图生成库检查点','2.747 m','3.320 m','6.480 m','8.342 m','5.221 m','87.64%'],
 ['32×32 灰度 DINO','24k，历史运行','0.1017 m','0.1842 m','0.4670 m','0.9651 m','0.2656 m','90.47%'],
 ['32×32 灰度 DINO','24k，本轮重训可重载','0.1079 m','0.2010 m','0.5514 m','1.1607 m','0.2797 m','90.75%'],
 ['<b>真实 RGB DINO</b>','24k，当前','<b>0.0874 m</b>','<b>0.1211 m</b>','<b>0.2614 m</b>','<b>0.5250 m</b>','<b>0.2307 m</b>','<b>93.33%</b>']])}
<p class="small">同一 0.25 m bin 划分，train/val/test=14,172/4,908/4,920。RGB 相对灰度历史 MAE 低约 16.4%，相对灰度重训低约 19.0%；这是单次运行描述，不是多种子显著性。rain 分组灰度 MAE 约 0.0725 m，RGB 约 0.0967 m，RGB 并非所有天气都更好。</p>
</div>

<div class="section"><h2>1.1 原 SPVC cGAN 原生闭环 vs DINO</h2>
<table><thead><tr><th>项目</th><th>原生 cGAN</th><th>原生 cGAN + 新 actor</th><th>新 DINO</th></tr></thead><tbody>
<tr><td>轨迹 q</td><td><b>1.9156 m</b></td><td>1.9156 m</td><td><b>0.4184 m</b></td></tr>
<tr><td>留出覆盖</td><td>92/100</td><td>92/100</td><td><b>99/100</b></td></tr>
<tr><td>留出行为</td><td>100/100 成功</td><td>100/100 成功</td><td>4800/4800 成功</td></tr>
<tr><td>静态误差</td><td>MAE 0.2219 m / max 1.9839 m（生成图）</td><td>同左</td><td>MAE 0.0874 m / max 0.5250 m（真实图）</td></tr>
</tbody></table>
<p class="small"><b>5.221 m 不是原生 cGAN 的 q</b>，那是旧 state_net 迁移到 24k 真实灰度图的跨库结果。原生 cGAN q=1.9156 m。两边行为都成功，收益在误差包络/覆盖，不在任务成功率。</p>
<table><thead><tr><th>同 400 图预算、同 actor</th><th>cGAN 400</th><th>DINO 400</th><th>结论</th></tr></thead><tbody>
<tr><td>轨迹 q / 留出覆盖</td><td>1.9156 m / 92/100</td><td>4.0205 m / 94/100</td><td>cGAN q 更小；DINO 覆盖高 2pp；没有 DINO 架构胜出</td></tr>
</tbody></table>
<p class="small">24k DINO 的 q=0.4184/99 是数据规模收益；要判定同预算架构胜负，需要把 cGAN 也重训到 24k。</p>
<img src="../artical-F122/results/aebs_cgan_frontend_error_benefit_20261110/error_by_distance.png" alt="cGAN vs DINO error by distance" style="max-width:100%;border:1px solid #d9dee3;border-radius:8px">
<p><a href="../docs/current/SPVC前端替换_收益证明_原SPVC_cGAN误差对照_20261110.md">完整原 SPVC cGAN 对照</a></p>
</div>

<div class="section"><h2>2. 原来的 10/29 灰度实现：单独的历史基线</h2>
{table(['项目','原 10/29 灰度包'],[
 ['数据','400 张 32×32 灰度图；240 / 80 / 80'],
 ['观测','冻结 DINOv2 ViT-S/14 CLS 384 → 128 → 128 → 1；最佳 epoch 243'],
 ['test 误差','MAE 0.129266 m；RMSE 0.165444 m；max 0.414889 m（80 test）'],
 ['CP','状态相关 Bernstein b0(d)；q=2.522199；均值半径约 0.4316 m；40 条校准'],
 ['SBC','非负 [2,16,8,1]；初始 0；危险下界 24.965932'],
 ['连续域 / 回放','4,971,398 / 105,608 区间；均 0 反例、0 未解决'],
 ['行为','名义 / 全图 / 留出各 1441/1441；32 项测试'],
])}
<p>这个包不能和 24k test 横比，也不能把它的 24.9659 危险下界套到当前 RGB 包。它作为“早期灰度链路能闭合”的历史证据保留。 <a href="../artical-F122/results/vision_sls_spvc_sbc_pass_20261029/README.md">原灰度包</a></p>
</div>

<div class="section"><h2>3. AEBS 包络与 SBC：固定 actor 的敏感性</h2>
{table(['误差集合','半径 / 统计','同一旧 SBC','结论'],[
 ['RGB 静态','0.230725 m','212 见证；150,033 未解决','不通过'],
 ['灰度历史静态','0.265584 m','202 见证；128,858 未解决','不通过'],
 ['灰度重训静态','0.279712 m','179 见证；141,107 未解决','不通过'],
 ['cGAN 旧检查点','5.221404 m','624 见证；59,278 未解决','不通过'],
 ['<b>RGB 轨迹 CP + 新 SBC v2</b>','0.418359 m','<b>292,024 区间；0/0；深度 29</b>','<b>通过</b>'],
 ['RGB 状态相关 CP（C 固定）','均值 0.3217 / max 0.7189 m','41,070 区间；0/0；深度 11','通过；留出 92/100；不替换常量'],
])}
<p class="small">旧 SBC 对四种误差集合都失败；新 SBC v2 是针对 RGB 轨迹半径重训的，不能反过来说灰度/cGAN 也已通过。状态相关只学 b(d)，C 仍是 [1,0]，不是联合学习新前端。</p>
</div>

<div class="section"><h2>4. 车道：常量盒与状态相关 v2</h2>
{table(['项目','常量盒（主结果）','状态相关 v2（修复对照）'],[
 ['拟合 / 校准','—','99 / 100 条，索引不重叠'],
 ['阈值','0.090093，190/199','q=1.7205338，第 96/100 秩；无 cap'],
 ['校准覆盖','190/199','96/100'],
 ['旧留出回归','95/100','93/100（旧数据回归，不是新盲测）'],
 ['平均 / 观测最大半径','0.090093 / 0.090093','0.0883927 / 0.1093692'],
 ['SBC','原审计 0/0','13,000 区间；三曲率 0/0'],
 ['同一 100 条校准常量对照','—','0.0886653；旧留出 94/100'],
])}
<p class="small">v2 已修复 v1 的留出残差误用、同批拟合+校准和 cap 问题；但它是旧数据回归，且没有明显优于常量盒。v1 仍为 withdrawn/unverified。</p>
</div>

<div class="section"><h2>5. 联合学习视觉投影与 C（11/09 独立版）</h2>
<table><thead><tr><th>实验</th><th>训练</th><th>新 CP</th><th>SBC / 当前状态</th></tr></thead><tbody>
<tr><td>车道联合 C</td><td>5 对训练种子；每对固定/联合 C 同预算、同初始化</td><td>q 均值：固定 <b>0.085263</b>，联合 <b>0.085699</b>；留出覆盖 95.6% vs 95.0%</td><td>两边各 5/5 通过数值 SBC；3/5 对联合 q 更小；<b>无稳定优势</b>。[阶段结果](../docs/current/SPVC前端替换_收益证明_阶段一至二结果_20261110.md)</td></tr>
<tr><td>RGB AEBS 联合 C</td><td>C≈[1.0,−8.44×10⁻⁶]；速度项接近零</td><td>q=0.394521 m；校准 190/199；<b>留出 91/100</b>；299 条成功</td><td>102,338 区间，0/0，数值 SBC 通过；但留出未达 95/100</td></tr>
<tr><td>灰度 AEBS 联合 C v2</td><td>C≈[0.999998,0.0019307]；视觉尺度 3.237806 m</td><td>q=3.515916 m；校准 190/199；留出 95/100；299 条成功</td><td><b>未通过</b>：4,096 个深度 0 区间中 78 个见证、74 个未解决</td></tr>
</tbody></table>
<p class="small">联合版把视觉投影和 C 一起训练，但使用独立 CP/SBC；不继承固定 C 版任何证书。车道已完成 <b>5 对训练种子配对</b>：q 均值固定 0.085263 / 联合 0.085699，留出覆盖 95.6% / 95.0%，两边各 5/5 通过 SBC；联合 C <b>没有稳定优势</b>。RGB 联合版“SBC 通过”不等于“留出覆盖 95%”；灰度联合版已经失败。详见 <a href="../docs/current/SPVC前端替换_收益证明_阶段一至二结果_20261110.md">阶段结果</a>。</p>
<img src="../artical-F122/results/lane_frontend_benefit_20261110/summary_5seed/q_coverage.png" alt="5-seed q/coverage" style="max-width:100%;border:1px solid #d9dee3;border-radius:8px;margin-top:8px">
</div>

<div class="section"><h2>6. 四个实验放在一起看</h2>
{table(['实验','任务 / 观测','CP / 误差集合','SBC / 验证','行为','当前状态'],[
 ['AEBS RGB','2D 状态；1D 距离；24k 真实 RGB','轨迹 0.418359 m','292,024 区间；0/0','4,800/4,800','条件证书；C 固定'],
 ['AEBS 10/29 灰度','2D；400 张 32×32；DINO','状态相关；4,971,398 区间','0/0；105,608 回放 0/0','1441/1441','历史小数据包；非当前同测'],
 ['车道','2D (e,θ)；3,600 渲染','0.090093 轨迹 CP','0 见证/0 未解决','12/12 图像闭环','证书通过；三个曲率'],
 ['停车场','4D / 3D；15 起点','启动 CP；233/240 审计','增广可达域神经 SBC','30/30','受保护闭环；必须在线后备'],
 ['移动机器人','3D / 3D；2,400 图','约 (0.0488,0.0488,0.3254)','SBC 未通过；7,360 未解决','24/24','任务跑通；完整证书失败'],
])}
</div>

<div class="section"><h2>7. 不能从这张页推出的结论</h2>
<ul>
<li>不能把 cGAN 旧检查点当作同预算架构对比；它只说明迁移参考。</li>
<li>不能把 32×32 灰度 vs 640×640 RGB 的差异全归因于颜色；源分辨率和预处理同时变了。</li>
<li>不能把静态 CP 0.2307 m 和轨迹 CP 0.418359 m 混成一个“前端 CP”。</li>
<li>不能把原 10/29 灰度包的 24.9659 或 1441/1441 直接套到当前 24k RGB 包。</li>
<li>不能把“新 SBC 在 RGB 轨迹半径通过”推成灰度/cGAN 也通过；需要各自重训/验证。</li>
<li>不能把状态相关 CP 的平均半径更小写成覆盖更好；AEBS 92/100、车道 93/100 都低于常量对照。</li>
<li>不能把联合 C 版和固定 C 版混用；联合 C 版有自己的 CP/SBC。车道 5 对配对已完成，但联合 C 没有稳定优势。</li>
</ul>
<p>完整证据：<a href="../docs/current/AEBS前端对比_cGAN_灰度_真实RGB_20261108.md">AEBS 前端对比文档</a> · <a href="../artical-F122/results/real_aebs_baselines/README.md">基线与 JSON</a> · <a href="../artical-F122/results/vision_sls_spvc_sbc_pass_20261029/README.md">原灰度包</a></p>
</div>
"""

# ---------------- data ----------------
data_body=f'''
<p>这里只列配置、结果包、关键 JSON 和完整报告入口；原始图片/大文件仍在服务器和结果包中。方法层面的统一比较见 <a href="compare.html">方法对比页</a>；联合 C 专题见 <a href="../docs/current/联合学习视觉投影与C_实验步骤与结果_20261109.md">联合学习视觉投影与 C</a>。</p>
<div class="section"><h2>实验配置与结果包</h2>
{table(['实验','数据','状态/观测/动作','结果包','关键文件'],[
 ['AEBS RGB','24,000 张 640×640 RGB；4 天气×3 颜色；5–16 m','2D / 1D 距离 + 实测速度 / 1D','<a href="../artical-F122/results/real_aebs_repair_20261109_release/README.md">release</a>','<a href="../artical-F122/results/real_aebs_repair_20261109_release/perception_contract.json">CP contract</a> · <a href="../artical-F122/results/real_aebs_repair_20261109_release/verification_report.json">verification</a>'],
 ['AEBS 状态相关 CP','同上','1D 距离','<a href="../artical-F122/results/real_aebs_state_cp_20261109_units_v2/README.md">state CP</a>','<a href="../artical-F122/results/real_aebs_state_cp_20261109_units_v2/perception_contract.json">contract</a> · <a href="../artical-F122/results/real_aebs_state_cp_20261109_units_v2/verification_tube/report.json">verify</a>'],
 ['AEBS 前端对比','同一 4,920 test','1D','<a href="../artical-F122/results/real_aebs_baselines/CORRECTION.md">baselines（tube 已更正）</a>','<a href="../artical-F122/results/real_aebs_baselines/comparison.json">comparison JSON</a><br><span class="small">前端误差可用；其中旧 tube 字段已被修复版 status.json 取代</span>'],
 ['原 SPVC cGAN 原生','400 图生成体系；cGAN 生成图','2D / 1D 距离 / 1D','<a href="../artical-F122/results/aebs_cgan_frontend_error_benefit_20261110/native_cgan/report.json">native cGAN</a>','q=1.9156 m；留出92/100；行为100/100'],
 ['DINO-400 同预算','与 cGAN 同 400 图；同新 actor','2D / 1D','<a href="../artical-F122/results/aebs_cgan_frontend_error_benefit_20261110/dino400_matched/report.json">DINO-400</a>','q=4.0205 m；留出94/100；行为100/100'],
 ['AEBS 原 10/29 灰度','400 张 32×32；240/80/80','2D / 1D 距离 / 1D','<a href="../artical-F122/results/vision_sls_spvc_sbc_pass_20261029/README.md">原灰度包</a>','<a href="../artical-F122/results/vision_sls_spvc_sbc_pass_20261029/perception_contract.json">state CP</a> · <a href="../artical-F122/results/vision_sls_spvc_sbc_pass_20261029/summary.json">summary</a>'],
 ['AEBS 灰度重训','24k 灰度，可重载','1D','<a href="../artical-F122/results/real_aebs_repair_20261108_v1/gray_retrained.json">gray_retrained.json</a>','<a href="../artical-F122/results/real_aebs_repair_20261108_v1/gray_retrained.pt">weights</a> · <a href="../artical-F122/results/real_aebs_repair_20261108_v1/tube_gray_retrained/report.json">tube</a>'],
 ['联合 C：车道训练','3,600 渲染；C 可训练','2D','<a href="../artical-F122/results/lane_vision_sls_joint_C_20261109_v1/report.json">lane joint train</a>','<a href="../artical-F122/results/lane_vision_sls_joint_C_20261109_cp_v1/perception_contract.json">单对历史 q=0.084338</a>'],
 ['前端收益 5 对配对','固定/联合 C；同数据/预算/外部条件','2D','<a href="../artical-F122/results/lane_frontend_benefit_20261110/README.md">benefit 5-seed</a>','<a href="../artical-F122/results/lane_frontend_benefit_20261110/summary_5seed/report.md">summary</a>'],
 ['联合 C：RGB AEBS','24k RGB；C 可训练','2D / 1D','<a href="../artical-F122/results/rgb_aebs_vision_sls_joint_C_20261109_cp_v1/report.json">RGB joint CP</a>','<a href="../artical-F122/results/rgb_aebs_vision_sls_joint_C_20261109_cp_v1/verification/report.json">SBC 102,338 0/0</a>；留出 91/100'],
 ['联合 C：灰度 AEBS v2','24k 灰度；C 可训练','2D / 1D','<a href="../artical-F122/results/gray_aebs_vision_sls_joint_C_20261109_cp_v2/README.md">gray joint</a>','<a href="../artical-F122/results/gray_aebs_vision_sls_joint_C_20261109_cp_v2/verification_witness/report.json">SBC 未通过</a>；78 见证/74 未解决'],
 ['车道','3,600 张 192×192 渲染；60 场景','2D / 2D (ê,θ̂)+κ / 1D','<a href="../artical-F122/results/lane_trajectory_cp_20261107_v1/README.md">trajectory CP</a>','<a href="../artical-F122/results/lane_repair_20261109_audit/bridge.json">audit bridge</a>'],
 ['车道状态相关 CP v2','复用 199 条轨迹；99/100 分离','2D','<a href="../artical-F122/results/lane_state_cp_20261109_repaired_v2/README.md">repaired v2</a>','<a href="../artical-F122/results/lane_state_cp_20261109_repaired_v2/perception_contract.json">contract</a> · <a href="../artical-F122/results/lane_state_cp_20261109_repaired_v2/verification_state/certificate.json">state cert</a>'],
 ['车道状态相关 v1（已撤回）','旧错误版','2D','<a href="../artical-F122/results/lane_state_cp_20261109_v1/AUDIT.md">audit</a>','不可引用；代码/报告不一致'],
 ['停车场','15 起点组；启动 975 张图像','4D / 3D / 2D','<a href="../artical-F122/results/vision_sls_parking_20261103_neural_sbc_certificate_v1/README.md">package</a>','<a href="../artical-F122/results/vision_sls_parking_20261103_neural_sbc_certificate_v1/certificate.json">certificate</a>'],
 ['移动机器人','2,400+400+400 张 224×224 渲染','3D / 3D / 2D','<a href="../artical-F122/results/mobile_robot_20261108_server_trajectory_cp_r1/report.json">trajectory CP</a>','<a href="../artical-F122/results/mobile_robot_20261108_server_attack_cp/report.json">attack</a> · <a href="../artical-F122/results/mobile_robot_20261108_server_sbc_r6/report.json">SBC summary</a>'],
])}</div>
<div class="section"><h2>完整文档</h2><ul>
<li><a href="../docs/current/README.md">docs/current 当前入口</a></li>
<li><a href="../docs/current/当前方案与验证结果.md">当前方案与验证结果</a></li>
<li><a href="../docs/current/AEBS真实RGB图像_完整实验_20261108.md">AEBS RGB 修复后完整实验</a></li>
<li><a href="../docs/current/AEBS前端对比_cGAN_灰度_真实RGB_20261108.md">AEBS 前端对比</a></li>
<li><a href="../docs/current/车道保持_轨迹级CP校准与SBC衔接_20261107.md">车道轨迹 CP 与 SBC</a></li>
<li><a href="../docs/current/停车场神经SBC_构造证明与实验_20261103.md">停车场神经 SBC 专题</a></li>
<li><a href="../docs/current/移动机器人_三维视觉实验_20261108.md">移动机器人专题</a></li>
</ul></div>
<div class="section"><h2>原始数据位置</h2><div class="kv">
<div>AEBS RGB</div><div><code>/root/work_based_on_spvc/carla_data_spvc_expanded/full01</code>（服务器，约 17 GB）</div>
<div>车道</div><div>由 <code>Aebs/lane_keeping/renderer.py</code> 在结果包中复现；完整 v2 数据在服务器</div>
<div>停车场</div><div>结果包 <code>vision_sls_parking_20261103_neural_sbc_certificate_v1/</code></div>
<div>移动机器人</div><div><code>/root/work_based_on_spvc_robot_20261108</code>（服务器）</div>
</div></div>
<div class="warnbox">站点不复制 24,000 张原图和 17 GB 数据；只提供缩略图、配置和结果包链接。</div>
'''

page('compare.html','方法对比：前端、包络与证书','compare.html',compare_body,'同一 test 的前端误差、历史灰度实现、CP/SBC 敏感性，以及四个实验的最终状态。')
page('process.html','实验过程：逐步输入/输出','process.html',process_body,'四个实验从协议、数据、观测、CP、SBC 到验证和行为的详细步骤。')
page('index.html','SPVC 实验总览','index.html',index_body,'真实图像前端、低维观测、CP 包络与非负 SBC：四个实验的配置、步骤、结果与边界。')
page('aebs.html','AEBS：真实 RGB 紧急制动','aebs.html',aebs_body,'640×640 真实 CARLA RGB → DINO 距离 → CP → 新非负 SBC；条件证书与单位修复。')
page('lane.html','车道保持：2D 视觉与轨迹 CP','lane.html',lane_body,'patch-DINO (ê,θ̂) → PPO → 学习式 SBC；轨迹 CP 0.090093，150+100 轨迹。')
page('parking.html','停车场：受保护闭环神经 SBC','parking.html',parking_body,'4D 车辆、3D 视觉、启动 CP 与在线后备；增广可达域证书。')
page('robot.html','移动机器人：三维视觉实验','robot.html',robot_body,'图像闭环跑通，但 CP 盒内有实际反例；完整 SBC 证书未通过。')
page('gallery.html','图片库：精选输入与结果','gallery.html',gallery_body,'只放缩略图；原始数据在结果包/服务器。')
page('data.html','数据、配置与报告','data.html',data_body,'实验配置、结果包、关键 JSON 和完整报告入口。')
print('generated', [p.name for p in sorted(ROOT.glob('*.html'))])
