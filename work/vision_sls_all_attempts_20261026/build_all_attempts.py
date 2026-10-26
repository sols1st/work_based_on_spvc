"""Consolidate all recorded VISION-SLS experiments without inventing missing runs."""
import hashlib,json,math
from pathlib import Path

repo=Path('/Users/johann/project/work_based_on_spvc')
base=Path('/Users/johann/Documents/Codex/2026-10-22/jian/outputs')
output=base/'vision_sls_全部尝试结果汇总_20261026';output.mkdir(exist_ok=True)
cons={'parking':base/'vision_sls_parking_direct_cp_20261025','quad':base/'vision_sls_direct_cp_20261025'}
add=base/'vision_sls_状态相关CP验证_20261026'
doc=base/'vision_sls_建议原方案CP验证_20261026'
initial={'parking':base/'vision_sls_10server_20261024','quad':base/'vision_sls_quad_10server_20261024'}
sources={}
def read(p):
    p=Path(p);raw=p.read_bytes();sources[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
def link(label,p):return f'[{label}](<{p}>)'
def pct(value):return f'{value*100:.2f}%'
def fmt(value):return '—' if value is None else f'{value:.6g}' if isinstance(value,float) else str(value)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def scene(s):return '停车场' if s=='parking' else '无人机'
constant={s:read(p/'direct_cp_comparison.json') for s,p in cons.items()}
additive={s:read(add/s/'state_cp_comparison.json') for s in initial}
document={s:read(doc/s/'doc_cp_comparison.json') for s in initial}
retro=read(base/'vision_sls_cp_feasibility_20261024/feasibility_summary.json')
refit=read(initial['parking']/'calibration/summary.json')
initial_coverage=read(Path(__file__).parent/'initial_reference_coverage_recheck.json')
additive_qp=read(add/'quad/qp_audit.json')
additive_adaptation=read(add/'quad/solver_adaptation.json')
records=[]
def record(id,s,name,phase,result,metrics=None,path=None,**extra):
    row={'id':id,'scene':s,'name':name,'phase':phase,'result':result,'metrics':metrics,'evidence':str(path) if path else None,**extra};records.append(row);return row

for s,id in [('parking','R01'),('quad','R02')]:
    m=read(initial[s]/'visual_rollouts/summary.json')
    record(id,s,'原方法复现','规划＋图像闭环','完成 50 条，任务成功 50，碰撞 0，实际联合违规 0',m,initial[s]/'visual_rollouts/summary.json')
record('R03','parking','原混合整数包络重拟合','离线拟合','250 点拟合覆盖 97.60%；4750 点拟合外覆盖 90.91%；未替换闭环基线',refit,initial['parking']/'calibration/summary.json')

offline=[]
for s,key,ids in [('parking','parking_lot',['E01','E02']),('quad','quadrotor',['E03','E04'])]:
    source=retro['scenes'][key]
    for norm,id in zip(['linf_pdf','l2_matches_code'],ids):
        name='无穷范数' if norm=='linf_pdf' else '二范数'
        evaluations=source['scores'][norm]['evaluations']
        record(id,s,'旧界归一化残差：'+name,'回顾算术','90%、95%、99% 三个目标；99% 得到无穷大界，无新图像闭环',path=base/'vision_sls_cp_feasibility_20261024/feasibility_summary.json',evaluations=evaluations)
        for e in evaluations:
            offline.append([id,scene(s),name,pct(e['target_coverage']),len(source['calibration_rollouts']),e['rank_k'],fmt(e['scale_q']) if e['finite'] else '+∞（样本不足）',f"{e['heldout_trajectories_covered']}/{e['heldout_trajectory_count']}",pct(e['heldout_point_coverage']),pct(e['axis_stress_trajectory_coverage'])])
for s,key,id in [('parking','parking_lot','E05'),('quad','quadrotor','E06')]:
    trial=retro['planning_trials'][key]
    record(id,s,'旧界乘以回顾校准倍率','仅重新规划',f"返回成功，倍率 {trial['factor']:.6f}；未执行新控制器视觉闭环",trial,base/'vision_sls_cp_feasibility_20261024'/('planning_trials/parking_l2_95/feasibility.json' if s=='parking' else 'planning_trials/quad_l2_95/feasibility.json'))

for s,id in [('parking','C01'),('quad','C02')]:
    d=constant[s];m=d['methods']['direct_cp']
    record(id,s,'直接常数二范数 CP','重新规划＋图像闭环',f"任务成功 {m['author_successes']}/50，碰撞 {m['collision_rollouts']}，实际联合违规 {m['full_state_input_collision_constraint_violations']}；整轨迹覆盖 {m['own_bound_l2_trajectories_covered']}/50",m,cons[s]/'direct_cp_comparison.json',baseline=d['methods']['baseline'],norm='二范数球',coverage_point=m['own_bound_l2_point_coverage'],coverage_trajectories=m['own_bound_l2_trajectories_covered'])

pa=additive['parking'];pm=pa['methods']['state_cp']
record('A01','parking','新状态预测器＋加法轨迹 CP，原求解器','重新规划＋图像闭环','50/50 成功，0 碰撞，0 实际联合违规；整轨迹覆盖 0/50',pm,add/'parking/state_cp_comparison.json',baseline=pa['methods']['baseline'],norm='二范数球',coverage_point=pm['new_state_bound_point_coverage'],coverage_trajectories=pm['new_state_bound_trajectories_covered'])
audit=read(add/'quad/strict_replacement_failure/qp_audit.json')
additive_strict_qp=audit
record('A02','quad','加法轨迹 CP，严格保持原求解器','规划失败','第一次局部 QP 返回 problem non convex；没有新控制器闭环',path=add/'quad/strict_replacement_failure/qp_audit.json',qp_audit=audit)
qa=additive['quad'];qm=qa['methods']['state_cp']
record('A03','quad','加法轨迹 CP＋局部 Hessian 投影','规划＋诊断闭环','1/50 成功，49 条碰撞，50 条实际联合违规；整轨迹覆盖 0/50',qm,add/'quad/state_cp_comparison.json',baseline=qa['methods']['baseline'],norm='二范数球',coverage_point=qm['new_state_bound_point_coverage'],coverage_trajectories=qm['new_state_bound_trajectories_covered'])

doc_trials=[('D01','parking','', '文档乘法界，原求解器'),('D02','quad','', '文档乘法界，原求解器'),
 ('D03','parking','_iter50000','仅增加 QP 最大迭代数到 50000'),
 ('D04','parking','_convexified_iter50000_outer80','局部凸化比例下限 1e-8，外层 80'),
 ('D05','quad','_convexified_iter50000_outer80','局部凸化比例下限 1e-8，外层 80'),
 ('D06','parking','_convexified_iter50000_outer100_rf0p001','局部凸化比例下限 1e-3，外层 100'),
 ('D07','quad','_convexified_iter50000_outer100_rf0p001','局部凸化比例下限 1e-3，外层 100'),
 ('D08','parking','_convexified_iter100000_outer100_rf0p001_normalized','再作目标归一化、QP 100000 次、容差 1e-7')]
planrows=[]
for id,s,suffix,name in doc_trials:
    path=doc/s;status=read(path/('plan_status'+suffix+'.json'));audit=read(path/('qp_audit'+suffix+'.json'))
    last=audit['records'][-1];comp=path/('doc_cp_comparison'+suffix+'.json')
    d=read(comp) if comp.exists() else document[s];m=d['methods'].get('doc_cp')
    result=(f"任务成功 {m['author_successes']}/50，碰撞 {m['collision_rollouts']}，实际联合违规 {m['full_state_input_collision_constraint_violations']}；盒整轨迹覆盖 {m['doc_box_trajectories_covered']}/50" if m else '无可测试控制器；'+last['solver_status'])
    if status['plan_exported'] and not status['converged']:result='未收敛但因上限被标记成功；'+result
    record(id,s,name,'规划＋诊断闭环' if m else '规划失败',result,m,comp if comp.exists() else path/('plan_status'+suffix+'.json'),
       baseline=d['methods']['baseline'],norm='文档无穷范数盒；控制使用外包球',coverage_point=m['doc_box_point_coverage'] if m else None,coverage_trajectories=m['doc_box_trajectories_covered'] if m else None,
       suffix=suffix,plan_status=status,qp_audit_summary={'calls':len(audit['records']),'projected_calls':sum(r['convexification_applied'] for r in audit['records']),'last':last},
       plan_checks=d.get('plan_checks'))
    planrows.append([id,scene(s),status['convergence_checks'],'是' if status['converged'] else '否',len(audit['records']),last['solver_status'],fmt(last['min_eigenvalue']),'是' if status['plan_exported'] else '否'])

# Each entry is one actual image batch; the original parking baseline is reused.
batches=[('B01','parking','原方法初次复现',initial['parking']/'visual_rollouts',42),('B02','quad','原方法初次复现',initial['quad']/'visual_rollouts',42),
 ('B03','parking','常数 CP 校准',cons['parking']/'calibration',2026102503),('B04','parking','常数 CP 新控制器',cons['parking']/'direct_cp',42),
 ('B05','quad','常数 CP 校准',cons['quad']/'calibration',2026102501),('B06','quad','常数 CP 配对基线',cons['quad']/'baseline',2026102502),('B07','quad','常数 CP 新控制器',cons['quad']/'direct_cp',2026102502)]
for s,prefix,calseed,testseed in [('parking','加法 CP',2026102601,2026102603),('quad','加法 CP',2026102602,2026102604)]:
    for phase,role,seed in [('calibration','校准',calseed),('baseline','配对基线',testseed),('state_cp','新控制器',testseed)]:batches.append((f'B{len(batches)+1:02d}',s,prefix+role,add/s/phase,seed))
for s,calseed,testseed in [('parking',2026102611,2026102613),('quad',2026102612,2026102614)]:
    for phase,role,seed in [('calibration','文档 CP 校准',calseed),('baseline','文档 CP 配对基线',testseed)]:batches.append((f'B{len(batches)+1:02d}',s,role,doc/s/phase,seed))
for suffix,role in [('_convexified_iter50000_outer80','文档 CP 无人机：比例 1e-8 诊断'),('_convexified_iter50000_outer100_rf0p001','文档 CP 无人机：比例 1e-3 闭环')]:batches.append((f'B{len(batches)+1:02d}','quad',role,doc/'quad'/('doc_cp'+suffix),2026102614))
batchrows=[];batchmeta=[];seen=set();total=frames=0
for id,s,role,path,seed in batches:
    meta=read(path/'summary.json');trace=path/'diagnostic_traces.npz';sha=hashlib.sha256(trace.read_bytes()).hexdigest()
    assert sha not in seen,('Duplicate counted image batch',path);seen.add(sha)
    n=meta['rollouts'];steps=meta['steps_per_rollout'];assert steps==(30 if s=='parking' else 35)
    if 'seed' in meta:assert meta['seed']==seed
    total+=n;frames+=n*steps
    batchrows.append([id,scene(s),role,n,steps,seed])
    batchmeta.append({'id':id,'scene':s,'role':role,'rollouts':n,'steps':steps,'seed':seed,'trace_path':str(trace),'trace_sha256':sha})
assert total==990 and frames==32800 and len(batches)==19
assert len({r['id'] for r in records})==len(records)==22
counts_by_scene={s:sum(d['rollouts'] for d in batchmeta if d['scene']==s) for s in ['parking','quad']}
assert counts_by_scene=={'parking':370,'quad':620}
reused=cons['parking']/'baseline/diagnostic_traces.npz'
assert hashlib.sha256(reused.read_bytes()).hexdigest()==batchmeta[0]['trace_sha256']

overview=table(['编号','场景','尝试内容','阶段','结果'],[[r['id'],scene(r['scene']),r['name'],r['phase'],r['result']] for r in records])
metricrows=[];pairedrows=[];evidence=[]
for r in records:
    if not r['metrics'] or 'collision_rollouts' not in r['metrics']:continue
    m=r['metrics'];point=r.get('coverage_point');covered=r.get('coverage_trajectories')
    if r['id']=='R01':point=constant['parking']['methods']['baseline']['own_bound_l2_point_coverage'];covered=0;distance=constant['parking']['methods']['baseline']['terminal_position_distance_mean_m']
    else:distance=m['terminal_position_distance_mean_m']
    if r['id']=='R02':point=initial_coverage['point_coverage'];covered=initial_coverage['trajectories_covered']
    metricrows.append([r['id'],scene(r['scene']),m['author_successes'],m['collision_rollouts'],m['full_state_input_collision_constraint_violations'],m['state_tube_violating_rollouts_tolerance_1e-6'],m['input_tube_violating_rollouts_tolerance_1e-6'],f'{distance:.6f}',pct(point),f'{covered}/50'])
    if 'baseline' in r:
        b=r['baseline'];pairedrows.append([r['id'],f"{b['author_successes']} → {m['author_successes']}",f"{b['collision_rollouts']} → {m['collision_rollouts']}",f"{b['full_state_input_collision_constraint_violations']} → {m['full_state_input_collision_constraint_violations']}",f"{b['terminal_position_distance_mean_m']:.6f} → {distance:.6f}"])
for r in records:evidence.append([r['id'],link('原始记录',r['evidence'])])

fitrows=[];calrows=[]
for s in ['parking','quad']:
    c=read(cons[s]/'direct_cp_calibration.json')
    calrows.append(['常数 CP',scene(s),c['calibration_trajectories'],c['rank_k'],'半径 R',f"{c['radius']:.9f}",pct(c['calibration_trajectory_coverage'])])
    a=read(add/s/'state_cp_model.json');fitrows.append(['加法状态 CP',scene(s),a['training_trajectories'],a['fitting_points'],a['monomials'],'0.9 分位损失＋L1',pct(a['training_point_coverage'])])
    calrows.append(['加法状态 CP',scene(s),a['calibration_trajectories'],a['rank_k'],'偏移 Q',f"{a['cp_offset']:.9f}",pct(a['calibration_bound_trajectory_coverage'])])
    d=read(doc/s/'doc_cp_model.json');fitrows.append(['文档乘法 CP',scene(s),d['training_trajectories'],d['training_points'],d['monomials'],'包络＋L2＋松弛',pct(d['training_point_coverage'])])
    calrows.append(['文档乘法 CP',scene(s),d['calibration_trajectories'],d['calibration_rank'],'倍率 q',f"{d['cp_multiplier']:.9f}",pct(d['calibration_trajectory_coverage'])])

text=rf'''# VISION-SLS：全部尝试结果总账

**汇总日期：2026 年 10 月 26 日。** 本文记录本轮对话中已有证据的 VISION-SLS 复现、CP 误差界和数值适配尝试，包含成功、失败、未收敛输出以及没有执行闭环的版本。本次只整理记录，没有新增实验或改变既有结果。

## 1. 怎样读这份总账

共列出 **{len(records)} 个汇总条目**，其中 E01—E04 各含三个覆盖目标，总计 **12 个回顾算术参数点**；其余包含原方法复现、原界重拟合、重新规划及新控制器测试。同配置的排错重跑没有当作新的独立算法。

19 个独立图像数据批次按文件哈希去重后，共 **990 条轨迹、32800 次图像观测**：停车场 370 条、无人机 620 条。早期 2026-10-23 目录仅保留图像和计划预览，没有足够批次元数据，单独列为准备记录，不计入这些正式统计。

“没有生成控制器”意味着该方案没有闭环成功率、碰撞率和覆盖率；不能填成零。所有正式控制测试均为仿真，没有硬件实机测试。多数闭环限定于原场景第一组初始条件，没有覆盖论文全部初始条件。

**任务成功、无碰撞、实际约束满足、管束覆盖、感知误差覆盖是五个不同指标。** 特别是无人机的作者任务成功口径较宽，不能只用它判断安全。

## 2. 所有尝试一览

{overview}

## 3. 原方法复现及原混合整数包络重拟合

R01 与 R02 保留作者网络、DINO 骨干、原始图像预处理与 float32、GUI/OpenGL 渲染、动力学及控制器。服务器通过 Xvfb 运行。两场景最初各测试 50 条，均为 50/50 作者任务成功、0 碰撞、0 实际联合约束违规。停车场状态／输入管束越界为 3/50 与 50/50；无人机最初为 20/50 与 50/50。管束越界与实际物理约束违规不可混用。

R03 使用作者已有 5000 组状态与特征，抽取 250 点按原停车场混合整数结构重拟合，另 4750 点作本次拟合外评估。求解器为 Gurobi，状态为最优、MIP gap 为 0；常数下界约为 {refit['b_min']:.9f}，z 总和 {refit['z_sum']:.0f}，q 总和 {refit['q_sum']:.0f}。拟合点覆盖 {pct(refit['fit_absolute_residual_coverage'])}，拟合外覆盖 {pct(refit['heldout_absolute_residual_coverage'])}，作者已保存包络在同一拟合外集合的覆盖为 {pct(refit['provided_envelope_heldout_coverage'])}。

这次重拟合只用于检查原生成流程，没有替换闭环基线，也不是后来直接 CP 的旧尺度输入。4750 点虽未参与本次 250 点拟合，但来自作者已有数据，不能认定它们与作者网络或原界训练独立；其离线覆盖也不能与新图像闭环覆盖混为一项。

## 4. 早期旧界归一化：所有覆盖目标均保留

E01—E04 在此前已经检查过的原策略数据上，排除轴向压力轨迹后，重新划分随机轨迹。停车场为 28 条校准、14 条回顾留出；无人机为 20 条校准、10 条回顾留出。分割单位为整条轨迹，不是帧。

分数按每条轨迹的最大归一化残差计算；二范数和无穷范数是两套口径。它们使用旧误差界，区别于后来从原始残差训练新尺度。以下“留出”仍是已看过数据上的回顾拆分，不是新采集的独立测试，不能据此声称新控制器 CP 保证。

{table(['编号','场景','范数','目标','校准轨迹数','k','倍率 q','回顾留出整轨迹覆盖','留出逐点覆盖','压力轨迹覆盖'],offline)}

99% 目标时，停车场 k=29>28、无人机 k=21>20，按标准约定只能得到无穷大界。表中的完全覆盖只是无限集合的平凡结果，不是可用的有限误差界，也没有用这些无穷大界做有限控制规划。

E05 与 E06 只把二范数 95% 回顾倍率交给规划器：停车场倍率 {retro['planning_trials']['parking_lot']['factor']:.9f}，无人机倍率 {retro['planning_trials']['quadrotor']['factor']:.9f}。两者均返回有限计划，动力学最大缺陷分别约为 {retro['planning_trials']['parking_lot']['max_nonlinear_dynamics_defect']:.3g}、{retro['planning_trials']['quadrotor']['max_nonlinear_dynamics_defect']:.3g}；**没有执行对应的新控制器视觉闭环**。所以只能说明当时的规划接口能运行。

## 5. 后来三套 CP 方法怎样区分

| 阶段 | 从什么数据拟合 | 最终界 | 范数与控制接口 | 是否使用旧包络 |
|---|---|---|---|---|
| 常数 CP，C01—C02 | 原始轨迹最大残差直接排序 | R | 二范数球，F=RI | 新界不读取旧包络 |
| 加法状态 CP，A01—A03 | 新二阶分位预测器 g，含姿态特征 | max(0.001,g+Q) | 二范数球 | 新模型不读取旧包络 |
| 文档公式 CP，D01—D08 | 文档式带松弛凸包络 b0 | q*max(0.001,b0) | 无穷范数盒；控制使用 sqrt(ny) 倍外包球 | 新模型不读取旧包络 |

“不读取旧包络”指新误差模型；产生训练／校准数据的原参考控制器仍使用原界。旧数据中的原始残差可复用来训练新的预测器，但复用不等于新增采样。

### 5.1 新尺度的全部拟合结果

{table(['尺度模型','场景','训练轨迹数','训练观测数','系数数','拟合目标','训练点未校准覆盖'],fitrows)}

加法版本停车场使用位置与 sin／cos 方向，15 项；无人机使用五个位置／姿态特征，21 项。文档版本停车场只用二维位置三阶，10 项；无人机五维四阶，126 项；目标按文档示例设 gamma=1e-4、mu=20。训练点非负和最终正尺度截断都不构成整个使用域的误差充足性证明。

### 5.2 每套方法的独立校准结果

{table(['方法','场景','校准轨迹数','k','参数含义','数值','校准集整轨迹覆盖'],calrows)}

R 是半径，Q 是加法偏移，q 是乘法倍率，不能直接比较数值大小。误差界数值属于低维观测空间，不能直接读成米；表中的终点位置距离才以米计。校准集的覆盖主要由所取顺序统计量决定，不是新策略泛化结果。95% 为各候选 CP 方案设置的目标，不是原论文脚本的任务成功验收阈值。

### 5.3 加法状态 CP 也有两个不同的无人机求解版本

A02 的第一轮控制 QP Hessian 最小特征值约为 {additive_strict_qp['records'][0]['min_eigenvalue']:.6g}，底层返回非凸，尚未进行新控制器闭环。这里读取的是该阶段单独保留的失败审计，不能与后来文档版 D02 的数值混淆。

A03 将局部 Hessian 的负特征值截到 1e-6，并在解后恢复原局部成本表达；原始误差模型、加法偏移、动力学和硬约束没有修改，但局部求解过程改变了。共记录 {len(additive_qp['records'])} 次 QP，其中 {sum(x['convexification_applied'] for x in additive_qp['records'])} 次应用投影。它虽输出计划，名义动力学最大缺陷约为 {qa['plan_checks']['max_nonlinear_nominal_dynamics_defect']:.6f}，最终重新线性化的最大 SLS 残差约为 {max(qa['plan_checks']['SLS_identity_max_abs_residuals_relinearized_at_final_plan']):.6f}，随后 49/50 条碰撞；不能将“输出计划”写成数值与安全验证通过。

## 6. 所有完成闭环的方案：统一保留主要指标

以下各行均为 50 条轨迹。R01、R02 使用原界；C 使用常数二范数球；A 使用加法状态二范数球；D 使用文档无穷范数盒计算误差覆盖。**不同集合的覆盖率不宜当作同一指标直接排名。**

{table(['编号','场景','任务成功','碰撞','实际联合违规','状态管束越界','输入管束越界','平均终点距离 m','误差逐点覆盖','误差整轨迹覆盖'],metricrows)}

R01 原始汇总没有直接保存平均目标距离和二范数整轨迹覆盖字段；这里引用与同一 NPZ 哈希一致的常数 CP 配对重算结果。R02 的原界二范数覆盖另外从原始残差和各分量界重算，50 条完整轨迹覆盖均为零，核验记录单独保存。两者没有新增采样。

### 6.1 每个新方案只与自己的同扰动基线比较

{table(['编号','任务成功：原→新','碰撞：原→新','实际联合违规：原→新','平均终点距离 m：原→新'],pairedrows)}

不同阶段的随机种子不同，所以原控制器无人机状态管束越界会出现 19、20、23 等不同数值。不能跨阶段选择有利基线来描述改善；上表每行均采用本方案自身的配对基线。

C01、C02 在本次扰动与初始条件下控制表现较好，但新策略整轨迹覆盖仍只有 9/50、31/50；不能据此宣称概率安全条件已经成立。A01 虽 50/50 成功且无实际违规，整轨迹覆盖却为零。A03 有 49 条碰撞，是必须保留的失败结果。

D05 达到外层迭代上限仍未收敛，源码却返回成功，故其 50 条回放只是诊断。D07 真正通过原步长收敛判据，但仍有 50 条阶段状态约束违规、整轨迹盒覆盖为零；任务成功 50/50 和零碰撞不能掩盖这一事实。

## 7. 文档误差界下的全部数值求解尝试

文档版误差界系数、校准轨迹与倍率在 D01—D08 中保持不变。不同数值配置改变的是局部控制规划；原参数组与适配组分开记录。控制器使用包含文档盒的外包球，并非原生盒集合 SLS。

{table(['编号','场景','外层检查数','原判据收敛','局部 QP 调用数','最终 QP 返回状态','最后 Hessian 最小特征值','输出计划'],planrows)}

### 7.1 每次改了什么

- D01、D02：保留原求解器。停车场最终达到 QP 迭代上限；无人机第一次 QP 返回非凸。不能把两者的最后状态统称为非凸。
- D03：只把停车场 QP 最大迭代数增加到 50000，仍未得到控制器。
- D04、D05：局部 Hessian 特征值投影，下限 max(1e-6,1e-8*最大绝对特征值)，QP 50000 次，外层最多 80 次。停车场失败；无人机未收敛而被上限标记成功，诊断表现较差。
- D06、D07：特征值比例下限改为 1e-3，外层最多 100 次，QP 50000 次。停车场失败；无人机第 10 次检查通过原收敛判据，完成独立记录的闭环。
- D08：在停车场 D06 基础上，对局部二次项和线性项同除正尺度，返回对偶量与代价乘回；QP 最多 100000 次，绝对／相对容差为 1e-7。仍未生成控制器。正比例缩放保持已构造局部 QP 的精确极小点，但不改变原非线性问题的难度。

原始不定 Hessian 的局部二次近似与凸化后的局部模型不同，不能把这些适配写成“求解器完全未变”。负特征值或数值失败也不能直接证明原非线性控制问题无可行解。

### 7.2 D07 的违规具体是什么

无人机模型索引 8 的状态分量允许绝对值不超过 2，本轮最大达到约 2.054589，超出约 0.054589，50 条都有阶段状态违规；输入和终端状态约束没有违规。使用的实际检查容差为 0.001，不能解释成极小舍入误差。

D07 保存计划的名义动力学最大缺陷约 5.03e-10，最终重新线性化的 SLS 最大残差约 3.36e-6，数值一致性比未收敛版本好；但收紧后阶段约束最大残差仍约 0.003276。新策略 1750 个观测中 1420 个落到正尺度下限且未被盒界覆盖，说明数值收敛没有修复误差尺度适用域和策略变化问题。

## 8. 全部实际图像批次与去重统计

{table(['批次','场景','用途','轨迹数','每条观测数','种子'],batchrows)}

停车场常数 CP 的基线目录复用 B01，未再计一个 50。模型训练复用的既有校准数据、早期回顾拆分、仅规划和排错重跑均不重复计入图像轨迹数。

| 阶段 | 新轨迹 | 新图像观测 |
|---|---:|---:|
| 原方法复现＋常数 CP 阶段 | 390 | 12950 |
| 加法状态 CP 阶段 | 300 | 9800 |
| 文档乘法 CP＋后续数值适配 | 300 | 10050 |
| 合计 | 990 | 32800 |

19 份原始轨迹文件的路径、SHA256、种子和数量保存在同目录 JSON；重复的停车场原基线已经做文件哈希一致性核对。这里的图像观测数不是独立样本数，轨迹内存在时间相关性。

## 9. 工程准备与排错记录

这部分也保留，但不计作新算法或成功实验：

| 事项 | 实际情况与处理 | 对结果统计的影响 |
|---|---|---|
| 2026-10-23 早期停车场准备 | 目录只保留首帧与计划预览，缺少完整批次汇总 | 不填推测的成功率，不计入 990 条 |
| 服务器无显示器 | 使用 Xvfb 与 Mesa 保留 GUI/OpenGL | 没有切成 TinyRenderer，也没有改成半精度推理 |
| 文档版首次采集命令 | 初始相对资产路径在工作目录切换后不可用，随后使用绝对路径正式启动 | 路径错误未生成完整数据批次，不重复计数 |
| 文档版首次无人机计划检查 | CasADi 禁止函数名 jac，改成非保留名称后重跑 | 发生在控制 QP 前；不是文档方法或优化可行性失败 |
| 版本／源代码检查 | 某次对复制源码直接查询 Git 遇到所有权／仓库根问题，改与原复现源码逐文件哈希核对 | 未修改科学实验；不将仓库查询失败当作求解失败 |
| 汇总与作图修正 | 修正未完成复制时的文件检查顺序、比较图标题和极端比值坐标显示等 | 数据文件与算法结果未据此改变 |

保留的 CasADi 排错日志见 {link('setup_retry_plan.log',doc/'quad/setup_retry_plan.log')}。未保存完整失败日志的准备项，只记录已知处理，不补造时间、数据或数值结果。

## 10. 总结：全部尝试支持什么

目前已充分确认 CP 误差界的数值计算与控制接口可以实现；常数 CP 在已测条件下控制表现较好，部分状态相关版本也能完成闭环。但**所有已测试 CP 新策略都没有建立目标整轨迹覆盖结论**，文档版进一步显示了使用域不足和数值规划的难点。

不能根据常数 CP 的较好仿真结果声称后来所有状态相关版本同样可靠；不能只报告 D07 的零碰撞；不能把规划失败方案的闭环指标填成零；也不能把当前文档示例拟合＋外包接口的失败推广为完整 CP 方向不可行。

当前没有原生盒集合 SLS 的完整验证，没有整个模型使用状态域的非负性／误差充足性验证，也没有解决参考策略校准结论如何适用于重新优化的新策略。这些限制需要在导师汇报中主动区分。

## 11. 每个条目的证据入口

{table(['编号','主要原始证据'],evidence)}

分阶段详解：{link('复现与常数 CP 报告',base/'vision_sls_复现与CP替换_综合报告_20261025/VISION-SLS_实验复现与直接CP替换_详细报告.md')}；{link('加法状态 CP 与建议差异',add/'VISION-SLS_状态相关CP_服务器验证与导师汇报.md')}；{link('文档公式及所有数值适配报告',doc/'VISION-SLS_按建议原公式CP_详细实验报告.md')}。
'''

report=output/'VISION-SLS_CP全部尝试_完整结果总账.md';report.write_text(text)
(repo/'docs/current/VISION-SLS_CP全部尝试结果总账.md').write_text(text)
summary='''## 11. 全部尝试结果总览（含早期回顾、失败与诊断版本）

本节补齐各阶段记录。所有条目的详细参数、12 个回顾覆盖目标、完整控制指标、19 个去重图像批次及证据链接，见 '''+link('全部尝试完整结果总账',report)+'''。

'''+overview+'''

累计正式图像记录为 990 条轨迹、32800 次观测。停车场常数 CP 基线复用最初原方法数据，只计一次；规划失败、离线重算和工程重试不增加图像轨迹数。不同范数／集合的覆盖指标不可直接混排，方法优劣应优先看同扰动配对结果。
'''
(Path(__file__).parent/'all_attempts_summary.md').write_text(summary)
inventory={'date':'2026-10-26','experiment_records':records,'retrospective_parameter_points':offline,
 'image_batches':batchmeta,'unique_image_batches':len(batches),'unique_visual_trajectories':total,'image_observations':frames,
 'unique_visual_trajectories_by_scene':counts_by_scene,
 'reused_parking_baseline_hash_equal':True,'source_json_sha256':sources,
 'output_reports':{str(report):hashlib.sha256(report.read_bytes()).hexdigest()}}
(output/'全部尝试_结构化记录与来源.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
print(report);print('Records',len(records),'batches',len(batches),'trajectories',total,'observations',frames,'characters',len(text))
