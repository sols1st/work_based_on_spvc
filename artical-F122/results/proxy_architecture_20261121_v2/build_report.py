from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Arial Unicode MS'];plt.rcParams['axes.unicode_minus']=False
root=Path('artical-F122/results/proxy_architecture_20261121_v2');assets=Path('docs/assets/proxy_architecture_20261121');assets.mkdir(parents=True,exist_ok=True)
names=['fixed_Cx','fitted_Cx','affine','tanh16','relu16','silu16','tanh16x16']
labels=['原 Cx','重拟合 Cx','仿射 Ax+b','Tanh 16','ReLU 16','SiLU 16','Tanh 16×16']
rows={task:json.loads((root/task/'results.json').read_text()) for task in ['aebs','lane']}
means={task:{name:np.mean([r['metrics']['test']['mae_physical'] for r in data if r['architecture']==name],axis=0) for name in names} for task,data in rows.items()}
fig,axs=plt.subplots(1,3,figsize=(15,5))
for ax,task,col,factor,title in zip(axs,['aebs','lane','lane'],[0,0,1],[100,100,180/np.pi],['AEBS：等效距离残差（cm）','车道：等效横向残差（cm）','车道：等效航向残差（度）']):
 vals=[means[task][name][col]*factor for name in names];ax.bar(range(7),vals,color=['gray']*3+['#387f9d']*4);ax.set_xticks(range(7),labels,rotation=40,ha='right');ax.set_title(title);ax.set_ylim(0,max(vals)*1.2)
 for i,v in enumerate(vals):ax.text(i,v,'%.3f'%v,ha='center',va='bottom',fontsize=9)
fig.suptitle('固定真实视觉输出，只比较状态代理；三种子均值，零起点纳入开发集选型')
fig.tight_layout();fig.savefig(assets/'comparison.png',dpi=170);plt.close(fig)
lines=['# AEBS 与车道：小型状态代理网络结构对照','', '日期：2026-11-21；本地 main。服务器 SSH 超时，本轮用本地 audere 环境的 PyTorch 1.13.1 完成 CPU 训练。','', '**结论：AEBS 中小网络只带来很小的代理拟合改善；车道中没有得到开发集支持的非线性收益。不能据此说网络越复杂误差越小。**','', '## 1. 这轮替换的是什么','', '固定已有 `图像 → DINO → 低维视觉向量 y`，把状态代理从 `h(x)=Cx` 改为 `h(x)=Cx+小型神经网络(x)`。所有方法预测同一份 y，因此不会通过把视觉向量缩小来人为降低误差。只改变代理，不联合训练视觉投影、不改控制器。实际相机输入和在线输出保持不变。','', '本轮直接回答“哪种小网络能更贴近现有视觉输出”。联合学习视觉投影与不同 h、重训控制器、重新建立闭环证书是后续实验，不是本轮已经完成的内容。','', '## 2. 网络、输入输出与预算','', '| 结构 | AEBS 输入/输出 | 车道输入/输出 | 说明 |','|---|---|---|---|','| 原 Cx | 1→1 | 2→2 | 固定原模型 |','| 重拟合 Cx | 1→1 | 2→2 | 无偏置最小二乘 |','| 仿射 Ax+b | 1→1 | 2→2 | 带偏置最小二乘，排除神经网络仅学会常数偏置的假收益 |','| Tanh / ReLU / SiLU | 1→16→1 | 2→16→2 | 残差分支分别 49 / 82 个参数 |','| 两层 Tanh | 1→16→16→1 | 2→16→16→2 | 残差分支分别 321 / 354 个参数 |','', 'AEBS 使用距离 d/3.18272066116333；原 C=[1,0]，静态图像没有实测速度标签，所以不凭空加入速度。车道使用 (e/0.85, θ/0.30)，输出为既有线性联合模型的二维低维向量。C 使用该冻结模型的已存矩阵。','', '四种神经结构各用三个种子 2026112101–03、1500 次 Adam 更新、batch 128、学习率 0.001。每个任务内同一种子使用完全相同的 minibatch 序列；以每50步及第0步的开发误差选模型。输出层零初始化，四种网络都从原 Cx 起步。线性控制组直接最小二乘求解，计算预算不同，作为强线性基线另列。','', '结构选择以三个种子的平均开发 MSE 为准，测试集只用于报告。主损失为 `(y-h(x)) C^{-T}` 的归一化坐标 MSE。所有结构共用原 C 的逆作单位换算，不使用不同模型自己的缩放来比较误差。','', '## 3. 数据划分与限制','', '| 任务 | 训练 | 开发 | 静态校准 | 测试 |','|---|---:|---:|---:|---:|']
for task in ['aebs','lane']:
 c=json.loads((root/task/'protocol.json').read_text())['counts'];lines.append('| %s | %d | %d | %d | %d |'%(task,c['train'],c['development'],c['calibration'],c['test']))
lines+=['','AEBS 的开发集从原训练行中按固定种子划出20%；原 validation 留给校准。车道沿用已有 train/validation/calibration/test。使用历史静态数据，未宣称新的独立盲测。AEBS 图像划分存在距离区间分布差别，因此开发与测试误差量级可能不同。','', '车道历史像素审计发现训练/开发与测试有重复图像；另对已有审计确认的306条不重叠测试行复算，作为敏感性分析。此复算不用于重新选择模型。','', '## 4. 测试结果：三个种子均值','', '| 模型 | AEBS 等效距离 MAE（cm） | 车道等效横偏 MAE（cm） | 车道等效航向 MAE（度） |','|---|---:|---:|---:|']
for name,label in zip(names,labels):lines.append('| %s | %.4f | %.4f | %.4f |'%(label,means['aebs'][name][0]*100,means['lane'][name][0]*100,means['lane'][name][1]*180/np.pi))
lines += ['', 'AEBS：测试 MAE 最低为 Tanh-16，约从8.7415降至8.7087 cm，仅约0.33 mm、0.37%的改善。**按预先规定的开发集选型规则，选中的是 SiLU-16**，其测试 MAE 为8.7285 cm，改善约0.15%。不能看完测试后把 Tanh 重新当成开发选中的赢家，也没有证据说这些微小差异是稳定的显著收益。','', '车道：四种神经结构的12次训练都在开发集选择第0步，即恢复原 Cx。表中它们与 Cx 相同，不是未执行训练；所有模型都完成1500步训练，完整训练结束时的模型另存。现有视觉投影本来按线性代理联合训练，固定它再加非线性，不一定有额外收益。','', '![结构比较](../../assets/proxy_architecture_20261121/comparison.png)','', '### 训练结束时的车道结果（不回退到第0步）','', '| 结构 | 横向 MAE（cm） | 航向 MAE（度） |','|---|---:|---:|']
for name,label in zip(names[3:],labels[3:]):
 v=np.mean([r['extra']['final_iterate_metrics']['test']['mae_physical'] for r in rows['lane'] if r['architecture']==name],0);lines.append('| %s | %.4f | %.4f |'%(label,v[0]*100,v[1]*180/np.pi))
ids=json.loads(Path('artical-F122/results/lane_proxy_comparison_20261117_v1/static_nonoverlap_audit.json').read_text())['nonoverlap_test_rows'];sens={}
for name in names:
 vals=[]
 for r in rows['lane']:
  if r['architecture']!=name:continue
  key=name if r['seed'] is None else name+'_'+str(r['seed']);e=np.load(root/'lane'/(key+'.npz'))['residual_physical'][ids];vals.append(abs(e).mean(0))
 sens[name]=np.mean(vals,0).tolist()
(root/'lane_nonoverlap_diagnostic.json').write_text(json.dumps({'n':len(ids),'mae_physical':sens},indent=2))
lines+=['', '不重叠测试子集的原 Cx 横向 MAE 为 %.4f cm、航向 MAE 为 %.4f 度；开发选中的神经模型仍是第0步，所以与它完全相同。'%(sens['fixed_Cx'][0]*100,sens['fixed_Cx'][1]*180/np.pi),'', '## 5. 误差变小意味着什么','', '这里衡量的是代理与真实视觉输出之差，不是车辆任务完成误差。我们也保存了 `总观测误差 = 代理残差 + 新代理相对原 Cx 的偏移`，逐项核对成立。固定视觉网络后，总观测误差不会因为换代理就自动变小；后端需要把新的名义代理一起纳入动力学验证，不能只拿更小残差套旧证书。','', '每个模型另外保存静态校准分数的第 ceil((n+1)×0.95) 个分位数和测试包含率。这些是单帧诊断，不是轨迹级 CP 或闭环安全承诺。普通残差 MLP 也未强制可逆，不能直接替换当前依赖逆映射的部署代码。','', '## 6. 可复现文件','', '- [训练代码](../../../artical-F122/Aebs/paper_comparison/proxy_architecture_compare.py)','- [结果与模型目录](../../../artical-F122/results/proxy_architecture_20261121_v2/)','- [独立复算](../../../artical-F122/results/proxy_architecture_20261121_v2/independent_audit.json)','- [车道无重复测试诊断](../../../artical-F122/results/proxy_architecture_20261121_v2/lane_nonoverlap_diagnostic.json)','', '独立复算验证了逐模型指标、checkpoint重放、误差分解和各结构共享批次。v2增加训练末步模型诊断，与v1相同设置重跑，选中模型的所有预测逐元素一致。所有模型和划分均保存，旧实验未覆盖。','', '复现：在 artical-F122 下运行 `python -m Aebs.paper_comparison.proxy_architecture_compare --output results/新的目录`。本轮不建议直接替换已认证部署模型；车道保留 Cx，AEBS 记录小网络候选但收益很小。']
p=Path('docs/experiments/跨实验对照/AEBS与车道_小型神经代理结构对照_20261121.md');p.write_text('\n'.join(lines)+'\n')
