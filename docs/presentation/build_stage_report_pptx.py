from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

OUT = Path(__file__).with_name("阶段汇报_SPVC视觉前端替换与安全验证_20261112.pptx")
FONT = "PingFang SC"
BLACK = RGBColor(0, 0, 0); DARK = RGBColor(35, 35, 35); GRAY = RGBColor(95, 95, 95)
LIGHT = RGBColor(238, 238, 238); MID = RGBColor(180, 180, 180); WHITE = RGBColor(255, 255, 255)

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "assets"

def set_run(run, size=14, bold=False, color=BLACK):
    run.font.name = FONT; run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None: el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", FONT)

def txbox(slide, x, y, w, h, text, size=14, bold=False, color=BLACK,
          align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.03); tf.margin_top = tf.margin_bottom = Inches(0.01)
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text; set_run(r, size, bold, color)
    return box

def title(slide, text, size=26):
    txbox(slide, 0.65, 0.40, 12.0, 0.62, text, size, True, BLACK)
    line = slide.shapes.add_connector(1, Inches(0.65), Inches(1.10), Inches(12.65), Inches(1.10))
    line.line.color.rgb = MID; line.line.width = Pt(0.8)

def lead(slide, text, y=1.26, size=12.5):
    txbox(slide, 0.65, y, 12.0, 0.50, text, size, False, DARK)

def note(slide, text, y=6.82, size=10.5):
    txbox(slide, 0.65, y, 12.0, 0.36, text, size, False, GRAY)

def set_cell_border(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = tcPr.find(qn(tag))
        if ln is None: ln = etree.SubElement(tcPr, qn(tag))
        ln.set("w", "6350"); ln.set("cap", "flat"); ln.set("cmpd", "sng"); ln.set("algn", "ctr")
        for child in list(ln): ln.remove(child)
        fill = etree.SubElement(ln, qn("a:solidFill"))
        clr = etree.SubElement(fill, qn("a:srgbClr")); clr.set("val", "B0B0B0")

def table(slide, headers, rows, y=1.78, widths=None, font=10.5, header_font=10.5,
          row_h=0.52, header_h=0.40, first_bold=True, total_w=12.0, x=0.65):
    nrows, ncols = len(rows) + 1, len(headers)
    widths = widths or [1] * ncols
    sw = sum(widths); widths = [w * total_w / sw for w in widths]
    shape = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(total_w),
                                   Inches(header_h + row_h * len(rows)))
    tbl = shape.table; tbl.first_row = False; tbl.horz_banding = False
    for j, w in enumerate(widths): tbl.columns[j].width = Inches(w)
    tbl.rows[0].height = Inches(header_h)
    for i in range(1, nrows): tbl.rows[i].height = Inches(row_h)
    for j, h in enumerate(headers):
        c = tbl.cell(0, j); c.fill.solid(); c.fill.fore_color.rgb = LIGHT
        c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.02)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE; c.text_frame.word_wrap = True
        r = c.text_frame.paragraphs[0].add_run(); r.text = h; set_run(r, header_font, True, BLACK); set_cell_border(c)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = tbl.cell(i, j); c.fill.solid(); c.fill.fore_color.rgb = WHITE
            c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE; c.text_frame.word_wrap = True
            r = c.text_frame.paragraphs[0].add_run(); r.text = str(val)
            set_run(r, font, bool(first_bold and j == 0), BLACK); set_cell_border(c)
    return shape


def figure(slide, path, x, y, w, h, caption, cap_y=None):
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    txbox(slide, x, (cap_y if cap_y is not None else y + h + 0.04), w, 0.32, caption, 8.5, False, GRAY)

def bullet_list(slide, x, y, w, h, items, size=13, gap=8):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    first = True
    for item in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        pPr = p._p.get_or_add_pPr(); pPr.set("marL", str(int(Inches(0.26)))); pPr.set("indent", str(int(-Inches(0.26))))
        p.space_after = Pt(gap)
        r = p.add_run(); r.text = "•  " + item; set_run(r, size, False, DARK)

# 1
s = prs.slides.add_slide(BLANK)
txbox(s, 0.9, 1.35, 11.5, 1.0, "SPVC 视觉前端替换与安全验证", 38, True, BLACK)
txbox(s, 0.9, 2.35, 11.5, 0.6, "阶段汇报", 23, False, DARK)
line = s.shapes.add_connector(1, Inches(0.9), Inches(3.10), Inches(12.4), Inches(3.10)); line.line.color.rgb = MID; line.line.width = Pt(0.8)
txbox(s, 0.9, 3.50, 11.4, 0.8, "换前端，但保留“代理误差建模 + 神经安全证书”的主线。", 21, True, BLACK)
txbox(s, 0.9, 4.20, 11.4, 0.8, "已完成：AEBS 与二维车道的条件数值证书；", 17, False, DARK)
txbox(s, 0.9, 4.80, 11.4, 0.8, "联合 C 没有稳定独立收益；前端排序依赖协议。", 17, False, DARK)
txbox(s, 0.9, 6.20, 11.4, 0.5, "不把“具体系统结果”写成“架构全面胜出”。", 12.5, False, GRAY)

# 2
s = prs.slides.add_slide(BLANK)
title(s, "目标与三类工作")
bullet_list(s, 0.85, 1.55, 11.7, 4.8, [
    "框架适配：冻结 DINO + 低维投影；实际观察与状态代理的误差进入 CP 和神经 SBC。",
    "场景验证：灰度 AEBS、RGB AEBS、二维车道获得条件数值证书；停车场用后备机制获得受保护闭环证书。",
    "组件与收益：完成前端静态对照、400/10k/24k 总包量实验、车道固定/联合 C 五对种子和共同半径扫描。",
    "结论：联合 C 可训练，但没有稳定独立优势；现有结果不支持“DINO 全面更好”。",
], size=17, gap=18)
note(s, "汇报按“整体方法 → 两个主实验 → 组件对照 → 扩展场景 → 保证范围”展开。")

# 3
s = prs.slides.add_slide(BLANK)
title(s, "两条观察分支：原 cGAN 与当前 DINO")
txbox(s, 0.75, 1.45, 5.7, 0.4, "原 SPVC", 18, True, BLACK)
txbox(s, 0.85, 1.95, 5.6, 2.2,
      "状态 d + 潜变量 z\n→ 生成器 G\n→ 32×32 生成图\n→ state_net\n→ PPO 制动", 14, False, DARK)
txbox(s, 6.85, 1.45, 5.7, 0.4, "当前方案", 18, True, BLACK)
txbox(s, 6.95, 1.95, 5.7, 2.2,
      "真实图像 I\n→ 冻结 DINO\n→ 投影 pθ\n→ 低维观察 y\n→ 与状态代理 Cx 比较", 14, False, DARK)
line = s.shapes.add_connector(1, Inches(6.65), Inches(1.45), Inches(6.65), Inches(4.18)); line.line.color.rgb = MID; line.line.width = Pt(0.8)
txbox(s, 0.85, 4.55, 11.6, 0.4, "为什么要显式处理代理误差", 16, True, BLACK)
txbox(s, 0.85, 5.05, 11.6, 1.2,
      "生成图上的预测好，不等于真实图像上的观察误差小。当前把实际观察 y 与名义代理 Cx 的残差 ε 接入轨迹 CP 和安全验证。", 14, False, DARK)
note(s, "真实位置只用于离线训练、仿真评价、校准和证明；控制运行时仍从图像得到 y。")

# 4
s = prs.slides.add_slide(BLANK)
title(s, "完整流程：从任务建模到留出评价")
table(s, ["步骤", "做什么", "输出"], [
    ["任务建模", "明确动力学、初始范围、时长、危险条件、额外已知观测", "统一任务定义"],
    ["视觉训练", "图像 + 状态标签 + 冻结 DINO；学习投影，部分版本学习 C", "前端权重与观察接口"],
    ["控制适配", "低维观察 + 速度/曲率输入 PPO；联合 C 用解码接口", "实际动作"],
    ["误差采集", "冻结视觉和控制器；按各自闭环生成完整轨迹", "轨迹与逐步误差"],
    ["轨迹 CP", "每条轨迹取最大误差；独立校准轨迹定分位", "q 与误差集合"],
    ["SBC 训练/检查", "学习非负 B；检查初始、危险和后继不增", "证书或失败见证"],
    ["留出评价", "未参与校准的新运行；统计覆盖、危险、成功、超时", "经验表现与范围"],
], y=1.62, widths=[1.8, 7.2, 3.0], font=11.2, header_font=11.2, row_h=0.62)
note(s, "每一步的输入/输出固定；不能把不同实验的 q、覆盖和行为拼成一次完整训练。")

# 5
s = prs.slides.add_slide(BLANK)
title(s, "CP 与 SBC 回答两个不同问题")
txbox(s, 0.85, 1.55, 5.7, 0.4, "CP：误差有多大", 19, True, BLACK)
txbox(s, 0.85, 2.05, 5.7, 2.7,
      "每条完整轨迹取最大误差，得到一个分数。\n"
      "199 条校准轨迹 → 199 个分数。\n"
      "95% 目标 → 第 190 小分数作为 q。\n"
      "留出轨迹只检查覆盖，不参与定 q。", 14, False, DARK)
txbox(s, 6.85, 1.55, 5.7, 0.4, "SBC：包络内是否安全", 19, True, BLACK)
txbox(s, 6.95, 2.05, 5.7, 2.7,
      "B 是非负安全分数，不是距离或碰撞概率。\n"
      "初始 B ≤ 1；危险处 B ≥ 20。\n"
      "误差在 CP 集合内时，所有允许后继 B 不增。\n"
      "连续域切盒检查：无反例、无未解决。", 14, False, DARK)
note(s, "整轨迹覆盖、任务成功、SBC 通过是三个不同指标；不能互相替代。")

# 6
s = prs.slides.add_slide(BLANK)
title(s, "实验地图：从一维 AEBS 到三维机器人")
table(s, ["实验", "物理状态", "视觉观察", "当前状态"], [
    ["灰度 AEBS", "距离、速度", "一维距离", "历史合同与固定库回放通过"],
    ["RGB AEBS 固定 C", "距离、速度", "RGB→一维距离", "轨迹 CP 与条件 SBC 通过"],
    ["RGB/灰度 AEBS 联合 C", "距离、速度", "一维可训练观察", "RGB 通过；灰度未通过"],
    ["车道固定/联合 C", "横向偏差、航向", "二维观察", "固定与联合均通过；五对消融完成"],
    ["停车场", "位置、朝向、速度", "三维视觉 + 延迟观测器", "受保护闭环证书通过"],
    ["移动机器人", "位置、朝向", "三维视觉位姿", "行为跑通；完整鲁棒证书未通过"],
], y=1.65, widths=[3.0, 3.0, 3.2, 4.3], font=10.7, header_font=10.7, row_h=0.72)
note(s, "“数值通过”均指所声明模型、状态域和误差集合下的连续域数值审计；不是实车保证。")

# 7
s = prs.slides.add_slide(BLANK)
title(s, "AEBS：距离/速度、制动、15–16 m 初态")
table(s, ["配置", "设置"], [
    ["状态", "d（m）、v（m/s）"],
    ["初态", "d∈[15,16]，v∈[2.5,3]"],
    ["动作", "a∈[−3,3]；正值制动，负值加速"],
    ["动力学", "d_next=d−0.05v；v_next=clip(v−0.05a,0,3)"],
    ["危险区域", "d∈[5,6] 且 v∈[0.5,3]"],
    ["视觉量", "归一化距离；速度实测，不由图像估计"],
    ["主版 C", "固定 [1,0]"],
], y=1.65, widths=[2.2, 9.6], font=15, header_font=14, row_h=0.62)
note(s, "“原始 RGB”是 CARLA 保存的 640×640 渲染图，不是实车摄影；DINO 实际输入为 224×224。")

# 8
s = prs.slides.add_slide(BLANK)
title(s, "灰度 AEBS：400 图小包先跑通证书链")
lead(s, "10 月 29 日已有基础；用来验证“替换前端后还能接回 SPVC 证书路线”。")
table(s, ["项目", "结果", "怎么读"], [
    ["数据", "400 张 32×32 灰度图；240/80/80", "训练/验证/测试图像数"],
    ["静态 test MAE", "0.129266 m", "平均单帧误差"],
    ["RMSE / 最大误差", "0.165444 / 0.414889 m", "单帧误差的均方根与最大值"],
    ["状态相关误差集合", "4,971,398 区间；0 见证、0 未解决", "连续域检查的区间数"],
    ["固定 400 图回放", "105,608 区间；0/0", "回放检查"],
    ["名义/全库/留出任务", "各 1441/1441 成功", "行为回放，不是新概率证书"],
], y=1.72, widths=[2.5, 3.9, 2.6], font=9.2, header_font=9.2, row_h=0.62, total_w=7.45, x=0.65)
figure(s, ASSETS / "aebs_input_examples.png", 8.35, 1.86, 4.25, 3.36, "灰度 AEBS 观测样例：32×32 灰度图（含距离标签）", 5.32)
note(s, "这版说明小规模灰度前端能接回证书路线；图片、合同和策略与 24k RGB 不同，MAE 不能直接横比。")

# 9
s = prs.slides.add_slide(BLANK)
title(s, "RGB AEBS：24k 保存图像，DINO 224×224")
lead(s, "数据包含 4 种天气、3 种颜色、5–16 m 距离；按距离 bin 分组。")
table(s, ["项目", "设置/结果", "说明"], [
    ["总图像数", "24,000", "训练、验证、测试图像总和"],
    ["划分", "14,172 / 4,908 / 4,920", "train / calibration-validation / test"],
    ["视觉链", "RGB→DINO CLS 384→128→128→1", "输出归一化距离"],
    ["静态 test MAE", "0.087415 m", "平均单帧误差"],
    ["RMSE / 最大误差", "0.121103 / 0.525 m", "单帧误差的均方根与最大值"],
    ["新 PPO 支线", "约 100k 步；未成为发布版", "独立子库回放暴露危险行为"],
], y=1.70, widths=[2.4, 4.1, 2.8], font=9.2, header_font=9.2, row_h=0.62, total_w=7.45, x=0.65)
figure(s, ASSETS / "aebs_real_rgb_examples.png", 8.35, 1.86, 4.25, 3.11, "RGB AEBS 观测样例：距离/天气/颜色组合", 5.08)
note(s, "当前发布版沿用已有语义 actor；重点是前端适配、独立轨迹校准和 SBC 重训。")

# 10
s = prs.slides.add_slide(BLANK)
title(s, "RGB 主证书：轨迹 q=0.4184 m，SBC 0/0")
lead(s, "冻结视觉和 actor；199 条校准轨迹、最长 400 步；100 条留出；12 组外观回放。")
table(s, ["数值", "含义", "怎么得到", "说明什么"], [
    ["0.4183585684 m", "轨迹 q", "199 条校准轨迹各取全程最大物理误差；第 190 小分数", "新轨迹全程最大误差约 95% 不超过它"],
    ["190/199", "校准覆盖", "199 条校准中 190 条在 q 内", "校准分位数产生"],
    ["99/100", "留出覆盖", "100 条新轨迹中 99 条在 q 内", "不是 99% 理论安全率"],
    ["292,024", "SBC 区间数", "连续状态盒检查", "0 实际见证、0 未解决"],
    ["4800/4800", "行为回放", "12 组外观网格回放", "任务行为成功，不等于证书"],
], y=1.72, widths=[2.4, 2.2, 5.0, 3.8], font=10.8, header_font=10.8, row_h=0.64)
note(s, "初始 B 上界 0、危险下界 25.213636；图像来自保存图片库，不是每步在线 CARLA 新图。")

# 11
s = prs.slides.add_slide(BLANK)
title(s, "单位错误：旧的“通过”为什么要撤回")
lead(s, "此前把归一化分位数直接当作米，实际验证范围比宣称范围小。")
table(s, ["数值", "含义", "怎么得到", "说明什么"], [
    ["0.07249", "旧归一化 q", "旧合同把归一化半径直接当米", "实际只验证到约 ±0.07249 m"],
    ["0.230725 m", "修正后的静态半径", "把归一化量换回米", "旧候选在正确半径下不再安全"],
    ["212", "旧 SBC 正漂移见证", "在正确静态半径下检查", "旧 tube 通过结论撤回"],
    ["0.4183585684 m", "新轨迹 q", "修正后重新采集 199 条校准轨迹", "新 SBC 在 292,024 区间 0/0"],
], y=1.80, widths=[2.4, 2.5, 4.8, 3.5], font=11.5, header_font=11.5, row_h=0.75)
note(s, "单位、分数和验证器必须一致；不能把静态半径直接当轨迹半径，也不能把旧结论沿用。")

# 12
s = prs.slides.add_slide(BLANK)
title(s, "状态相关 CP：更紧的半径，但覆盖更低")
lead(s, "在训练残差上拟合随距离变化的正尺度 b(d)，用 r(d)=q·b(d)；C 仍固定 [1,0]。")
table(s, ["指标", "常量主版", "状态相关版", "怎么读"], [
    ["半径", "0.418359 m", "平均约 0.3217 m；最大约 0.720 m", "状态相关版不是所有状态都更小"],
    ["留出全程覆盖", "99/100", "92/100", "总覆盖下降"],
    ["SBC 检查", "292,024 区间", "41,070 区间", "区间数减少不等于时间等比例减少"],
], y=1.85, widths=[2.2, 3.1, 4.4, 3.0], font=12.5, header_font=12.5, row_h=0.85)
note(s, "状态相关版是对照结果，不替换常量主版；这批数据已用于开发/回归，不能称为全新盲测。")

# 13
s = prs.slides.add_slide(BLANK)
title(s, "车道：二维状态、80 步、三种曲率")
table(s, ["配置", "设置"], [
    ["物理状态", "e（m）、θ（rad）"],
    ["归一化状态", "x=(e/0.85, θ/0.30)"],
    ["初态", "e∈[−0.25,0.25]；θ∈[−0.06,0.06]"],
    ["安全边界", "e∈[−0.85,0.85] m；θ∈[−0.30,0.30] rad"],
    ["速度/步长/轴距", "4 m/s；0.1 s；2.6 m"],
    ["曲率", "−0.01、0、0.01；每条固定"],
    ["观察时域", "80 步，8 s"],
], y=1.65, widths=[2.2, 9.6], font=14.5, header_font=13.5, row_h=0.60)
note(s, "车道与 AEBS 使用同类 PPO/神经 SBC 方法，但动力学、动作含义和安全集不同。")

# 14
s = prs.slides.add_slide(BLANK)
title(s, "车道前端与 SBC：patch token + 非负网络")
lead(s, "空间位置依赖强，不能只用全局 CLS；保留 16×16 patch token，经卷积和空间读出得到二维观察。")
table(s, ["项目", "设置/结果", "说明"], [
    ["视觉数据", "3600 张 192×192", "2160 train / 540 val / 540 静态校准 / 360 test"],
    ["测试误差", "横向 MAE≈0.0211 m；航向≈0.0059 rad", "固定 C 原版"],
    ["SBC 结构", "2→16→8→1；Tanh 隐层、ReLU 输出", "与 AEBS 同类 NonnegativeSBC"],
    ["训练", "两个种子约 2000 / 1750 次更新", "两个候选均通过对应主审计"],
    ["行为回归", "720 条误差轨迹 + 12 条图像轨迹", "无越界；12 条中 10 条在旧包络内"],
], y=1.78, widths=[2.3, 4.2, 2.9], font=9.6, header_font=9.6, row_h=0.70, total_w=7.45, x=0.65)
figure(s, ASSETS / "lane_input_examples.png", 8.30, 1.86, 4.35, 3.23, "车道观测样例：横向偏差/航向/曲率组合", 5.20)
note(s, "旧 12 条中有两条各一帧超包络；它们仍安全，但不能把全程都算作旧包络保证。")

# 15
s = prs.slides.add_slide(BLANK)
title(s, "车道轨迹 CP：q=0.090093，95/100")
lead(s, "冻结视觉与策略，独立生成 199 条校准、100 条留出；每步按本闭环状态重新渲染。")
table(s, ["数值", "含义", "怎么得到", "说明什么"], [
    ["0.0900931580", "轨迹 q", "每条 80 步取最大归一化误差，再取 95% 分位", "归一化 latent 坐标，不是米"],
    ["199 / 100", "校准/留出", "独立轨迹条数", "校准和留出分开"],
    ["95/100", "新半径全程覆盖", "100 条新轨迹中 95 条全程在 q 内", "单帧覆盖高不等于全程覆盖"],
    ["82/100", "旧 0.08 的同批覆盖", "同一批轨迹用旧半径评估", "旧半径漏出更明显"],
    ["0/100", "留出实际越界", "新 q 下检查", "0 条越界"],
], y=1.78, widths=[2.2, 2.6, 4.8, 3.6], font=10.8, header_font=10.8, row_h=0.64)
note(s, "新 q 下，三个曲率的 SBC 仿射界与普通 IBP 均完成数值检查。")

# 16
s = prs.slides.add_slide(BLANK)
title(s, "车道状态相关 v2：修复了，但不优于常量")
lead(s, "v1 的报告/代码不一致已撤回；v2 用前 99 条拟合尺度、后 100 条校准，不截断半径。")
table(s, ["指标", "修复 v2", "怎么读"], [
    ["平均 / 观测最大半径", "0.088393 / 0.109369", "实际访问状态上的尺度"],
    ["全域解析最大半径", "0.235815", "不能用观测最大值替代"],
    ["回归全程覆盖", "93/100", "同校准常量盒为 94/100"],
    ["实际集合审计", "13,000 区间；三个曲率通过", "只针对实际状态相关集合"],
], y=1.90, widths=[3.0, 4.8, 4.0], font=13, header_font=13, row_h=0.85)
note(s, "把全域最大半径施加到所有状态会得到更大集合，该扩大盒验证失败；当前证据不显示 v2 明显优于常量。")

# 17
s = prs.slides.add_slide(BLANK)
title(s, "联合 C：接口接通，单对 q=0.084338")
lead(s, "联合版用可训练二维矩阵 C；原控制器通过 π_new(y,κ)=π_old(C⁻¹y,κ) 接回同一语义。")
table(s, ["项目", "结果", "说明"], [
    ["训练", "30 epoch、batch 64、lr 1e-4", "按开发验证误差选检查点"],
    ["单对历史", "q=0.0843382；留出 93/100", "C 从初始旋转轻微变化"],
    ["条件数", "约 1.00158", "接近可逆坐标变换"],
    ["接口", "误差 δ=C⁻¹(y−Cx)", "不是对 y 每维加同一个 q"],
    ["结论", "联合训练、误差和验证接口接通", "不说明联合 C 比固定 C 更好"],
], y=1.78, widths=[2.4, 5.4, 4.2], font=11.5, header_font=11.5, row_h=0.70)
note(s, "若同时令 y′=Ty、C′=TC，则 (C′)⁻¹y′=C⁻¹y；不能仅因 C 非单位矩阵就说可观测性提高。")

# 18
s = prs.slides.add_slide(BLANK)
title(s, "五对种子：可训练，但没有稳定收益")
lead(s, "同一图像、视觉头、warm start、优化配置、原语义 actor 和同一个 B；每对各 199/100。")
table(s, ["种子", "固定 C q", "联合 C q", "固定覆盖", "联合覆盖"], [
    ["20261109", "0.088690", "0.086439", "96/100", "95/100"],
    ["20261111", "0.087447", "0.085980", "96/100", "97/100"],
    ["20261112", "0.082158", "0.086194", "94/100", "94/100"],
    ["20261113", "0.085749", "0.084497", "98/100", "97/100"],
    ["20261114", "0.082272", "0.085384", "94/100", "92/100"],
], y=1.70, widths=[2.4, 2.6, 2.6, 2.2, 2.2], font=12, header_font=11.1, row_h=0.52)
txbox(s, 0.85, 5.25, 11.8, 0.9,
      "汇总：q 均值 0.085263 vs 0.085699；覆盖均值 95.6% vs 95.0%；越界 0/500；SBC 5/5 vs 5/5；3/5 对联合更小。",
      11.5, False, DARK)
note(s, "联合训练可行，但当前没有稳定的独立误差、覆盖或证书获得率收益。", 6.95)

# 19
s = prs.slides.add_slide(BLANK)
title(s, "AEBS 联合 C：RGB 通过，灰度失败")
lead(s, "图像只有距离标签；一维 C 几乎只能取正负号，因此用 x=(d/S,v) 学 C=(C_d,C_v)。")
table(s, ["版本", "学到的 C", "新轨迹 q", "留出覆盖", "对应 SBC"], [
    ["RGB 联合版", "[1, −8.44×10⁻⁶]", "0.394521 m", "91/100", "同一 RGB 新 B 数值通过"],
    ["灰度联合 v2", "[0.999998, 0.001931]", "3.515916 m", "95/100", "旧灰度候选未通过"],
], y=1.95, widths=[2.3, 3.3, 2.5, 1.8, 3.5], font=11.5, header_font=10.8, row_h=0.90)
txbox(s, 0.85, 4.62, 11.8, 1.3,
      "RGB 联合版静态 test MAE≈0.0846 m，固定 C 为 0.0874 m；两版不是完整配对重复，不能据此宣布改善。灰度版少量严重误差被轨迹访问，不能把“q 大/覆盖高”当成功。",
      12, False, DARK)
note(s, "灰度候选记录了 78 个实际 B 正漂移见证；这不等于已经展示初始出发的碰撞。")

# 20
s = prs.slides.add_slide(BLANK)
title(s, "前端对照：具体检查点，不是纯架构")
lead(s, "左表在同一 4,920 张真实 test 上；右表是原生生成闭环与 RGB 回放两个具体系统。")
table(s, ["前端", "MAE (m)", "半径 (m)", "覆盖"], [
    ["旧 cGAN state_net 读真实灰度", "2.747", "5.221", "87.64%"],
    ["灰度 DINO 历史", "0.1017", "0.2656", "90.47%"],
    ["灰度 DINO 重训", "0.1079", "0.2797", "90.75%"],
    ["RGB DINO", "0.0874", "0.2307", "93.33%"],
], y=1.72, widths=[5.2, 2.2, 2.2, 2.2], font=11.8, header_font=11.2, row_h=0.55)
txbox(s, 0.85, 4.35, 11.8, 1.2,
      "原生 cGAN 生成闭环：轨迹 q=1.915645 m、留出 92/100、成功 100/100。\n"
      "RGB DINO 回放：轨迹 q=0.418359 m、留出 99/100、成功 100/100。",
      12.5, False, DARK)
note(s, "数据量、图像来源、策略和训练资源不同；只能作为系统观察，不是严格前端替换因果。")

# 21
s = prs.slides.add_slide(BLANK)
title(s, "数据预算：排序随协议改变")
lead(s, "总包量指全部图像；同档共用 test，三档 test 不同；cGAN 用 32×32 灰度 realhead，DINO 用 RGB。")
table(s, ["总包", "像素 realhead MAE / q / 覆盖", "DINO MAE / q / 覆盖"], [
    ["400", "0.6937 / 4.1690 / 71/72", "0.2978 / 0.8649 / 69/72"],
    ["10k", "0.0085 / 0.0209 / 1894/2000", "0.0205 / 0.0557 / 1893/2000"],
    ["24k", "0.0247 / 0.0307 / 4382/4920", "0.0874 / 0.2307 / 4592/4920"],
], y=1.85, widths=[1.6, 5.2, 5.2], font=11.2, header_font=10.4, row_h=0.78)
txbox(s, 0.85, 4.82, 11.8, 1.1,
      "400 图 DINO 更好；10k/24k realhead 的 MAE/q 更小，但 24k realhead 覆盖更低。"
      "排序依赖数据、输入处理、训练和评价协议；没有建立单向架构优势。", 12.3, False, DARK)
note(s, "24k DINO 复用已有检查点；像素 realhead 100 epoch、DINO 小包 250；未完成等训练计算预算对照。")

# 22
s = prs.slides.add_slide(BLANK)
title(s, "共同半径：0.12 通过，0.16 失败")
lead(s, "固定同一个 actor、B、动力学、初始域和验证器，只改变允许误差半径。")
table(s, ["测试半径", "条件数值审计", "说明"], [
    ["0.04、0.06、0.08", "三个曲率均通过", "低半径"],
    ["0.0843382、0.0900932", "三个曲率均通过", "固定/联合 q 附近"],
    ["0.10、0.12", "三个曲率均通过", "已测试通过的最大点"],
    ["0.16", "出现正漂移见证，未通过", "不能宣称通过"],
], y=1.80, widths=[3.5, 5.0, 3.5], font=13, header_font=12.2, row_h=0.72)
txbox(s, 0.85, 4.80, 11.8, 1.0,
      "r_cert=0.12。与固定 q=0.090093 的比值约 1.33，与联合 q=0.084338 的比值约 1.42；"
      "0.14/0.15 未测，0.12 不是理论上限。", 12.5, False, DARK)
note(s, "固定/联合 C 共享同一语义误差盒，因此共享扫描；这不构成 C 的独立收益。")

# 23
s = prs.slides.add_slide(BLANK)
title(s, "停车场：后备改变控制结构，换来证书")
lead(s, "状态 (p_x,p_y,θ,v)，30 步、RK4、零过程扰动；旧无后备控制器会出现速度越界。")
table(s, ["项目", "结果/做法", "说明"], [
    ["启动校准", "975 次启动采集；拟合/校准/审计分开", "不是用审计结果调半径"],
    ["启动审计覆盖", "233/240", "审计观察值，不是理论置信度"],
    ["图像运行", "原中心/新扰动各 15/15 成功", "零碰撞、零越界"],
    ["SBC 条件", "初始≤0.305；危险≥20；每步变化≤−0.005", "几何 ReLU 分支 + 有界学习头 + 时钟"],
    ["保证", "受保护增广闭环", "不是原直接 PPO 的全域证书"],
], y=1.72, widths=[2.3, 4.3, 2.9], font=9.2, header_font=9.2, row_h=0.66, total_w=7.35, x=0.65)
figure(s, ASSETS / "parking_observation_examples.png", 8.45, 1.72, 3.75, 4.10, "停车场俯视视觉观测样例（历史渲染，非 11/03 新采）", 5.90)
note(s, "学习头在图像运行中作用很小；不能把结果归因于自由神经训练独立解决了任务。")

# 24
s = prs.slides.add_slide(BLANK)
title(s, "机器人：行为成功，盒内仍有 28/32 反例")
lead(s, "三维视觉位姿，6×4 m 地图，绕过中间圆障碍；固定俯视 224×224，不是 SLAM/实物。")
table(s, ["检查", "结果", "说明"], [
    ["视觉 MAE", "x≈2.55 cm；y≈1.86 cm；θ≈0.0783 rad", "位置与朝向"],
    ["开发图像闭环", "24/24 安全到达并停稳", "行为证据"],
    ["轨迹 CP", "39 条校准、120 步；半宽约 (0.0488,0.0488,0.3254)", "归一化误差盒"],
    ["独立留出", "18/20 全程覆盖；20/20 安全到达", "覆盖与行为分开"],
    ["后继不增", "未通过；28/32 不安全", "21 条机器人/障碍物圆重叠"],
], y=1.72, widths=[2.3, 4.4, 2.9], font=9.4, header_font=9.4, row_h=0.64, total_w=7.35, x=0.65)
figure(s, ROOT / "artical-F122/results/mobile_robot_20261108_perception_r1/examples.png", 8.40, 1.86, 4.35, 2.18, "机器人俯视视觉观测样例：8 个初始场景", 4.15)
note(s, "问题在控制器对整个误差盒不鲁棒；实际视觉误差有结构，盒却允许最不利的时序/坐标组合。")

# 25
s = prs.slides.add_slide(BLANK)
title(s, "保证范围与下一步")
table(s, ["实验", "当前可汇报的保证范围"], [
    ["RGB AEBS 主版", "冻结图片库回放、注册初态与停止规则；轨迹 CP + 集合内条件数值安全"],
    ["车道主版/联合版", "注册道路渲染、4 m/s、三种固定曲率、80 步；轨迹 CP + 集合内条件安全"],
    ["早期灰度", "原状态相关合同与固定图片库规则；不外推新图像分布"],
    ["停车场", "注册初态盒、模型、时域和后备机制下的受保护闭环"],
    ["机器人", "没有完整鲁棒证书；只能报告行为、覆盖与失败证据"],
], y=1.65, widths=[2.3, 9.5], font=10.2, header_font=10.5, row_h=0.58)
txbox(s, 0.85, 5.25, 11.8, 0.35, "下一步优先", 14, True, BLACK)
txbox(s, 0.85, 5.65, 11.8, 1.0,
      "① 统一评估分布、模型选择/校准分工和计算预算，比较原生成链路与当前联合 C 的代理残差；\n"
      "② 在车道半径余量上研究更有意义的可认证范围（如初始偏差）；③ 机器人先解决允许误差下的控制失效。",
      10.8, False, DARK)
note(s, "所有数值验证以区间/仿射界有效为前提；不是实车、新相机或无限时域保证。")

prs.save(str(OUT))
print(OUT)
