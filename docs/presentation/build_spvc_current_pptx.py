from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

OUT = Path(__file__).with_name("SPVC当前方案_实验与数值解释_20261112.pptx")
FONT = "PingFang SC"
BLACK = RGBColor(0, 0, 0)
DARK = RGBColor(35, 35, 35)
GRAY = RGBColor(95, 95, 95)
LIGHT = RGBColor(238, 238, 238)
MID = RGBColor(180, 180, 180)
WHITE = RGBColor(255, 255, 255)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

def set_run(run, size=14, bold=False, color=BLACK):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", FONT)

def txbox(slide, x, y, w, h, text, size=14, bold=False, color=BLACK,
          align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.03)
    tf.margin_top = tf.margin_bottom = Inches(0.01)
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run(); r.text = text; set_run(r, size, bold, color)
    return box

def title(slide, text, size=27):
    txbox(slide, 0.65, 0.40, 12.0, 0.62, text, size, True, BLACK)
    line = slide.shapes.add_connector(1, Inches(0.65), Inches(1.10), Inches(12.65), Inches(1.10))
    line.line.color.rgb = MID; line.line.width = Pt(0.8)

def lead(slide, text, y=1.26, size=13):
    txbox(slide, 0.65, y, 12.0, 0.50, text, size, False, DARK)

def note(slide, text, y=6.82, size=11):
    txbox(slide, 0.65, y, 12.0, 0.36, text, size, False, GRAY)

def set_cell_border(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = tcPr.find(qn(tag))
        if ln is None:
            ln = etree.SubElement(tcPr, qn(tag))
        ln.set("w", "6350")
        ln.set("cap", "flat"); ln.set("cmpd", "sng"); ln.set("algn", "ctr")
        for child in list(ln):
            ln.remove(child)
        fill = etree.SubElement(ln, qn("a:solidFill"))
        clr = etree.SubElement(fill, qn("a:srgbClr")); clr.set("val", "B0B0B0")

def table(slide, headers, rows, y=1.78, widths=None, font=11.5, header_font=11.5,
          row_h=0.52, header_h=0.42, first_bold=True, total_w=12.0, x=0.65):
    nrows, ncols = len(rows) + 1, len(headers)
    widths = widths or [1] * ncols
    sw = sum(widths); widths = [w * total_w / sw for w in widths]
    shape = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y),
                                   Inches(total_w), Inches(header_h + row_h * len(rows)))
    tbl = shape.table
    tbl.first_row = False; tbl.horz_banding = False
    for j, w in enumerate(widths):
        tbl.columns[j].width = Inches(w)
    tbl.rows[0].height = Inches(header_h)
    for i in range(1, nrows):
        tbl.rows[i].height = Inches(row_h)
    for j, h in enumerate(headers):
        c = tbl.cell(0, j)
        c.fill.solid(); c.fill.fore_color.rgb = LIGHT
        c.margin_left = c.margin_right = Inches(0.06)
        c.margin_top = c.margin_bottom = Inches(0.02)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        c.text_frame.word_wrap = True
        r = c.text_frame.paragraphs[0].add_run(); r.text = h
        set_run(r, header_font, True, BLACK); set_cell_border(c)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.fill.solid(); c.fill.fore_color.rgb = WHITE
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.text_frame.word_wrap = True
            r = c.text_frame.paragraphs[0].add_run(); r.text = str(val)
            set_run(r, font, bool(first_bold and j == 0), BLACK)
            set_cell_border(c)
    return shape

def bullet_list(slide, x, y, w, h, items, size=15, gap=9):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    first = True
    for item in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(Inches(0.28))))
        pPr.set("indent", str(int(-Inches(0.28))))
        p.space_after = Pt(gap)
        r = p.add_run(); r.text = "•  " + item
        set_run(r, size, False, DARK)

def formula_box(slide, x, y, w, h, lines):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.08)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        r = p.add_run(); r.text = line; set_run(r, 18, False, BLACK)

# 1 cover
s = prs.slides.add_slide(BLANK)
txbox(s, 0.9, 1.45, 11.5, 1.0, "SPVC 前端替换实验", 40, True, BLACK)
txbox(s, 0.9, 2.45, 11.5, 0.6, "从 cGAN 到 DINO：实验设置、证书与结果", 22, False, DARK)
line = s.shapes.add_connector(1, Inches(0.9), Inches(3.25), Inches(12.4), Inches(3.25))
line.line.color.rgb = MID; line.line.width = Pt(0.8)
txbox(s, 0.9, 3.65, 11.4, 0.8, "先讲实验，再讲数字。", 22, True, BLACK)
txbox(s, 0.9, 4.35, 11.4, 0.8, "真实图 DINO 部署系统的轨迹误差包络更小；", 18, False, DARK)
txbox(s, 0.9, 4.95, 11.4, 0.8, "同数据预算排序交叉；联合 C 没有独立收益。", 18, False, DARK)
txbox(s, 0.9, 6.25, 11.4, 0.5, "不把“部署收益”写成“架构全面胜出”。", 13, False, GRAY)

# 2 overview
s = prs.slides.add_slide(BLANK)
title(s, "四组实验，分别回答不同问题")
table(s, ["实验组", "实验问题"], [
    ["AEBS 主实验", "换前端后，观测误差包络是否更小？证书是否仍成立？行为是否保持？"],
    ["原 cGAN 对照", "相对原 SPVC 的部署系统，新前端误差/覆盖如何？"],
    ["数据预算", "数据从少到多，cGAN 与 DINO 的排序会不会变？"],
    ["联合 C / 车道 / 停车与机器人", "组件是否有独立收益？跨任务是否仍成立？"],
], y=1.75, widths=[3.4, 8.6], font=14, header_font=14, row_h=0.78)
note(s, "先看每个实验怎么设计，再看它的数字；不把不同实验的数字混在一起。")

# 3 AEBS task/data
s = prs.slides.add_slide(BLANK)
title(s, "AEBS：从图像估计距离，再交给制动控制器")
txbox(s, 0.75, 1.45, 5.5, 0.45, "实验对象", 19, True, BLACK)
txbox(s, 0.85, 2.00, 5.2, 3.2,
      "任务：车辆跟随前车，控制器输出制动\n\n"
      "状态：真实距离与速度\n\n"
      "目标：从图像估计距离，供控制器使用", 14, False, DARK)
txbox(s, 7.05, 1.45, 5.5, 0.45, "数据与标签", 19, True, BLACK)
txbox(s, 7.15, 2.00, 5.2, 3.2,
      "数据：保存的 CARLA RGB 图像\n\n"
      "标签：每张图对应一个真实距离\n\n"
      "总图像数：24,000 张；这是训练、验证、测试图像的总和，不是轨迹数", 14, False, DARK)
txbox(s, 0.85, 5.10, 11.6, 0.35, "为什么强调真实图", 16, True, BLACK)
txbox(s, 0.85, 5.55, 11.6, 0.85,
      "原 cGAN 是“状态→生成图”；本实验直接使用真实采集图，评估真实图像分布上的观测误差，而不是只评估生成图自洽。", 13, False, DARK)
note(s, "这是保存图像上的闭环回放实验，不是实车，也不是在线 CARLA。")

# 4 procedure
s = prs.slides.add_slide(BLANK)
title(s, "先把前端训好，再用独立轨迹做 CP、SBC 和行为")
rows = [
    ["训练观测头", "用带距离标签的图像训练 DINO 投影头，输出估计距离 d_est(t)。"],
    ["冻结模型", "固定前端、控制器和 SBC；校准集与留出集分开。"],
    ["闭环回放", "在独立初始条件下跑轨迹；记录真实距离 d(t)、估计距离 d_est(t) 和误差 e(t)。"],
    ["校准与验证", "每条轨迹取最大误差，定 CP 半径 q；用 q 做 SBC 验证；行为回放单独跑。"],
]
table(s, ["步骤", "做什么"], rows, y=1.80, widths=[2.0, 10.0], font=14, header_font=14, row_h=0.92)
note(s, "CP 只看观测误差；SBC 验证“误差在包络内时风险分数不增”；行为回放是第三类证据。")

# 5 symbols
s = prs.slides.add_slide(BLANK)
title(s, "符号表：每个字母只表示一件事")
table(s, ["符号", "含义", "单位/范围"], [
    ["d(t)", "第 t 步真实距离", "米"],
    ["d_est(t)", "第 t 步估计距离；est=estimated", "米"],
    ["e(t)", "第 t 步估计误差", "米"],
    ["T", "一条轨迹的步数", "步"],
    ["s(j)", "第 j 条轨迹的分数", "米"],
    ["q", "CP 误差包络半径", "米"],
    ["α", "允许超出目标覆盖的比例", "0 到 1"],
    ["B(x)", "安全分数函数", "非负"],
    ["x", "系统状态", "状态向量"],
    ["Cx", "由状态得到的名义观测", "与观测同单位"],
], y=1.62, widths=[1.35, 5.6, 2.4], font=10.8, header_font=10.8, row_h=0.39, header_h=0.35, total_w=8.35, x=0.65)
formula_box(s, 9.25, 1.68, 3.4, 3.2, [
    "e(t) = d_est(t) − d(t)",
    "s(j) = max_t |e(t)|",
    "q = Quantile(",
    "  {s(j)}, 1−α)",
])
txbox(s, 9.25, 4.65, 3.35, 1.6,
      "max_t：对所有步 t 取最大\nQuantile：按分数排序取目标分位\nq：轨迹分数的分位半径", 10.5, False, DARK)
note(s, "e(t)、s(j)、q 不是同一个量；先定义字母，后面才写数字。")

# 6 CP method
s = prs.slides.add_slide(BLANK)
title(s, "CP 先用校准轨迹定 q，再用留出轨迹检查覆盖")
lead(s, "q 在留出评估前冻结；测试/留出数据不参与选 q。")
table(s, ["数值", "含义", "怎么得到", "说明什么"], [
    ["199", "校准轨迹数", "校准集条数", "一条轨迹是一个样本"],
    ["0.05", "允许超出比例", "1 − 目标覆盖 95%", "目标覆盖不是每帧覆盖"],
    ["190", "排序位置 k", "ceil((199+1)×0.95)", "取第 190 个校准分数作为 q"],
    ["100", "留出轨迹数", "独立留出集", "只评估，不参与定 q"],
], y=1.85, widths=[1.4, 2.2, 5.0, 4.0], font=13.5, header_font=13.5, row_h=0.78)
note(s, "n 条校准轨迹 → n 个 max 误差分数 → 排序 → 目标分位 → q；不是把每一帧当样本。")

# 7 AEBS result
s = prs.slides.add_slide(BLANK)
title(s, "AEBS：轨迹 q=0.4184 m，证书通过，行为保持")
lead(s, "24,000 张真实图训练；199 条校准、100 条留出；每条轨迹 400 步。")
table(s, ["数值", "含义", "怎么得到", "说明什么"], [
    ["0.4184 m", "轨迹 CP 半径", "校准轨迹的全程最大误差取目标分位", "新轨迹全程最大误差约 95% 不超过它"],
    ["99/100", "留出覆盖", "100 条新轨迹中 99 条全程最大误差 ≤ 0.4184 m", "计数单位是轨迹，不是帧"],
    ["292,024", "SBC 验证区间数", "连续状态域切盒后枚举的区间总数", "每个区间检查了包络内所有后继"],
    ["0/0", "反例 / 未解决", "验证器输出", "没有找到正漂移见证和未解决盒子"],
    ["4,800/4,800", "行为成功 / 总回放", "12 组外观，每组 400 条回放", "行为全部成功；不等于所有状态安全"],
], y=1.78, widths=[1.4, 2.2, 5.0, 4.0], font=12.5, header_font=12.5, row_h=0.72)
note(s, "0.4184 m 是观测误差包络；证书是“包络内 B 不增”；4,800/4,800 是行为证据。")

# 8 q interpretation
s = prs.slides.add_slide(BLANK)
title(s, "同一个前端，三个不同的误差数字")
table(s, ["数值", "是什么", "怎么得到", "说明什么"], [
    ["95%", "目标覆盖", "预设目标覆盖率", "单帧和轨迹 CP 使用同一目标"],
    ["0.0874 m", "平均单帧 MAE", "所有帧 |d_est(t)−d(t)| 取平均", "平均准，不代表最坏帧小"],
    ["0.2307 m", "单帧 CP q", "一张图一个样本；单帧误差取目标分位", "只管单帧；不能当整条轨迹包络"],
    ["0.4184 m", "轨迹 CP q", "每条轨迹先取全程最大误差，再取目标分位", "证书需要的是整条轨迹包络"],
], y=1.85, widths=[1.4, 2.2, 5.0, 4.0], font=13.5, header_font=13.5, row_h=0.82)
note(s, "0.4184 m 不是平均误差，也不是“安全概率”；它回答整条轨迹的最大观测误差可能有多大。")

# 9 cGAN comparison
s = prs.slides.add_slide(BLANK)
title(s, "原 cGAN 与新 DINO：同一任务，两种前端")
lead(s, "原 cGAN 是“状态→生成图→state_net”；新 DINO 是“真实图→DINO→投影”。")
table(s, ["数值", "含义", "怎么得到", "说明什么"], [
    ["1.9156 m", "原 cGAN 轨迹 q", "原生 cGAN 生成图闭环，轨迹 CP", "原系统需要的包络更大"],
    ["92/100", "原 cGAN 留出覆盖", "100 条新轨迹中 92 条在 1.9156 m 内", "少数轨迹超出包络"],
    ["0.4184 m", "新 DINO 轨迹 q", "真实图训练，同一轨迹 CP 协议", "新系统包络更小"],
    ["99/100", "新 DINO 留出覆盖", "100 条新轨迹中 99 条在 0.4184 m 内", "覆盖更高，行为没有下降"],
], y=1.85, widths=[1.4, 2.2, 5.0, 4.0], font=13.5, header_font=13.5, row_h=0.80)
note(s, "这是部署系统对照；输入类型、训练数据和预训练不同，不能写成“DINO 架构全面胜出”。")

# 10 transfer
s = prs.slides.add_slide(BLANK)
title(s, "5.221 m 是“旧 state_net 直接读真实图”的迁移误差")
lead(s, "把只在生成图训练的旧 state_net 直接用于真实灰度图；不是原 cGAN 自己生成图闭环。")
table(s, ["数值", "含义", "怎么得到", "说明什么"], [
    ["24,000", "真实图总数", "本轮数据总包", "迁移实验用的真实图规模"],
    ["5.221 m", "迁移轨迹 q", "旧 state_net 读真实灰度图；轨迹 CP", "跨分布迁移误差大"],
    ["87.64%", "迁移留出覆盖", "以留出轨迹总数为分母，误差 ≤5.221 m 的占比", "只是该迁移协议的覆盖率"],
    ["1.9156 m", "原生 cGAN q", "原 cGAN 自己生成图闭环", "原方法应引用这个数"],
], y=1.85, widths=[1.4, 2.2, 5.0, 4.0], font=13.5, header_font=13.5, row_h=0.80)
note(s, "5.221 m 不能当成“原 cGAN 的误差”；否则会夸大新旧差距。")

# 11 budget setup
s = prs.slides.add_slide(BLANK)
title(s, "数据预算：总图像数改变时，排序会不会变？")
lead(s, "每档各自划分 train/val/test；cGAN 的 state_net 用真实灰度图训练，DINO 用 RGB；同一档共用 test。", size=12.5)
table(s, ["总包量", "含义", "test 数量", "这档回答什么"], [
    ["400", "总图像数", "72 张", "小数据预算"],
    ["10,000", "总图像数", "2,000 张", "中数据预算"],
    ["24,000", "总图像数", "4,920 张", "大数据预算"],
], y=1.85, widths=[1.8, 2.6, 2.4, 4.0], font=15, header_font=14, row_h=0.80)
note(s, "三档 test 集不同，只能做同一档内比较；不能把三档连成 scaling 曲线。")

# 12 400
s = prs.slides.add_slide(BLANK)
title(s, "400 张总包：DINO 的 MAE 和 q 都更小")
lead(s, "总包 400 张；test 72 张；cGAN 和 DINO 都在真实图上训练/评估，共用同一 test。")
table(s, ["方法", "MAE", "q", "覆盖"], [
    ["cGAN realhead", "0.694 m", "4.169 m", "71/72"],
    ["DINO", "0.298 m", "0.865 m", "69/72"],
], y=1.95, widths=[2.9, 2.4, 2.4, 3.0], font=17, header_font=13.5, row_h=0.85)
txbox(s, 0.85, 4.25, 11.8, 0.55, "400=总图像数；72=test 图像数；MAE=平均单帧误差；q=单帧 CP；覆盖=误差≤q 的 test 张数。", 12, False, DARK)
txbox(s, 0.85, 4.55, 11.8, 0.45, "0.694/0.298=平均单帧误差；4.169/0.865=单帧 CP 半径；71/72、69/72=72 张 test 中覆盖数。", 10.5, False, DARK)
txbox(s, 0.85, 5.25, 11.8, 0.75, "DINO 的 MAE/q 更低；cGAN 覆盖略高，但 q 明显更大。", 16, False, BLACK)
note(s, "小数据预算下，DINO 的误差/包络更好；覆盖不是单看数字大小。")

# 13 10k
s = prs.slides.add_slide(BLANK)
title(s, "10,000 张总包：真实图 cGAN 头的 q 更小")
lead(s, "总包 10,000 张；test 2,000 张；cGAN/DINO 共用同一 test。")
table(s, ["方法", "MAE", "q", "覆盖"], [
    ["cGAN realhead", "0.0085 m", "0.0209 m", "1894/2000"],
    ["DINO", "0.0205 m", "0.0557 m", "1893/2000"],
], y=1.95, widths=[2.9, 2.4, 2.4, 3.0], font=17, header_font=13.5, row_h=0.85)
txbox(s, 0.85, 4.25, 11.8, 0.55, "10,000=总图像数；2,000=test 图像数；MAE=平均单帧；q=单帧 CP；覆盖=误差≤q 的 test 张数。", 12, False, DARK)
txbox(s, 0.85, 4.55, 11.8, 0.45, "0.0085/0.0205=平均单帧误差；0.0209/0.0557=单帧 CP 半径；1894/2000、1893/2000=2,000 张 test 中覆盖数。", 10.0, False, DARK)
txbox(s, 0.85, 5.25, 11.8, 0.75, "cGAN 的 MAE/q 更小；覆盖几乎相同。", 16, False, BLACK)
note(s, "中数据预算下，真实图 cGAN 头的 q 更好。")

# 14 24k
s = prs.slides.add_slide(BLANK)
title(s, "24,000 张总包：cGAN q 更小，但覆盖更低")
lead(s, "总包 24,000 张；test 4,920 张；cGAN/DINO 共用同一 test。")
table(s, ["方法", "MAE", "q", "覆盖"], [
    ["cGAN realhead", "0.0247 m", "0.0307 m", "4382/4920"],
    ["DINO", "0.0874 m", "0.2307 m", "4592/4920"],
], y=1.95, widths=[2.9, 2.4, 2.4, 3.0], font=17, header_font=13.5, row_h=0.85)
txbox(s, 0.85, 4.25, 11.8, 0.55, "24,000=总图像数；4,920=test 图像数；MAE=平均单帧；q=单帧 CP；覆盖=误差≤q 的 test 张数。", 12, False, DARK)
txbox(s, 0.85, 4.55, 11.8, 0.45, "0.0247/0.0874=平均单帧误差；0.0307/0.2307=单帧 CP 半径；4382/4920、4592/4920=4,920 张 test 中覆盖数。", 10.0, False, DARK)
txbox(s, 0.85, 5.25, 11.8, 0.75, "cGAN 的 q 更小，但 test 覆盖反而更少；q 和覆盖必须一起看。", 16, False, BLACK)
note(s, "大数据预算下，排序再次变化：没有单向赢家。")

# 15 joint C
s = prs.slides.add_slide(BLANK)
title(s, "联合 C：5 对训练种子，没有稳定收益")
lead(s, "同一数据、同一前端结构、同一训练预算；固定 C 与联合 C 各训练 5 次。")
table(s, ["数值", "含义", "怎么得到", "说明什么"], [
    ["5 对", "训练种子对数", "两个方法各训练 5 次", "统计单位是种子，不是帧"],
    ["0.085263 / 0.085699", "q 均值", "5 个种子的 CP q 取平均", "联合 C 略大，不是更小"],
    ["95.6% / 95.0%", "覆盖均值", "5 个留出集覆盖率取平均", "联合 C 没有更高"],
    ["5/5 / 5/5", "SBC 通过", "两个方法各 5 个种子都通过", "可行，但不等于收益"],
    ["3/5", "联合 q 更小的对数", "5 对配对中联合更小的次数", "不到稳定多数"],
], y=1.78, widths=[2.2, 2.1, 4.6, 3.1], font=12.5, header_font=12.5, row_h=0.72)
note(s, "联合训练可行，但没有独立 q/覆盖收益。")

# 16 lane
s = prs.slides.add_slide(BLANK)
title(s, "车道：2D 轨迹 CP 和 SBC 都通过")
lead(s, "2D 车道状态；每条轨迹 80 步；199 条校准、100 条留出；q 是归一化 latent 坐标。")
table(s, ["数值", "含义", "怎么得到", "说明什么"], [
    ["0.090093", "轨迹 q", "归一化 latent；每条取 80 步最大误差，再取目标分位", "这是归一化量，不是米"],
    ["199", "校准轨迹数", "车道校准集", "一条轨迹是一个样本"],
    ["80", "每条轨迹步数", "闭环回放长度", "取全程最大误差"],
    ["95%", "目标覆盖", "预设目标覆盖率", "轨迹级边缘覆盖"],
    ["190/199", "校准覆盖", "199 条中 190 条在 q 内", "校准集目标秩"],
    ["95/100", "留出覆盖", "100 条新轨迹中 95 条在 q 内", "计数单位是轨迹"],
    ["0/0", "反例 / 未解决", "SBC 连续域验证", "没有正漂移见证和未解决盒子"],
    ["12/12", "图像闭环成功", "12 条图像回放全部完成", "行为成功，不等于概率证书"],
], y=1.68, widths=[1.4, 2.2, 5.3, 3.1], font=9.8, header_font=10.5, row_h=0.48, header_h=0.38)
note(s, "车道主结果仍是常量盒；q 很小，证书通过。")

# 17 radius
s = prs.slides.add_slide(BLANK)
title(s, "同一 actor/SBC：0.12 通过，0.16 失败")
lead(s, "固定 actor、SBC、动力学和验证器，只改变误差半径。")
table(s, ["数值", "含义", "怎么得到", "说明什么"], [
    ["0.12", "r_cert", "已测试半径中通过验证的最大值", "当前证书的数值容忍下界"],
    ["0.16", "失败半径", "同一验证器在更大半径下检查", "出现实际见证和未解决，不能宣称通过"],
    ["1.42×", "半径余量", "0.12 ÷ 0.084338", "联合 C 的 q 到 r_cert 的倍数"],
    ["0.090093", "常量 q", "车道主证书半径", "低于 r_cert"],
    ["0.084338", "联合 q", "联合 C 的轨迹半径", "也低于 r_cert"],
], y=1.85, widths=[1.4, 2.0, 4.8, 4.0], font=13, header_font=13, row_h=0.76)
note(s, "0.12 是“已验证通过的最大测试点”，不是理论极限；更细的中间半径未测。")

# 18 parking/robot
s = prs.slides.add_slide(BLANK)
title(s, "行为成功不等于安全证书")
lead(s, "停车场使用受保护闭环和在线后备；机器人是三维视觉闭环。")
table(s, ["数值", "含义", "怎么得到", "说明什么"], [
    ["30/30", "停车场受保护闭环成功", "30 条图像仿真回放", "依赖在线后备；不是原无过滤器全域"],
    ["233/240", "停车场启动审计覆盖", "240 个独立启动样本中 233 个被包络覆盖", "审计观察值，不是新理论置信度"],
    ["24/24", "机器人图像闭环成功", "24 条图像回放", "任务能完成"],
    ["28/32", "机器人 CP 盒内反例", "32 个盒内检查案例中 28 个确认违反安全条件", "完整 SBC 未通过"],
], y=1.85, widths=[1.4, 2.4, 4.9, 4.0], font=13, header_font=13, row_h=0.82)
note(s, "固定控制器 + 固定 CP 盒不成立时，继续只训 SBC 不够。")

# 19 claims
s = prs.slides.add_slide(BLANK)
title(s, "能写与不能写")
txbox(s, 0.75, 1.55, 5.8, 0.45, "能写", 20, True, BLACK)
bullet_list(s, 0.85, 2.10, 5.7, 3.6, [
    "真实图 DINO 部署系统相对原 cGAN：轨迹包络更小、留出覆盖更高、任务行为保持",
    "小数据预算下 DINO 的 MAE/q 更好",
    "联合 C 可行，但没有稳定独立收益",
    "车道常量盒证书通过，r_cert 有数值余量",
], size=14)
txbox(s, 6.85, 1.55, 5.8, 0.45, "不能写", 20, True, BLACK)
bullet_list(s, 6.95, 2.10, 5.7, 3.6, [
    "DINO 架构在所有数据预算下全面优于 cGAN",
    "联合 C 提高可观测性或安全",
    "原 cGAN 已被证明不安全",
    "新系统在实车/新相机上安全",
    "迁移口径的 q 是原生 cGAN 的 q",
], size=14)
note(s, "所有结论都限定在保存图像/渲染图、注册分布、数值验证范围内。")

# 20 next/summary
s = prs.slides.add_slide(BLANK)
title(s, "下一步与总结")
txbox(s, 0.75, 1.45, 5.8, 0.4, "还缺的实验", 18, True, BLACK)
bullet_list(s, 0.85, 1.95, 5.8, 3.5, [
    "固定 common test：各预算评估同一批图",
    "同输入口径：两边都用同样的低分辨率灰度输入",
    "多种子、同训练/调参预算：把种子作为统计单位",
    "同 actor/SBC 的轨迹 CP + 证书：静态 q 不能替代闭环证书",
], size=13)
txbox(s, 6.85, 1.45, 5.8, 0.4, "结论收束", 18, True, BLACK)
bullet_list(s, 6.95, 1.95, 5.7, 3.5, [
    "部署系统：真实图 DINO 的轨迹误差包络更小，任务行为保持",
    "同预算：小数据 DINO 更好；中/大数据 cGAN q 更小，但大数据覆盖更低",
    "联合 C：可行，但没有独立收益",
], size=13)
txbox(s, 0.85, 5.95, 11.5, 0.45, "不把“部署收益”写成“架构全面胜出”。", 15, False, GRAY)

prs.save(str(OUT))
print(OUT)
