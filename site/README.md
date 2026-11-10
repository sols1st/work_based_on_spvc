# SPVC 实验可视化站点

轻量静态站点，无前端框架、无外部 CDN。包含：

- `index.html`：总览与四个实验卡片
- `aebs.html`、`lane.html`、`parking.html`、`robot.html`：配置、逐步输入/输出、结果、保证边界、精选图片
- `gallery.html`：精选输入/结果缩略图
- `process.html`：四个实验逐步详细过程（输入/操作/输出/为什么）
- `compare.html`：所有方法对比（同一 test 前端误差、原灰度实现、CP/SBC、四实验状态）
- `data.html`：数据/配置/结果包/完整报告链接

## 本地打开

直接双击 `site/index.html` 即可（相对链接可访问 `../artical-F122/...` 和 `../docs/...`）。

## 用本地服务器

从**项目根目录**启动（不要从 `site/` 启动，否则 `../artical` 链接会越界）：

```bash
cd /Users/johann/project/work_based_on_spvc
python3 -m http.server 8765
```

打开：

```text
http://127.0.0.1:8765/site/index.html
```

## 重新生成

```bash
cd /Users/johann/project/work_based_on_spvc
.venv-vision-sls/bin/python site/build_site.py
```

图片在 `site/assets/img/`，已压缩到约 1.5 MB；原始 24k/3.6k 图片不复制。
