# 灰度 AEBS 联合 C 训练

训练冻结 DINO 的视觉投影和逐行归一化 C。辅助速度是静态图片模型的人工状态增广，不是采集到的速度标签。

精确矩阵、训练预算、内部验证选择和数据来源见 report.json、protocol.json。v1 的预算为 300 epoch；v2 为 1000 epoch，按开发验证选择第 507 epoch。原图部署与缓存重载检查见 v2 的 raw_image_reload.json。

此目录仅表示前端训练完成，不继承旧安全证书；v2 对应的新常量轨迹 CP/SBC 检查未通过，见相邻 gray_aebs_vision_sls_joint_C_20261109_cp_v2 目录。
