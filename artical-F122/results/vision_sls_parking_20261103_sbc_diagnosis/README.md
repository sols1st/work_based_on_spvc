# 新闭环 SBC 求解

名义候选和 CP 候选均未通过完整验证；已确认 CP 盒内可达超速反例。summary.json 绑定了模型、候选与反例。控制器没有修改，安全概率未认证。

```bash
python -m Aebs.vision_sls_parking.train_planned_sbc --candidate results/vision_sls_parking_20261103_planned_candidate --output-dir results/new_nominal_sbc_run --epochs 8000
python -m Aebs.vision_sls_parking.search_planned_error --candidate results/vision_sls_parking_20261103_planned_candidate --error-model results/vision_sls_parking_20261102_error_v1/error_model.json --output-dir results/new_cp_error_search
```
