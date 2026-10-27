"""Archive superseded prose reversibly; leave scientific evidence and programs in place."""
from pathlib import Path
import hashlib
import json
import re
import shutil
from urllib.parse import unquote

ROOT=Path('/Users/johann/project/work_based_on_spvc')
BASE=Path('/Users/johann/Documents/Codex/2026-10-22/jian/outputs')
ARCHIVE=ROOT/'docs/archive/2026-10-26_VISION-SLS_CP阶段文档'
MASTER=ROOT/'docs/current/VISION-SLS_CP完整说明与可行性分析.md'
ARCHIVE.mkdir(parents=True,exist_ok=True)
sources=[ROOT/'docs/current'/n for n in [
 'VISION-SLS_CP误差界可行性.md',
 'VISION-SLS_CP误差界可行性_导师汇报稿.md',
 'VISION-SLS_CP全部尝试结果总账.md',
]]
# These are the stage report exports, not their raw result directories.
sources+=sorted(BASE.glob('vision_sls*/*.md'))
assert len(sources)==16,len(sources)
mapping={p:ARCHIVE/('project_docs' if p.is_relative_to(ROOT) else 'exports')/p.relative_to(ROOT/'docs' if p.is_relative_to(ROOT) else BASE) for p in sources}
pattern=re.compile(r'(!?\[[^\]\n]*\])\((<[^>]+>|[^)\s]+)(\s+"[^"\n]*")?\)')

def rewrite_links(text,old_path):
    def replace(m):
        raw=m[2].strip('<>')
        if raw.startswith(('#','http:','https:','mailto:','data:','codex:','plugin:','app:')):
            return m[0]
        value,mark,fragment=raw.partition('#')
        if not value:return m[0]
        candidate=Path(unquote(value))
        candidate=(candidate if candidate.is_absolute() else old_path.parent/candidate).resolve()
        target=mapping.get(candidate,candidate)
        suffix=('#'+fragment) if mark else ''
        title=m[3] or ''
        return f'{m[1]}(<{target}{suffix}>{title})'
    return pattern.sub(replace,text)

manifest=[]
for source,target in mapping.items():
    assert source.is_file(),source
    assert not target.exists(),target
    original=source.read_bytes()
    target.parent.mkdir(parents=True,exist_ok=True)
    # Keep an exact original text in a single recovery bundle. Readable archived
    # copies add a historical notice and repair links for the new location.
    manifest.append({'original_path':str(source),'archive_path':str(target),
                     'original_sha256':hashlib.sha256(original).hexdigest(),
                     'original_bytes':len(original)})

import zipfile
bundle=ARCHIVE/'归档前原文_可恢复.zip'
assert not bundle.exists()
with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for source,target in mapping.items():
        z.writestr(str(target.relative_to(ARCHIVE)),source.read_bytes())

for entry in manifest:
    source=Path(entry['original_path']);target=Path(entry['archive_path'])
    original=source.read_text()
    revised=rewrite_links(original,source)
    first,sep,rest=revised.partition('\n')
    notice=f'\n\n> 历史归档：本稿只代表该阶段。CP 的现行完整说明与全部结论见 [统一主报告](<{MASTER}>)。原文路径与哈希见本归档清单；本可读副本只增加此提示并迁移文件链接，逐字原文保存在同目录恢复压缩包。\n'
    target.write_text(first+notice+'\n'+rest)
    source.unlink()
    entry['archive_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    entry['recovery_zip_entry']=str(target.relative_to(ARCHIVE))

manifest_data={'date':'2026-10-26','main_report':str(MASTER),
 'moved_documents':len(manifest),'raw_data_moved':False,'experiment_code_moved':False,
 'recovery_bundle':str(bundle),'recovery_bundle_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),
 'entries':manifest}
(ARCHIVE/'归档清单.json').write_text(json.dumps(manifest_data,ensure_ascii=False,indent=2)+'\n')

rows=['| 原位置 | 归档文件 |','|---|---|']
for entry in manifest:
    old=Path(entry['original_path'])
    label=str(old.relative_to(ROOT)) if old.is_relative_to(ROOT) else 'outputs/'+str(old.relative_to(BASE))
    rows.append(f"| {label} | [{old.name}](<{entry['archive_path']}>) |")
readme=f'''# VISION-SLS CP 阶段文档归档

整理日期：2026 年 10 月 26 日。这里用于追溯历史，不是日常阅读入口。

**现在只需阅读 [CP 加入 VISION-SLS 统一主报告](<{MASTER}>)。** 主报告已经整合方法来由、符号、手算例子、代码链路、全部 22 项尝试、12 个回顾参数点、19 个图像批次及可行性分析。

本次移入 16 份重复/阶段说明：3 份原 current CP 文档，10 份复现/CP 阶段导出稿，以及 3 份逐步详解导出副本、备份和修订说明。逐步详解的现行参考版本仍在 docs/references，未删除独有内容。

没有删除或移动任何原始 JSON、NPZ、模型、图片、视频、日志或实验程序。归档可读稿增加历史提示，并把相对路径和相互引用迁移到有效位置；正文的实验数字与结论没有改写。

如需逐字恢复整理前原文，使用 [原文恢复压缩包](<归档前原文_可恢复.zip>)，按 [归档清单](<归档清单.json>) 中的 original_path 放回。压缩包每个原文的 SHA256 已核对；恢复时注意不要覆盖后来新建的文件。

历史稿中“当前”“下一步”“最新”只对当时有效。一些早期结论已经被后续状态相关实验限定，不能单独据此宣称当前 CP 安全替换已经完成。

## 原位置与可读归档对照

{chr(10).join(rows)}

## 保留在现行目录的其他专题

- 原论文与代码解读：用于查理论和原代码；其中 CP 历史片段以主报告为准。
- 图像到低维表示、停车场前端接 SPVC/PPO/SBC：包含其他独立实验，不属于本轮 SLS CP 的重复阶段稿。
- 原始实验目录：继续保存原位置，由主报告附录直接链接。

历史报告生成脚本保留以便追溯，但它们生成的是历史文稿，不是当前主报告。不要将重新生成的旧阶段稿作为现行入口。
'''
(ARCHIVE/'README.md').write_text(readme)

# Live references should lead readers to the consolidated result.
changed=[]
for folder in [ROOT/'docs/current',ROOT/'docs/references']:
    for path in folder.glob('*.md'):
        if path==MASTER:continue
        text=path.read_text();updated=text
        for source in mapping:
            updated=updated.replace(str(source),str(MASTER))
        if path.name in ['VISION-SLS_论文与代码逐步详解.md','VISION-SLS_图像到低维表示与误差.md']:
            first,sep,rest=updated.partition('\n')
            notice=f'\n\n> CP 专题阅读入口（2026-10-26）：[CP 加入 VISION-SLS 的完整说明与可行性分析](<{MASTER}>)。本文件保留原论文/视觉前端及独立专题内容；其中 CP 阶段记录不再作为当前实验结论入口。\n'
            if 'CP 专题阅读入口（2026-10-26）' not in updated:updated=first+notice+'\n'+rest
        if updated!=text:path.write_text(updated);changed.append(str(path))

with zipfile.ZipFile(bundle) as z:
    for entry in manifest:
        assert hashlib.sha256(z.read(entry['recovery_zip_entry'])).hexdigest()==entry['original_sha256']
print(json.dumps({'archived_documents':len(manifest),'live_reference_files_updated':changed,
                  'exact_originals_verified':len(manifest)},ensure_ascii=False))
