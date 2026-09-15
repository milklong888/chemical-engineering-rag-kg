# 来源与提取状态

本页描述独立项目 `chemical-engineering-rag-kg` 的四源公开预览，以及与之明确分开的旧八源历史基线。

## 当前四源预览

| source_id | 来源题名 | 版本/版次 | 选定 KU | 主题根 |
|---|---|---|---:|---|
| RE01 | 《化学反应工程》 | 第三版 | 14 | `reaction-engineering` |
| OC02 | 《化工过程的优化设计与控制》 | 版次未核 | 13 | `chemical-engineering-principles` |
| TH03 | 《化工热力学》 | 第二版 | 25 | `chemical-engineering-principles` |
| EN04 | 《过程工业能量系统优化——换热网络与蒸汽动力系统》 | 第一版，第一次印刷 | 17 | `chemical-engineering-principles` |
| **合计** | **4 个来源卷** |  | **69** | **2 个主题根** |

预览图谱为 177 个节点、191 条内部边；另有 252 条外部关联，记录在 [`external_references.jsonl`](../knowledge/curated-four-books-v1/external_references.jsonl)，目标命名空间为 `chemical-engineering-skills`。外部关联不创建内部节点，不复制外部正文。

本批保存的是经审阅的原创工程释义及其定位元数据，不是四本资料的全文数字化。四个来源卷的 `full_source_digitization_complete` 均为 `false`；公开包不含原始 PDF、扫描页、页图或 OCR 全文。页码、定位字符串和来源 SHA 用于合法来源复核，来源 SHA 不等于页媒体哈希。

每个 KU 只在有独立实际用途、内容及必要条件可无歧义复述时保留；重复、纯导航、无信息增量的内容跳过，不设“每章必须多少块”的微粒度配额。读取时应一次读完完整 KU 的标题、正文、适用条件、单位/基准和证据定位，再按需沿 `related_to` 补读；该关系不表示因果或可替代关系。实际项目输入仍须核实适用条件和当前来源，不能把教材示例值直接当作项目值。

资格字段显式为真才可进入相应阶段；缺少字段时 fail closed，不根据题名、来源类型或向量命中自动补资格。来源疑点或当前项目证据不足时保持待核或排除直接答案。

## 公开包文件

公开包入口为 [`knowledge/curated-four-books-v1/`](../knowledge/curated-four-books-v1/)：

1. [`source_manifest.json`](../knowledge/curated-four-books-v1/source_manifest.json)：四个来源卷、书目、主题根和审核证明；
2. [`knowledge_units.jsonl`](../knowledge/curated-four-books-v1/knowledge_units.jsonl)：69 个完整 KU，保留原 node ID、正文、适用范围和量纲/基准；
3. [`evidence_registry.jsonl`](../knowledge/curated-four-books-v1/evidence_registry.jsonl)：每条 KU 的来源页区间、定位和正文哈希；
4. [`kg_nodes.jsonl`](../knowledge/curated-four-books-v1/kg_nodes.jsonl) / [`kg_edges.jsonl`](../knowledge/curated-four-books-v1/kg_edges.jsonl)：粗粒度层级和关系；
5. [`external_references.jsonl`](../knowledge/curated-four-books-v1/external_references.jsonl)：外部项目关联；
6. [`conversion_report.json`](../knowledge/curated-four-books-v1/conversion_report.json)：确定性转换结果，不是独立审计报告。

语义向量已按合同生成并公开：82 个分段、82×512 维，覆盖全部 69 个 KU；13 个长 KU 仅作索引分段。16 个公开正例开发题，lexical/dense/hybrid 均为 hit@5=1.0、MRR@5=1.0、首位命中率=1.0；不是封存测试，不代表拒答能力通过。详见[向量清单](../knowledge/curated-four-books-v1/vector_manifest.json)、[开发评测](../reports/curated-dev-evaluation.json)及[独立校验](../reports/curated-independent-validation.json)。人工批准锚点为 0，生产激活为 `false`。

## 读取和验证入口

入库判断以 [READING_GUIDE.md](READING_GUIDE.md)、[INGESTION_SPEC.md](INGESTION_SPEC.md) 和 [`../contracts/curated_knowledge_contract.json`](../contracts/curated_knowledge_contract.json) 为准。公开结构检查使用：

```powershell
python scripts/validate_curated_bundle.py `
  --bundle knowledge/curated-four-books-v1 `
  --require-vectors
```

查询使用当前模块的 lexical 入口：

```powershell
python src/query_curated.py `
  --bundle knowledge/curated-four-books-v1 `
  --query "换热网络中的夹点目标" `
  --method lexical `
  --limit 5
```

源记录转换由维护者使用 `src/build_curated_bundle.py` 完成；公众不需要取得私有父级映射、审计或源记录输入。

## 历史：旧八源基线

以下事实沿用正式来源状态记录，时间基准为 2026-08-21，状态说明更新至 2026-09-09。它们是旧八源历史，不与本节四源 69 KU 混算。

| ID | 页数 | Identity | Page baseline | Authority / Projection 历史状态 | 旧 Global |
|---|---:|---|---|---|---|
| S001 | 371 | PASS | 371/371 | 全书 11 个语义包候选的结构与映射通过独立检查，未通过来源内容准入；蒸发章 p301–331 独审发现的 3 项问题已修复并独立关闭，13 条来源疑点仍待核；p369 一条答案解释修复已验收，旧候选隔离状态未改 | 否 |
| S002 | 316 | PASS | 316/316 | 全书 10 个语义包候选的结构与映射通过独立检查，未通过来源内容准入；包外页处置已独审通过，全文校订和逐题答案关联仍待核对 | 否 |
| S003 | 177 | PASS | 177/177 | 7 个化工原理包、2 个反应工程包；两章 39 页正文独审发现的 49 项问题已分批修复并独立关闭，4 项来源冲突与 25 条校订注保留；未获整章准入，旧 48 条公式的补漏验收不能沿用为全文通过 | 否 |
| S004 | 1,159 | PASS | 1,159/1,159 | 第一章 p14–92（79 页、200 个主对象）的 1 项解释修复已独立关闭，57 个对象的来源疑点保留；第二章 p93–134（42 页、50 个对象）的指定修复已独立关闭，30 条来源疑点保留；第三章 p135–267 的 133 页、199 个对象已完成分段独审，6 项定点修复已通过独立复核，59 条来源疑点保留；第四章 p268–356 的 12 项限定修复已交独立闭环报告，待维护方接收，80 条源疑点未解；第五章 p357–481 的 125 页、306 个对象已交付，独审进行中；未获全书准入 | 否 |
| S005 | 427 | PASS | 427/427 | 旧投影覆盖 p33–413 的 12 包；C01 的 3 处漏登引用、前后置页的 5 项书目修复均已独立复核；仍有 23 个书目字段待核，未获全书准入 | 是，阶段版；不含新 C01 |
| S006 | 547 | PASS | 547/547 | 9 包；第一章 p12–86 的 75 页、495 个内容块、86 个对象及 49 处可见表体已完整独审，2 项修复已独立关闭，26 条来源疑点保留；第二章 p87–183 的 97 页、595 个内容块、187 个对象及 84 处可见表体独审已接收，4 项问题经原页确认并安排修复，40 条旧来源疑点未解；第三章 p184–253 的 70 页、394 个内容块、138 个对象及 73 处可见表体独审进行中，32 条来源疑点保留；软件案例未运行复现，整章正式准入及全书内容验收未完成 | 否 |
| S007 | 398 | PASS | 398/398 | 674 个对象分三批修复；第 1、2 批对象检查已通过，第 3 批待处理；不代表全书正文及完整表体已校订 | 否 |
| S008 | 582 | PASS | 582/582 | 全书阶段性语义投影与主要 source-detail 审计已完成 | 是，阶段版 |

旧八源全局阶段快照为 24 packages、2,423 RAG chunks、2,350 retrieval-eligible、2,326 embedding-eligible、2,217 evidence records、2,308 KG nodes、2,786 KG edges、2,423 chunk→KG maps、72 ordered relations；其中 2,326 条向量属于旧私有阶段数据。旧记录的 6 项未通过门槛继续保持历史状态，不作为四源预览的当前结果。

旧基线中的页数、对象数、阶段索引和门禁均不用于推算本批内容覆盖或向量评测。本项目不把历史 8 源资料、3,977 页和旧私有向量与本批四源包合并相加。
