# 入库规范（Contributor Intake Spec v0.2）

这是正式入库包的公开验收规范。只想提交解释或纠错，可以先使用 [简化贡献入口](../CONTRIBUTING.md#最简单的参与方式)，由维护者完成结构转换与整包验收。

普通贡献的最低要求是：**来源与页码、对象范围、自己的解释、必要条件、未确定之处、公开权限说明**。一条公式或一张图表也可以提交；看过原页与必要上下文，把它讲什么、怎么用、有什么限制写明白即可，无须额外生成向量或细粒度图谱。

公开贡献按本文评审；内部格式如有不同，由维护者提供明确的转换说明，不能要求外部贡献者自行查阅未公开合同。`contracts/` 与 `schemas/` 是阶段性参考，不代表一条命令就能完成全部内容验收。

## 1. 来源身份与权限

每个正式登记来源必须提供（局部贡献可引用已有 `source_id`，无须重复提供整份文件信息）：

- `source_id`：仓库内唯一，如 `S009`；
- 题名、版本/版次、出版信息（已知时）；
- 取得渠道与核对方式：例如自有原稿、出版社公开页面或已授权提供；只记录可公开信息，未知项明确写“未核实”，不填写私密链接或本机路径；
- 文件字节数、SHA-256、PDF 总页数；
- 是否加密、是否可正常逐页读取；
- 权限状态：自有原创、已获授权、仅本地研究、或未知；
- 公共发布范围。权限未知时只接收元数据和非重建性描述。

禁止通过文件名、相对路径或 OCR 内容猜测来源身份。相同 SHA-256 视为同一个物理来源。

## 2. 页账本：exactly once

每个 PDF 页必须在 `page_ledger.jsonl` 中恰好出现一次：

```json
{"source_id":"S009","pdf_page":1,"printed_page":null,"physical_container_id":"s009-front","semantic_package_id":"s009-front-cover","content_class":"no_value","ocr_status":"available","evidence_sha256":"<sha256>"}
```

要求：

- 页码从 1 连续到 `total_pdf_pages`；
- 不得缺页、重复或重叠；
- `printed_page` 可以为空，但不得用印刷页码替代 PDF 页码；
- 空白页、封面、目录、参考文献、广告也必须入账；
- page hash 必须绑定实际证据字节，不能只检查格式。

这是全书整合后的要求；单章或单对象贡献只声明本次覆盖的页段，不得把未处理页标为已完成。

## 3. 物理容器与语义包分层

`physical_container` 描述原始章节包或来源文件的物理范围；`semantic_package` 描述入库语义范围。两者不能混为一谈。

一个物理尾包可以拆成多个互斥语义包，例如：

- 附录：`answerable_content`；
- 习题答案：`supporting_content`；
- 参考文献：`evidence_only`；
- 封底或广告：`no_value`。

每个语义包只能有一个 `content_class`。允许的正式值及含义如下：

| `content_class` | 用途 |
|---|---|
| `answerable_content` | 能直接回答工程问题的正文、公式、图表或附录 |
| `supporting_content` | 依赖主内容理解的答案、补充说明等上下文 |
| `evidence_only` | 仅保留证据、来源或书目信息 |
| `relation_only` | 仅保留目录、顺序或引用关联 |
| `front_matter` | 非直接回答内容的前置页 |
| `back_matter` | 非直接回答内容的后置页 |
| `no_value` | 无检索价值，但仍需保留页账本 |

v0.1 中的简写 `answerable`、`supporting` 在正式整合时分别转为 `answerable_content`、`supporting_content`；普通文字投稿无需自己填写这些字段。

物理容器清单和语义包清单必须分别 exactly once 覆盖全书。未落入旧物理章节包的前置页要使用保留的前置页记账容器，并明确标为 `uncontained_front_matter`，不得静默丢弃或当作可检索正文。

## 4. 顶层主题与层级

只允许：

- `chemical-engineering-principles`（化工原理）
- `reaction-engineering`（反应工程）

标准层级：

```text
subject_root → source_volume → chapter_or_support_unit → coarse_KU → evidence
```

禁止章节直接挂在 root；禁止把书名、卷名或 `chemical_data` 变成新 root。

正式来源卷 ID 使用 `volume:{source_id小写}:{subject_root}:{document_slug}`。同一资料涉及两个顶层主题时分别建来源卷，仍共用原始来源身份，不复制 PDF 或重复计算页数。

## 5. 文本块

每个正文块至少包含：

- 稳定 `block_id`；
- `source_id`、`semantic_package_id`、`pdf_page`；
- 标题路径与章节归属；
- 规范化文本；
- OCR/原生文本来源类型；
- evidence reference 与 hash；
- `retrieval_eligible`、`embedding_eligible` 和排除原因。

文本按可检索语义切分，不能只按固定字符数破坏公式、定义或上下文。

## 6. 视觉对象解释

所有解释先读对象附近正文，再读对象本身。禁止仅根据检测类别或图号套模板。

解释以读懂工程含义为度：清楚写出内容、用途、必要条件，不要求冗长教学。以下项目只填写适用且能从来源确认的内容；未给出、看不清或不确定时明确记录，不为凑齐字段补造事实。处理整页/整章时还要检查检测框之外是否有遗漏对象。

### 公式

至少给出：

- 精确转写与式号；
- 用途；
- 主要变量含义；
- 可见单位或量纲；
- 适用条件、经验范围或限制；
- 与邻式/表/图的关系。

检测框中的标题、正文引用或重复框必须标为 false/duplicate，不得伪造公式记录。

### 表格

至少给出：

- 表号与表题；
- 表头、单位和关键字段；
- 代表性行或选择规则；
- 续表顺序与跨页关系；
- 版本、标准或历史适用性风险。

“代表性行”仅用于解释，不能替代原表数字化：有数据价值的完整表体仍应在获准处理的本地证据层保留，未转写或未核对的单元格明确标为未完成。公开贡献不得因此复制受保护的完整表。

不得把相邻表题、正文或下一对象裁入当前证据。

### 曲线、列线图、函数图和选型图

至少给出：

- 横纵轴、单位与尺度；
- 曲线族、图例或参数；
- 趋势和可用范围；
- 如何查值或选型；
- 读数精度与不确定性。

### 工程图与流程图

说明对象、流向/构造、关键标注和工程用途。不能只写“用于设计判断”。

### 软件界面

默认 discard。仅在界面独立承载定量结果、诊断、参数选择或流程决策时保留；纯设置页、文字碎片和装饰界面不进入检索。

## 7. 跨页与组合对象

跨页对象使用有序关系：

```json
{"relation_id":"...","ordered_object_ids":["part-1","part-2"],"pages":[10,11],"physical_merge":false}
```

默认不物理拼图。每个分段保留自己的页码、bbox 和像素证据；关系层表达顺序。

## 8. RAG eligibility

所有资格必须显式布尔化，缺字段时 fail closed：

- `answerable_content`：通过来源与内容验收后，才可能作为直接答案；类别本身不自动授予检索或嵌入资格；
- `supporting_content`：可作关联展开的上下文，但不进入 primary retrieval/embedding；
- `evidence_only`：只做证据；
- `relation_only`、`front_matter`、`back_matter`：不进入 primary retrieval/embedding；
- `no_value`：不检索；
- `discard`：不检索、不嵌入；
- 受保护且未人工批准的内容：不嵌入。

缺失必填资格字段时记录为校验失败并阻止入库，不静默补为 true 或 false；也不得从来源类型猜测允许嵌入。`discard` 是对象处置状态，不是语义包的 `content_class`。

## 9. 知识图谱粒度

KG 只建立：主题根、来源卷、章节/支持单元、粗知识单元、证据节点及其关系。

以下内容留在 RAG/evidence，不建立独立 KG 节点：

- 单个符号、变量和单位；
- 表格行、列、单元格；
- 曲线点与读取值；
- 软件 UI 字段；
- 无独立语义的公式片段。

每个进入知识单元层的 `answerable_content` / `supporting_content` chunk 必须映射到且只映射到一个 coarse KU；每个 KU 必须有证据。支持单元还需关联到同一来源、同一顶层主题下的可回答知识单元。纯 `evidence_only` / `relation_only` / `no_value` 记录保留证据、关系或处置归属，不为满足映射数而虚构知识单元。

## 10. 正式入库包

这是整合目标，不是普通内容 PR 的必交文件清单。维护者可以从小范围内容贡献逐步生成这些文件。

```text
submission/
├── source_manifest.json
├── physical_containers.json
├── semantic_packages.json
├── page_ledger.jsonl
├── text_blocks.jsonl
├── visual_objects.jsonl
├── relations.jsonl
├── rag_chunks.jsonl
├── kg_nodes.jsonl
├── kg_edges.jsonl
├── chunk_to_kg.jsonl
├── evidence_registry.jsonl
└── qa/
    ├── validation.json
    └── independent_audit.json
```

公开 PR 中的 `evidence_registry` 只能包含非敏感 locator、页码和 hash；不得包含本机绝对路径或版权图像。

## 11. 门禁

正式内容包进入下一层投影前必须通过以下检查；普通解释、文档或工具 PR 的合并不替代这些入库检查：

- schema；
- 页覆盖与范围；
- ID 唯一和命名空间；
- 关系/边/映射无孤儿；
- evidence 当前字节 hash；
- eligibility fail-closed；
- 无未知 root、无细粒度 KG 泄漏；
- 无受保护载荷、秘密或本机路径；
- producer 输入零写；
- 不同审阅者的 fresh independent audit；
- blocking findings = 0。

验收结论必须写明来源版本、页段或对象范围、实际核对内容、遗留待核项以及允许进入的下一阶段。局部修复只关闭被检查的具体问题，不能把未修改内容、整章或全书一并标为通过。来源本身有歧义或错误时保留原始证据与明确说明；影响答案可靠性的字段保持待核或排除检索，不擅自改成推测值。

同一内容有可读页、章节正文和结构化记录等多个表示时，修复必须同步其直接关联的表示与哈希，并列明实际改动和未改范围。只改表格元数据而留下错误的正文标题，或只改一份解释而保留另一个可消费旧版本，均不算该问题已关闭。

入库通过仍不等于 production activation。全局重建、向量化、人工评测和发布各自有单独门禁。
