# Chemical Engineering RAG / KG

面向独立项目 `chemical-engineering-rag-kg` 的化工知识检索与粗粒度知识图谱公开预览。公开内容以有实际用途、可完整复述并带必要条件的工程释义为核心，不把整本资料或无信息增量内容塞入检索。

## 统一查询已接纳内容

现在可以一次查询两批已接纳内容：**5 个来源卷、98 个完整主题、113×512 维已有真实语义向量**。入口为 [`query_collection.py`](src/query_collection.py)，批次范围由[明确清单](knowledge/curated-collection-v1.json)锁定；不是自动扫描历史八源库或外部 Skills 项目。

```powershell
python src/query_collection.py --query "换热器压降和泵的工作点怎样一起检查？" --method lexical
python src/query_collection.py --query "固定床热点如何限制放大尺度？" --method hybrid --model-dir ./local-model --lock contracts/curated_model_lock.json
```

两包的知识、图谱和向量文件保留原样，不复制成第三份库。词法检索在筛选后的全体分段上计算统一统计，语义检索只编码一次查询后统一排序，混合检索在全局主题排名上融合。结果带所属批次和完整 KU；`--source`、`--subject` 可限定范围，`--node-id` 可精确读取。候选命中不等于工程答案，当前仍没有自动无答案阈值。

32 个公开开发题在**不筛选来源或学科**的统一范围下，实际执行了 96 次查询，并核对返回 KU 的完整性：

| 方法 | 前 5 命中率 | 首位命中率 | MRR@5 |
|---|---:|---:|---:|
| 词法 lexical | 100% | 96.875% | 0.984375 |
| 语义 dense | 100% | 96.875% | 0.979167 |
| 混合 hybrid | 100% | 100% | 1.000000 |

详见[逐题结果和运行依据](reports/collection-dev-evaluation.json)。词法与语义各有一题目标未排首位；混合检索在本开发集全部首位命中。上述结果不是封存测试、拒答能力验收或普遍正确性证明，也不表示全书数字化已经完成。

如需复跑，准备锁定模型及下文列出的查询依赖，将 `./local-model` 与 `./local-vendor` 换成自己的模型和依赖目录；复跑结果另存，不覆盖发布报告：

```powershell
python src/evaluate_collection.py `
  --collection knowledge/curated-collection-v1.json --knowledge-root knowledge `
  --query-module src --four-queries examples/curated-dev-queries.jsonl `
  --s001-queries examples/s001-public-development-queries.jsonl `
  --model-dir ./local-model --lock contracts/curated_model_lock.json `
  --vendor ./local-vendor --output ./local-collection-evaluation.json
```

## 新增：化工原理上册选题包

[`curated-s001-upper-v1`](knowledge/curated-s001-upper-v1/) 收录 S001《化工原理 上》（王瑶、贺高红主编）的 **29 个完整主题**，覆盖流体输送、机械分离及流态化、传热与换热器、蒸发四章的选定内容。来源共 371 页，这不表示 371 页均已完成数字化。

本批为 1 个来源卷、4 个章节节点、64 个图节点、63 条内部边。17 条指向四源包的关联存入 `cross_bundle_references.jsonl`，绑定目标 KU 及正文哈希；95 条 Skills 关联仍属于外部项目，不复制关联对象的正文。它是新增批次，首批四源包的内容与向量保持原样。

已生成 **31×512 维真实 BGE 语义向量**，完整覆盖 29 个 KU；只有两个长主题作索引分段，正文不截断。批次内容、证据、图谱与向量均放在同一包中；原始 PDF、页图和 OCR 全文不公开。

16 个公开开发正例的三种检索方式均为 hit@5=1.0；lexical/hybrid 的首位命中率为 1.0，dense 为 0.9375（有一题排第 3），详见[本批评测](reports/s001-dev-evaluation.json)。另有 4 个库外诊断题，当前接口仍会返回候选，不声称自动拒答通过。[向量复核](reports/s001-vector-audit.json)记录全部 31 行真实模型重编码逐字节一致。以上均不是全书验收或普遍可靠性证明。

如只查本批，使用原单包入口并指定 `--bundle knowledge/curated-s001-upper-v1`。`query_curated.py` 的默认仍是四源包；统一查询使用上面的 `query_collection.py`。跨包关联本身只表示补读关系，不自动构成合并检索或工程推理。

```powershell
python src/query_curated.py --bundle knowledge/curated-s001-upper-v1 --query "多效蒸发的进料方向怎样比较？" --method lexical
python scripts/validate_curated_bundle.py --bundle knowledge/curated-s001-upper-v1 --bundle-id curated-s001-upper-v1 --related-bundle knowledge/curated-four-books-v1 --require-vectors
```

## 四源公开预览

本批包含 4 个来源卷、69 个粗知识单元（KU）：

| 来源 | 资料 | 版本/版次 | 选定 KU |
|---|---|---|---:|
| RE01 | 《化学反应工程》 | 第三版 | 14 |
| OC02 | 《化工过程的优化设计与控制》 | 版次未核 | 13 |
| TH03 | 《化工热力学》 | 第二版 | 25 |
| EN04 | 《过程工业能量系统优化——换热网络与蒸汽动力系统》 | 第一版，第一次印刷 | 17 |
| 合计 | **4 个来源卷** | **2 个主题根** | **69** |

预览图谱包含 2 个主题根、4 个来源卷、29 个章节节点、4 个跨章节支持单元、177 个节点和 191 条内部边。外部 `related_to` 关系另存为 252 条 `external_references`，不混入内部图边。

主题根只有：

- `chemical-engineering-principles`（化工原理）
- `reaction-engineering`（反应工程）

### 选择口径

进入预览的内容必须对实际计算、设计、选型、排错或关键概念解释有独立用途，并能连同必要条件被读者无歧义复述。重复、纯导航、无信息增量和仅作证据的内容不另造 KU；知识单元按完整可复述的粗粒度组织，不设微粒度数量配额，也不把符号、单位、表格单元格或曲线点拆成图节点。

KU 命中后应完整阅读标题、正文、适用范围、单位/基准、来源定位和证据。`related_to` 只表示有关联，不自动表示因果、计算先后或可以互相替代。实际项目数值、经验关联式、设备曲线和材料限值必须核实适用条件，并回到当前项目及可合法核对的来源确认；教材示例值不能直接当作项目输入。

## 四源包入口

公开包入口为 [`knowledge/curated-four-books-v1/`](knowledge/curated-four-books-v1/)，关键文件如下：

- [`source_manifest.json`](knowledge/curated-four-books-v1/source_manifest.json)：来源书目、来源卷、主题根、审核证明和发布边界；
- [`knowledge_units.jsonl`](knowledge/curated-four-books-v1/knowledge_units.jsonl)：69 条完整 KU；
- [`evidence_registry.jsonl`](knowledge/curated-four-books-v1/evidence_registry.jsonl)：来源页范围、定位和正文哈希；
- [`kg_nodes.jsonl`](knowledge/curated-four-books-v1/kg_nodes.jsonl)、[`kg_edges.jsonl`](knowledge/curated-four-books-v1/kg_edges.jsonl)：主题根—来源卷—章节/支持单元—KU—证据的粗图；
- [`external_references.jsonl`](knowledge/curated-four-books-v1/external_references.jsonl)：指向外部 Skills 项目的关联，不创建伪内部节点；
- [`conversion_report.json`](knowledge/curated-four-books-v1/conversion_report.json)：确定性转换统计和输入/产物哈希。

预览包的语义向量随包公开；已生成 **82 个检索分段、82×512 维 BGE 语义向量**，完整覆盖 69 个 KU；13 个长知识块仅在索引层分段，不截断正文。向量预览遵循当前合同和模型锁，不公开模型权重。原始 PDF、扫描页、裁图和 OCR 全文不发布；公开内容限于获准的原创释义、书目/定位元数据、粗图及经校验的派生向量预览。

向量与验证文件：[`rag_chunks.jsonl`](knowledge/curated-four-books-v1/rag_chunks.jsonl)、[`chunk_to_kg.jsonl`](knowledge/curated-four-books-v1/chunk_to_kg.jsonl)、[`embeddings.f32`](knowledge/curated-four-books-v1/embeddings.f32)、[`vector_manifest.json`](knowledge/curated-four-books-v1/vector_manifest.json)、[发布清单](knowledge/curated-four-books-v1/manifest.json)。向量不是哈希向量；模型为锁定的 `BAAI/bge-small-zh-v1.5`，权重不随库发布。

16 个公开开发题中，lexical、dense、hybrid 的 hit@5、MRR@5 与首位命中率均为 1.0，见[实际评测报告](reports/curated-dev-evaluation.json)。这是公开正例集的开发检查，不是封存测试集，不验证无答案拒答能力或普遍正确性。[独立校验报告](reports/curated-independent-validation.json)记录来源身份、关系与向量字节复核；代理审阅和人类审核分开记录。

## 读取、验证与查询

建议先阅读 [READING_GUIDE.md](docs/READING_GUIDE.md)，再按 [INGESTION_SPEC.md](docs/INGESTION_SPEC.md) 判断内容用途、证据和资格。选题知识合同见 [`contracts/curated_knowledge_contract.json`](contracts/curated_knowledge_contract.json)。资格字段必须显式给出；缺失字段不推断放行。代理审阅不写作人类逐条审核，检索命中也不是工程认证答案。

在仓库根目录执行。结构校验仅需 Python 3.12+ 标准库；查询另需 NumPy，语义检索还需模型锁指定的 tokenizers / ONNX Runtime。当前已验证版本分别为 NumPy 2.5.1、tokenizers 0.23.1、ONNX Runtime 1.27.0。

公开结构及向量字节验证：

```powershell
python scripts/validate_curated_bundle.py `
  --bundle knowledge/curated-four-books-v1 `
  --require-vectors
python scripts/validate_public_release.py
```

公开预览的 lexical 查询：

```powershell
python src/query_curated.py `
  --bundle knowledge/curated-four-books-v1 `
  --query "换热网络中的夹点目标" `
  --method lexical `
  --limit 5
```

lexical 不加载模型；dense/hybrid 需要本地具备[模型锁](contracts/curated_model_lock.json)列出的六个文件，脚本核对哈希，不自动下载或换模型。把下面 `./local-model` 替换为你的模型目录：

```powershell
python src/query_curated.py --query "固定床热点如何限制放大尺度？" --method hybrid --model-dir ./local-model --lock contracts/curated_model_lock.json
```

输出按 KU 去重，携带完整正文、条件、单位和来源；不是只返回命中的片段。

维护者使用的源记录转换器位于 `src/build_curated_bundle.py`，其输入参数为 `--mapping`、`--source-audit`、`--records-root` 和 `--output-dir`；这些父级审阅输入不要求公众取得，也不写入公开包。

## 状态边界

四个来源卷的 `full_source_digitization_complete=false`；`human_review=false`、`human_approved_anchor_count=0` 和 `production_activated=false` 保持不变。正式入库、独立审计、向量评测和生产激活分别有门禁，不能由文档、格式转换或一次检索命中互相替代。

旧八源阶段快照中的 3,977 页、2,326 条私有向量和 6 项未通过门槛只作为历史数据保留；它们不与本批 69 KU、177 节点或 191 条边相加，也不构成四源预览的当前准入结论。详细历史状态见 [来源与提取状态](docs/SOURCE_STATUS.md)。

本包不新增许可证或来源授权；既有合同和第三方边界继续适用。
