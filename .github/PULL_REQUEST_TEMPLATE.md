## 范围

- 提交层级：内容解释/纠错 / 正式入库包 / 工具或文档（不适用项可删）
- 来源/章节：
- 变更类型：新投影 / scoped repair / schema / validator / docs
- 实际变更 allowlist：
- 明确未改范围：

## 来源与权限

- source_id：
- PDF 页数与 SHA-256：
- 权限状态：
- 本 PR 是否包含 PDF、图片、OCR、原文、向量、数据库或本机路径：否

## 验证

内容解释/纠错只需完成前四项；完整入库检查由维护者在整合时补齐。工具和文档 PR 对不适用项填写 N/A 及原因。

- [ ] 已标明来源、页码及本次对象范围
- [ ] 已阅读原页和必要上下文，解释清楚且未猜测未知项
- [ ] 已说明未核对和未覆盖的部分
- [ ] 无受保护原文载荷、秘密或本机路径

正式入库包检查：

- [ ] 页账本 exactly once，无缺页/重叠
- [ ] physical containers 与 semantic packages 分离
- [ ] 公式/表格/曲线/工程图解释符合入库规范
- [ ] eligibility 显式且 fail closed
- [ ] KG 保持粗粒度
- [ ] schema、ID、关系、hash、零写检查通过
- [ ] activation/production/publish flags 保持 false
- [ ] 已由不同人员/代理完成 fresh independent audit

## Findings 与限制

- blocking findings：
- nonblocking debts：
- 已知未覆盖范围：
