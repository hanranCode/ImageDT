# 全项目技术评估与 Python 3 重构

## Executive Summary

ImageDT 是约 1 万行的历史工具包，其中约一半为生成的 Caffe protobuf 代码。核心价值是文件/图像工具、标注、指标、文本生成和轻量模型链路，而不是现成的训练平台。问题集中在旧运行时、入口耦合和缺少正确性测试。已采用保留历史公共名称的增量迁移方案，不引入新架构框架。

用户已决策：保留旧入口，迁移现代后端，无法兼容的功能明确报错。结果与限制见 [迁移指南](../../migration.md)。

## Decision Summary

基础代码已迁移 Python 3，隔离可选依赖，补齐自动测试和构建；Caffe protobuf 改为 descriptor pool，保留 schema；采用 TF compatibility、Keras、LiteRT 和 PDFium。不存在的 CoreML Caffe 和未提供资产的手写模型入口抛出明确错误。LeNet checkpoint 名称不能保证兼容。

## Prioritized Roadmap

| 顺序 | 关联 finding | 负责人类型 | 结果与验证 |
|---|---|---|---|
| 1 已实施 | PTR-ENGINEERING-PYTHON2, PTR-ARCH-EAGER-OPTIONALS | 包维护者 | 最低/最新 Python 安装验证、模块/API 契约 |
| 2 已实施 | PTR-DATA-METRIC-CORRECTNESS, PTR-RELIABILITY-RESOURCE-LIFETIME | 包维护者 | 边界回归、文件故障注入与合成模型链路 |
| 3 已实施 | PTR-PERFORMANCE-NESTED-SCANS | 包维护者 | 可复现 ROC 数值/性能比较与深图排序测试 |
| 4 候选发布 | PTR-INTEGRATION-REMOVED-BACKENDS | 发布者 | 审阅迁移限制，wheel/sdist 构建、metadata 检查、全新安装 |
| 5 实际采用时 | PTR-INTEGRATION-REMOVED-BACKENDS | 模型使用者 | 增加真实历史模型输入/输出 fixture；原环境作为对照，可回退旧包版本 |

## Scope, Context, and Evidence Limitations

审阅整个仓库：运行组件、打包、依赖、测试和发布脚本。旧 Git revision 与当前工作区分开记录。没有生产数据、用户使用频率、实际旧模型、业务 SLA 或性能遥测。跨平台配置不是跨平台成功记录。PyPI 所有权和账号凭据未查询，未上传。

## Project Profile and Capability Map

见 [架构模型](architecture-model.md)。它是进程内 Python 库，无已实现的服务或数据存储。caffe/torch 的 trainer 为空，不把 TODO 计为已实现能力。

## Current Architecture and Architecture Drift

原星号导入使依赖边界消失；当前入口是按需导出，内部执行仍为普通函数和轻量对象。文件和图像功能共享目录扫描；两份 TFRecord 实现已合并。没有微服务拆分的现实依据。

## Architecture Scorecard

| 维度 | 基线→当前 | 置信度与证据 |
|---|---|---|
| 工程基础 | 1→4 | HIGH：版本矩阵、可重复 fixtures、构建元数据 |
| 模块边界 | 2→4 | HIGH：懒加载、extras、独立核心安装 |
| 正确性与可靠性 | 1→4 | HIGH：错误结果修复与回归；真实历史模型仍有盲区 |
| 性能 | 2→4（被测工具链） | HIGH：ROC 等价比较；生产吞吐 INSUFFICIENT EVIDENCE |
| 安全性 | 2→3（输入/文件边界） | MEDIUM：XML entity 关闭、生成源码名称约束；不作为全面依赖漏洞审计结论 |
| 部署/运维 SLA、成本 | NOT REVIEWED / N/A | 仅库；缺少生产记录 |

## Root Cause Map

长期缺少运行时与接口契约 → Python 2 和过时 backend 绑定 → 导入/运行失败；缺少自包含正确性 fixtures → 指标错误、数据泄漏和生命周期缺陷无法自动发现。以接口基线、依赖隔离和临时数据测试作为共同补救，避免只做格式化。

## Technical Debt Map

代码债：重复序列化、循环扫描、共享默认值；结构债：跨层 eager 导入；过程债：个人目录测试、缺少发行检查；集成债：删除的上游 API 和未提供模型资产。前三项已处理；最后一项通过明确支持范围和失败契约控制，不能宣称恢复不存在的能力。

## Evolution Constraints

任意 Caffe 自定义层、旧 TF-Slim checkpoint 和量化校准都需要具体工作负载 fixture；没有资产就不能保证恢复。保持模块接口比引入新插件系统更符合当前规模。

## Product / Platform Opportunities

现有 XML/图像/TFRecord 原语可形成更一致的离线数据处理工具，但当前没有需求证据，未扩展为训练平台或代理服务。

## Detailed Findings

完整结构化证据与建议见 [findings.json](findings.json)。

### PTR-ENGINEERING-PYTHON2 — Python 2 运行语义与陈旧依赖阻碍安装

P1 / HIGH / RESOLVED。unicode 与老版本 NumPy/protobuf，安装元数据没有依赖。根因：多年未维护，缺少 Python 版本及依赖契约。处理：迁移 Python 3、统一现代元数据并设置版本测试矩阵。取舍：停止 Python 2；最低支持 Python 3.10。验证：Python 3.10/3.12/3.14 安装与临时数据回归。

### PTR-ARCH-EAGER-OPTIONALS — 通用入口耦合重型可选依赖

P1 / HIGH / RESOLVED。多层星号导入加载图像、绘图、Excel 和 TensorFlow。根因：模块边界通过 eager star import 连接，没有可选能力分层。处理：保留历史名称，按需导入并拆分 extras。取舍：第一次调用时出现可选依赖错误，而非导入包时。验证：独立 subprocess 检查顶层不加载框架；无 TensorFlow 的核心环境运行。

### PTR-DATA-METRIC-CORRECTNESS — 指标与数据切分存在正确性错误

P1 / HIGH / RESOLVED。固定类别、未匹配误计、按行配对、表头混排、训练包含验证样本。根因：缺少边界契约和独立的正确性 fixtures。处理：明确评估契约并修复匹配/对齐/切分。取舍：指标数值会因修复而改变。验证：完美/重复/错误类别/空框评估；训练验证标签集合不重叠。

### PTR-RELIABILITY-RESOURCE-LIFETIME — 清理及模型资源生命周期不完整

P1 / HIGH / RESOLVED。局部函数作为进程目标、分片遗漏、未 join、默认图污染、池没有关闭契约。根因：并发与资源所有权没有显式生命周期。处理：线程池等待、隔离图和 close/context-manager，配对重命名预检/回滚。取舍：同步等待完成；配对回滚不提供跨进程事务。验证：多余尾文件、两个独立模型、重命名故障注入和复用生成器。

### PTR-INTEGRATION-REMOVED-BACKENDS — 旧框架私有接口已无法直接维护

P1 / HIGH / IMPROVED。tensorflow.contrib/TOCO 与 CoreML Caffe 旧后端依赖。根因：直接绑定上游历史私有实现。处理：迁移公开兼容 API、LiteRT 和 Keras；不存在的能力保留入口明确失败。取舍：TF-Slim 检查点变量名改变；部分功能不再实现。验证：合成 Caffe→GraphDef→TFLite 链路；错误分支；用户已选择现代后端方案。

### PTR-PERFORMANCE-NESTED-SCANS — ROC 与图遍历存在可消除的扫描成本

P2 / HIGH / RESOLVED。ROC 阈值样本嵌套循环，递归排序头插列表。根因：旧实现没有利用排序索引和线性图算法。处理：searchsorted 阈值查询与迭代 DAG 排序，向量化背景。取舍：需要数组排序；多个合法拓扑序的顺序可能不同。验证：5000 样本/1001 阈值与旧算法逐项一致；2000 节点图无递归。

## Measurements and Access Required

最终本机验证记录见 [发布就绪报告](../../release-readiness.md)。性能样本、源码版本和完整测试结果应一起解释。若用户要保证旧模型结果，需要提供可分发的模型、样例输入和旧环境输出；若要正式上传需要发布者的 PyPI 授权。未请求或读取账号密钥。

## Architecture Gate Summary

见 [gate-review.md](gate-review.md)。高风险项按具体失败场景复核，删除“所有旧模型已兼容”一类没有证据的结论。当前是候选版发布准备与明确的能力边界，仍需发布者审核实际迁移约束。
