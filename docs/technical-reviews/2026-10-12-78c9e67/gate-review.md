# Architecture Gate

由同一执行者进行独立重读和失败路径审查；没有获得额外子代理授权，未派生代理。

| Finding | Outcome | 复核 |
|---|---|---|
| PTR-ENGINEERING-PYTHON2 | ACCEPT | 字符串、字节、整数尺寸的实际回归测试支持结论；不声称 Python 3 语法迁移足以保证旧模型等价。 |
| PTR-ARCH-EAGER-OPTIONALS | ACCEPT | 无 TensorFlow 的独立环境暴露并修复 taffe 的残余 eager 导入。 |
| PTR-DATA-METRIC-CORRECTNESS | ACCEPT | 数学定义、人工小例和数据拆分集合交叉检验；说明指标行为改变。 |
| PTR-RELIABILITY-RESOURCE-LIFETIME | ACCEPT | worker 完成、独立 graph、rollback 故障注入；降级跨进程事务保证。 |
| PTR-INTEGRATION-REMOVED-BACKENDS | REVISE | 真实小模型只验证受支持子集；CoreML/手写模型不能宣称迁移成功，改为明确失败并记录 checkpoint 限制。 |
| PTR-PERFORMANCE-NESTED-SCANS | ACCEPT | 可复现算法比较验证数值，时间仅为本机结果；不推断线上吞吐。 |

跨平台 CI 和真实历史模型 equivalence 没有执行证据，明确列为证据限制；未引入新的服务、调度或分布式组件。
