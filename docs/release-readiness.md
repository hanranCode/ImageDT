# 0.6.0rc1 候选版发布就绪报告

本次重构已达到 PyPI **候选版**发布准备状态；尚未上传 PyPI。代码与验证结果将提交候选 PR 审阅。正式稳定版仍应由实际使用者审阅行为变化与模型兼容性范围。

## 已执行验证

| 环境/检查 | 结果 |
|---|---|
| macOS arm64 / Python 3.12.9，完整后端 | 225 tests passed，无 skip |
| macOS arm64 / Python 3.10.19，核心 + plot/excel/pdf/caffe | 181 passed，26 明确 skip（未安装 TensorFlow） |
| macOS arm64 / Python 3.14.7，核心 + plot/excel/pdf/caffe | 181 passed，26 明确 skip（未安装 TensorFlow） |
| 覆盖 | 行覆盖约 93.4%；分支约 83.4%；合计约 91.3% |
| Ruff / Python 编译 / shell 语法 | 通过 |
| 原版本 API 对照 | 原公共函数、类、方法的名称与参数名检查通过；shared implementations 的 re-export 也检查 |
| 真实集成 | Caffe→冻结图→TFLite/LiteRT；隔离 graph 推理；TFRecord 切分、读取、检测标注；PDF 单页和多页 |
| wheel / sdist 构建 | 成功，Python 3 wheel (`py3-none-any`) |
| twine metadata check | wheel 与 sdist 均 PASSED |
| 全新 wheel 安装 + 源码包测试 | 离开仓库目录运行，181 passed、26 明确 skip；确认从独立环境的 site-packages 导入 |
| 分发包完整性 | 源码包包括接口 fixture、迁移说明与测试；wheel 源码可编译 |
| 架构评估报告 schema | 6 findings，通过技能 validator |

生成代码 caffe_pb2.py 与重复的独立 test_lenet.py 未纳入覆盖率分母；前者有 protobuf roundtrip 测试，后者另外直接执行。覆盖率不是所有历史模型或每个错误分支已经验证的保证。

完整后端的 225 项测试已在最后的 BatchNorm/Scale 融合与 TFRecord 重生成修复后重跑。Python 3.10/3.14 核心环境及独立 wheel 安装记录来自这些后端修复之前，且明确跳过 TensorFlow 集成；不作为最后后端修复的独立环境验证证据。

CI 已配置 Linux/macOS/Windows 的 Python 3.10/3.12/3.14 核心矩阵与 Linux Python 3.12 后端集成。远端 CI 尚未执行，因此不声称已验证所有操作系统。

## 本机依赖样本

NumPy 2.5.4、Pillow 12.3.0、OpenCV headless 5.0.0.93、lxml 6.1.3、TensorFlow 2.21.0、protobuf 7.36.2、LiteRT 2.3.0、PDFium 5.14.0、pytest 9.1.1。这是已测试版本样本，不是对所有下界版本组合逐一验证的声明。

## 性能验证

`python scripts/benchmark_utils.py` 使用固定随机数据：5000 样本 × 1001 阈值。新旧 ROC 扫描数值逐项一致。本机一次观测：旧扫描约 0.296 秒，新实现约 0.00349 秒（约 84.6 倍）。这里只比较计算阶段，排除绘图/I/O；结果受机器负载影响，不作为跨平台性能承诺。

Caffe DAG 排序使用迭代算法，2000 节点链验证不触发递归限制。目录扫描/文件对匹配和背景生成采用 `scandir`、集合和向量运算；没有对这些部分声称未测量的速度提升比例。

## PyPI 版本核对

2026-10-12 只读核对 PyPI：现有最新版本是 0.5.0；本次版本改为 0.6.0rc1，查询时该版本尚未发布。发布账号权限未核实，版本可用不等于账号已获授权。

## 发布范围与限制

发布版本为 `0.6.0rc1`，支持 Python 3.10+。已按用户决策保留旧接口并迁移现代后端。Caffe-to-CoreML、缺失资产的 TF1 手写模型、旧 DOT/未校准 uint8 转换明确抛出错误。LeNet 保留函数和 endpoint，但旧 TF-Slim checkpoint 变量名改变。任意真实历史模型或自定义 Caffe 层需要自己的对照 fixture；[迁移说明](migration.md)列明具体行为变化。

文件修改/删除工具保留原有用途。配对重命名预检并在正常可处理失败时回滚，不承诺面对并发写入或进程崩溃时具有文件系统事务。

## 可审阅产物和发布操作

当前产物在 `dist/imagedt-0.6.0rc1-py3-none-any.whl` 与 `dist/imagedt-0.6.0rc1.tar.gz`。`python scripts/check_release.py` 输出完整性检查与 SHA256。`bash upload_pip.sh` 仅构建检查；显式 `--upload` 才上传。本次没有查看 PyPI 凭据或确认账号所有权。

发布者审阅后，可用配置好的 PyPI 账号执行 `PYTHON=python3 bash upload_pip.sh --upload`。上传不可覆盖同一版本，源码与构建检查应和本次批准的代码保持一致。
