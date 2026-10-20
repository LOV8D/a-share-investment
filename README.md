# A share investment

沪深300日频量化与AI研究产品。当前模拟资金 **1,000,000元**，每天收盘后评分，取最终分数前10名，并为下一交易日开盘生成条件交易计划。

第一次使用请先看 [网站使用说明与需求对照](docs/网站使用说明与需求对照.md)：它说明每晚怎样运行、六个页面怎么看，以及目前实现和你原始要求的对应关系。

腾讯云部署方式见 [腾讯云部署操作手册](docs/腾讯云部署.md)。服务器地址、登录凭据、API密钥和部署环境文件属于本机私密配置，不进入 Git 仓库。

**用户提供的 CSMAR 与 RESSET 数据已经完成处理和严格校验：日行情、点时财务、完整历史沪深300成分、价格/全收益基准、公司行动和历史税费均已进入正式回测集。历史结构化研究不再缺必需数据。**

最新真实数据研究运行 `20260920-210329-586070` 覆盖 2010-05-31 至 2026-08-31。扣除模拟成本后年化收益 6.45%，沪深300全收益同期年化 5.70%；最大回撤 26.9%，基准 46.1%。这段历史已经参与开发和规则选择，结果不能当成未见样本或实盘承诺；模型卡仍明确标记为未通过实盘验收。

## 打开产品

双击本目录 `启动工作台.command`，然后打开 <http://127.0.0.1:8765>。在已安装的环境中：

```bash
cd "A share investment"
.venv/bin/python -m ashare serve --port 8765
```

界面包含收盘决策、次日计划、因子与回测、AI事件、数据中心、策略参数。所有操作和数据在本机，未连接券商。

日常模拟盘已能自动采集公开增量数据，不再要求手工补行情。数据源、质量闸门和失败处理见 [自动数据采集策略](docs/自动数据采集策略.md)。历史新闻、政策和论坛不回测；模拟盘开始后按首次抓取时间接入。

## 功能与架构

`数据导入 → 时间点校验 → 核心因子 + WorldQuant 101 → GLM事件后融合 → 日度前十组合 → 次日开盘条件订单 → 结果与证据存档`

- 83个可用 WorldQuant 101 因子全部进入生产结构化分数；另18个因缺少逐日 sector/industry/subindustry 三层分类而明确停用。核心价值、质量、低风险因子与 WorldQuant 组各占50%。
- 回测只用当时可知的结构化数据，训练仅用已成熟标签，时间隔离与滚动样本外预测。
- 100万元模拟账户默认理论前10只、单股10%、行业30%、日度调仓、持仓缓冲0；同时处理整手、板块申报数量、ADV、现金、历史费用、T+1与限价约束。
- GLM只从当时可见原文抽取事件。有合格事件时按20%上限与结构化分数后融合；无事件时实际文本权重为0。三重时间闸门禁止未来信息和事后回填。
- 交易计划为条件计划，不是已成交订单，不是自动券商下单。实际数据、模型和策略都未完成实盘验收。

详细设计和限制见 [研究设计与边界](docs/研究设计与边界.md) 与 [WorldQuant101因子审计与融合规则](docs/WorldQuant101因子审计与融合规则.md)。

## 常用命令

以下均在项目目录执行。

```bash
# 重建合成演示，完整训练、回测并保存结果
.venv/bin/python -m ashare demo

# 导出标准CSV表头
.venv/bin/python -m ashare templates

# 真实数据字段/时间/基础质量校验
.venv/bin/python -m ashare validate --data data/inbox

# 真实数据滚动研究
.venv/bin/python -m ashare research --data data/inbox --capital 100000

# 收盘后生成最新一天计划（不会重复完整回测）
.venv/bin/python -m ashare daily

# 自动采集公开增量数据，但暂不生成计划
.venv/bin/python -m ashare collect-live

# 完整模拟盘作业：采集、质量闸门、账户推进、评分、计划和第二源复核
.venv/bin/python -m ashare paper

# 按当时可知数据回放指定区间；不倒灌事后文本
.venv/bin/python -m ashare replay --data data/live/current --start 2026-09-01 --end 2026-10-15

# 有限预算因子挖掘：32个安全表达式，仅在训练区间内发现/验证
.venv/bin/python -m ashare mine --data data/inbox --budget 32

# AI只提出安全代数表达式，先保存，再做独立的训练区间验证
AI_MODEL='your-installed-model' .venv/bin/python -m ashare propose-factors --output runs/proposals.json
.venv/bin/python -m ashare mine --data data/inbox --proposals runs/proposals.json

# 原文抽取：使用你实际安装的本地Ollama模型名称
AI_MODEL='your-installed-model' .venv/bin/python -m ashare extract --data data/inbox --provider ollama

# 已知RSS地址，只取一个有明确证券关联的允许抓取源
.venv/bin/python -m ashare fetch-rss --url 'https://your-authorized-feed.example/rss' --symbol 600000.SH --output data/inbox/documents.csv

# 用当天竞价数据和账户可卖数量做预检；RUN_ID换为实际运行目录名
.venv/bin/python -m ashare precheck --run RUN_ID --data data/inbox

# 核心测试
.venv/bin/python -m pytest -q
```

兼容远端接口使用 `AI_BASE_URL`（包括供应商约定的/v1前缀）、`AI_API_KEY`、`AI_MODEL` 环境变量，选择 `--provider compatible`。不把密钥写入配置或上传到仓库。只有执行extract命令才发送原文；请使用允许处理所选数据的服务。

## 数据导入

当前 CSMAR 批次可用下面命令从 `Downloads` 重新生成 staging 数据和质量报告：

```bash
.venv/bin/python scripts/process_csmar_downloads.py --batch 20260920
```

该脚本不修改下载原件；它保存 SHA-256、统一代码与单位、计算 TTM 基本面，并明确标记仍缺的点时数据。

RESSET 指数行情、指数成分候选和 CSMAR 公司行动用下面命令处理：

```bash
.venv/bin/python scripts/process_remaining_downloads.py --batch 20260920
```

该脚本已产出 `data/staging/real`：4,291 行基准指数、公司行动、交易日历、历史费用和通过完整性检查的历史成分。随后运行下面的组装脚本生成正式 `data/inbox`：

```bash
.venv/bin/python scripts/assemble_real_dataset.py
```

正式数据为 2,525,434 行价格、1,286,994 行逐日历史成分、50,513 行点时基本面和 10,977 个股票池内公司行动；严格校验 0 个错误。

界面支持按表导入CSV/Parquet。大文件可直接放入 `data/inbox`。表头在 `data/templates`，明确定义每个字段的单位、时间和主键。列名不同可用显式映射：

```bash
.venv/bin/python -m ashare import --source raw-export.csv --table prices --mapping docs/column_mapping.example.json --dest data/inbox
```

示例映射只有结构示范，并不是已核验的Wind/CSMAR真实字段ID。必须按数据字典修改。

有持仓时需同时提供 `holdings.csv` 与 `account.json`（现金、收盘日期）。没有账户快照时按 `config.json` 的 `initial_cash` 以全现金启动，不读取任何真实券商账户。

## Git仓库中的数据边界

仓库只保存代码、测试、配置模板和研究说明，不包含学校授权数据库、原始下载文件、加工后的Parquet、模拟账户状态或运行结果。克隆后需要自行把有权使用的数据放入 `data/inbox`，或使用公开增量采集流程启动新的前向模拟。

以下目录只保留说明文件或空目录占位符：

- `data/raw`：原始数据库导出，禁止提交。
- `data/staging`：中间加工数据，禁止提交。
- `data/inbox`：正式历史研究数据，禁止提交。
- `data/live`：前向模拟数据与账户状态，禁止提交。
- `runs`：模型、报告、订单和日志，禁止提交。

从 `.env.example` 创建本地 `.env`，真实API密钥不得写入版本控制。

## 日常运行

完整模拟盘命令会自行采集和校验，不再要求手工导出日行情。每个交易日更新评分并产生次日开盘计划。没有新交易日时不会重复计划；行情、股票池或交易日历未通过闸门时不发布新数据。GLM只处理新抓到且通过时间闸门的文本。

最新容器版本在北京时间每天00:00尝试一次，处理最近一个已结束交易日；成功或失败后当天都不自动重试。部署步骤见 [腾讯云部署操作手册](docs/腾讯云部署.md)。

## 输出

`runs/<运行编号>/` 包含：`report.json`、`plan.json`、`orders.csv`、`rankings.csv`、`equity_curve.csv`、`executions.csv`、`daily_ic.csv`、`oos_predictions.parquet`、`manifest.json`、配置与模型文件。每日计划运行不会生成整段回测文件。

latest.json仅指向最近一次成功结果。合成与真实输入目录分离；每次运行保留模式标记和输入哈希。

## 在另一台机器安装

Python 3.11+。推荐使用uv：

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/python scripts/fix_macos_openmp.py
.venv/bin/python -m ashare demo
```

macOS脚本仅给项目内LightGBM添加scikit-learn内置OpenMP查找路径，不修改系统Homebrew。Windows/Linux无需此修复。依赖锁文件记录此次已验证环境的版本；跨系统需对应的官方wheel。
