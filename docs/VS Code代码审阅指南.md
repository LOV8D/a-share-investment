# 用 VS Code 审阅 A share investment

## 从哪里打开

在 VS Code 选择 **File → Open Folder…（文件 → 打开文件夹）**，粘贴下面这个完整路径：

`/Users/watson/Documents/ChatGPT/Investment/A share investment`

必须打开 `A share investment` 这一层，而不是它的上级 `Investment`。打开后左侧文件树顶层应看到 `ashare`、`tests`、`docs`、`config.json`、`compose.yaml` 和 `README.md`。

也可以直接在 Finder 中双击仓库里的 `A share investment.code-workspace`，它已经把项目虚拟环境和 pytest 测试目录配置好。

也可以在 macOS Terminal 执行：

```bash
cd "/Users/watson/Documents/ChatGPT/Investment/A share investment"
code .
```

若提示 `code: command not found`，在 VS Code 按 `Command + Shift + P`，运行 **Shell Command: Install 'code' command in PATH**，再执行上面的命令。

## 建议阅读顺序

1. `README.md`：产品目标、运行命令、数据边界。
2. `config.json`：资金、持股数、仓位、交易成本、调仓频率。
3. `ashare/features.py`：量价、基本面和文本特征怎样在每个时点生成。
4. `ashare/models.py`：生产因子分数，以及 LightGBM/Ridge 挑战者。
5. `ashare/portfolio.py`：选股、仓位约束、整手、费用、次日开盘模拟成交。
6. `ashare/replay.py`：指定日期区间的逐日回放。
7. `ashare/events.py`：LLM 提示词、API 调用和证据校验。
8. `ashare/live_data.py`：每天从公开源采集行情、成分、公告和公司行动。
9. `ashare/pipeline.py`：把数据、因子、模型和组合串起来。
10. `ashare/web.py` 与 `ashare/static/`：网页后端和前端。
11. `tests/`：检查未来信息、成交规则和数据质量的自动测试。

在 VS Code 中按 `Command + P`，输入文件名即可快速打开；按 `Command + Shift + F` 可以搜索整个仓库，例如搜索 `robust_prior`、`make_plan`、`SYSTEM` 或 `available_at`。

这次9月1日至10月15日的回放先看 `docs/模拟盘回放_20260901_20261015.md`；底层逐日输出在 `runs/replay-20261018-201359-321716/`。

## 当前 AI/LLM 状态

产品不是由一个自主交易 Agent 控制。生产订单来自透明的结构化因子和组合规则。LightGBM LambdaRank 与 Ridge 是研究挑战者，当前不参与订单。

`ashare/events.py` 支持两类可选 LLM 接口：

- 本机 Ollama：`AI_MODEL` + 可选 `AI_BASE_URL`。
- OpenAI 兼容的 `/chat/completions` API：`AI_MODEL`、`AI_BASE_URL`、`AI_API_KEY`。

云端目前三个变量都未配置，因此没有调用任何 LLM API，文本收益权重也是0。LLM 只负责从公告/新闻原文抽取事件；它不能直接下单。事件进入特征前还要通过 `published_at`、`first_seen_at`、`processed_at` 三个时点的硬过滤。
