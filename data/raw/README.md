# 原始供应商数据保存区

把 Wind、CSMAR、RESSET、CNRDS、WinGo、朝阳永续等数据库的**原始下载文件**放在这里。

建议为每个来源创建 `SAMPLE` 和 `FULL` 两个子文件夹：

```text
CSMAR/SAMPLE
CSMAR/FULL
WIND/SAMPLE
WIND/FULL
RESSET/SAMPLE
RESSET/FULL
CNRDS/SAMPLE
CNRDS/FULL
WINGO/SAMPLE
WINGO/FULL
GOGOAL/SAMPLE
GOGOAL/FULL
```

规则：

1. 原始 ZIP、CSV、Excel、PDF 字典都保留。
2. 不改原始列名，不在 Excel 中覆盖保存原始 CSV。
3. 每批数据旁边放一个 `__说明.txt`，写明表名、时间、股票范围、字段、单位和筛选条件。
4. 清洗后的标准表才放进 `../inbox`。
5. 具体点击步骤见 [`docs/数据库下载傻瓜式操作手册.md`](../../docs/数据库下载傻瓜式操作手册.md)。
