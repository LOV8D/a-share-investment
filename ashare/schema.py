"""Canonical import contracts. Currency CNY, volume shares, ratios decimals."""
SCHEMAS = {
    "calendar": ["date"],
    "prices": ["date", "symbol", "open", "high", "low", "close", "prev_close", "volume", "amount", "adj_factor", "upper_limit", "lower_limit", "suspended", "is_st", "market_cap", "float_market_cap", "turnover_rate", "industry", "min_order", "lot_step", "listed_days", "available_at"],
    "universe": ["date", "symbol", "weight", "known_at"],
    "fundamentals": ["symbol", "report_period", "available_at", "revenue_ttm", "net_profit_ttm", "book_equity", "operating_cashflow_ttm", "total_assets", "total_liabilities", "roe", "gross_margin", "revenue_growth", "profit_growth"],
    "benchmark": ["date", "close", "total_return_close"],
    "actions": ["date", "symbol", "share_multiplier", "cash_per_share", "known_at"],
    "documents": ["document_id", "symbol", "source_type", "source_url", "published_at", "first_seen_at", "title", "text"],
    "events": ["event_id", "document_id", "symbol", "category", "sentiment", "materiality", "confidence", "novelty", "source_type", "source_url", "published_at", "first_seen_at", "processed_at", "title", "evidence", "model", "prompt_version", "text_sha256"],
    "holdings": ["symbol", "shares", "sellable_shares", "cost_price"],
    "auction": ["date", "symbol", "indicative_open", "upper_limit", "lower_limit", "suspended", "available_at"],
    "fees": ["start_date", "end_date", "sell_stamp_rate", "transfer_rate"]
}
REQUIRED = ["calendar", "prices", "universe", "fundamentals", "benchmark", "actions", "fees"]
DESCRIPTIONS = {
    "calendar": "沪深交易日历，包含至少未来 10 个交易日",
    "prices": "未复权日线、当日复权因子、历史交易状态和证券属性",
    "universe": "逐日历史沪深300成分及权重，禁止用今天成分回填",
    "fundamentals": "按实际可用时间保存的财务指标历史版本",
    "benchmark": "沪深300价格指数及全收益指数",
    "actions": "除权除息事件；无事件也提供带表头的空文件",
    "documents": "模拟盘实时抓取的公告、政策、新闻、论坛原文与首次抓取时间",
    "events": "模拟盘前向观察用、证据可追溯的 AI 事件标签",
    "holdings": "信号日收盘账户持仓；模拟时默认为全现金",
    "auction": "次日 09:25 集合竞价结果或指示价格，用于预检",
    "fees": "分历史区间的卖出印花税和双边过户费"
}
