"""Public, evidence-based AI scoring rubric; no request for hidden reasoning."""
RUBRIC = '''按以下可审计评估表处理原文，不输出内部思维链：
1. fact_summary：一句话概括原文事实，禁止把计划当已发生结果；
2. surprise：相对原文明示的预期/指引的意外程度，-1至1；无可核验预期时填null，不能凭记忆编造；
3. novelty：0至1，是否新增实质信息，无法核实旧闻时保守；
4. affected_business：影响公司的哪项业务，不能捏造供应链关系；
5. horizon_days：1至30，预计事件影响持续天数；
6. priced_in：仅在输入提供决策前价格反应证据时评估，否则填null；
7. risk_flags：原文支持的风险列表；
8. sentiment/materiality/confidence：方向-1至1、重要性0至1、置信度0至1。
没有预期或价格证据不会被当成超预期/尚未反应。evidence必须逐字引用输入原文连续8至240字。
只输出事实摘要、字段评分和证据；不输出推演过程。'''
