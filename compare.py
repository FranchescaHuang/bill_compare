import os
import sys
import asyncio
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from llama_index.core.agent import FunctionAgent
from llama_index.llms.deepseek import DeepSeek
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.tools.mcp import McpToolSpec, BasicMCPClient

# 1. 环境与模型配置
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("未检测到 API Key。请在 .env 文件中设置 DEEPSEEK_API_KEY。")

llm = DeepSeek(
    model="deepseek-chat",
    api_key=api_key,
    temperature=0
)


# 自定义Logger类同步输出到终端和文件，保留完整执行痕迹
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# 异步主函数架构
async def main():
    # 生成带时间戳的文件名（精确到分）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = f"comparison_log_{timestamp}.txt"

    # 重定向 stdout
    sys.stdout = Logger(log_filename)

    try:
        # MCP 服务集成
        # 2. 连接到 MCP Server (Model Context Protocol)
        # BasicMCPClient 会通过 stdio 与 my_finance_mcp_server.py 进行交互
        mcp_client = BasicMCPClient(
            command_or_url=sys.executable,
            args=["my_finance_mcp_server.py"]
        )

        # 3. 创建 McpToolSpec 并获取工具列表
        # 这会将 MCP Server 中定义的 tool (如 get_fee_description) 转换为 LlamaIndex 可用的工具
        mcp_spec = McpToolSpec(client=mcp_client)
        mcp_tools = await mcp_spec.to_tool_list_async()

        # 4. 准备本地数据与工具 (PandasQueryEngine)
        # internal_df: 公司内部系统的数据集创建查询引擎
        internal_df = pd.DataFrame({
            "trans_id": ["T101", "T102", "T103"],
            "date": ["2023-10-01", "2023-10-02", "2023-10-03"],
            "amount": [150.00, 200.00, 3000.00],
            "desc": ["Starbucks Coffee", "Apple Store", "Business Flight"]
        })

        # bank_df: 银行导出的对账单
        bank_df = pd.DataFrame({
            "bank_ref": ["B_REF_01", "B_REF_02", "B_REF_03"],
            "bank_date": ["2023-10-01", "2023-10-02", "2023-10-04"],
            "bank_amount": [150.00, 200.00, 3090.00],  # 3000 + 3% 跨境手续费
            "bank_desc": ["SBUX #4829 SEATTLE", "APPLE.COM/BILL", "AIRLINE TICKETS"]
        })

        # 创建 Pandas 查询引擎，自然语言查询：PandasQueryEngine让LLM能用自然语言查询结构化数据
        internal_engine = PandasQueryEngine(df=internal_df, llm=llm, verbose=True)
        bank_engine = PandasQueryEngine(df=bank_df, llm=llm, verbose=True)
        # 工具封装：通过QueryEngineTool添加元数据描述，帮助LLM理解数据结构
        local_tools = [
            QueryEngineTool(
                query_engine=internal_engine,
                metadata=ToolMetadata(
                    name="internal_records",
                    description="查询公司内部系统的交易记录，包含 trans_id, date, amount, desc"
                ),
            ),
            QueryEngineTool(
                query_engine=bank_engine,
                metadata=ToolMetadata(
                    name="bank_statement",
                    description="查询银行导出的账单明细，包含 bank_ref, bank_date, bank_amount, bank_desc"
                ),
            ),
        ]

        # 5. 合并 MCP 工具和本地 Pandas 工具，并创建 Agent智能体
        all_tools = mcp_tools + local_tools

        agent = FunctionAgent(
            name="FinanceAgent",
            tools=all_tools,
            llm=llm,
            verbose=True,
            system_prompt="""你是一个财务对账机器人。
    你的任务是对比内部记录和银行账单。
    1. 首先分别查询内部记录和银行账单，找到可能匹配的项。
    2. 如果金额不完全一致（例如差异在3%以内），请务必利用工具（如 get_fee_description）分析是否为手续费或汇率原因。
    3. 如果商户名称不同，利用你的常识判断它们是否指向同一个实体（如 SBUX 指 Starbucks）。
    4. 给出清晰、专业的对账总结。"""
        )

        # 6. 执行对账指令
        print("\n--- 开始对账任务 ---")
        handler = agent.run(
            "请帮我核对内部记录和银行账单中的所有交易。请逐一对比各项，解释任何金额、日期或描述上的差异，并给出完整的对账报告。"
        )
        response = await handler
        print("\n--- AI 对账结论 ---")
        print(response)

        print(f"\n[系统通知] 日志已保存至: {log_filename}")

    finally:
        # 恢复 stdout 并关闭日志文件
        if isinstance(sys.stdout, Logger):
            sys.stdout.log.close()
            sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    asyncio.run(main())
