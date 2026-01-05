import asyncio
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("FinanceServer")

@mcp.tool()
async def get_exchange_rate(currency: str) -> float:
    """Get the exchange rate for a currency."""
    rates = {"USD": 7.2, "EUR": 7.8}
    return rates.get(currency.upper(), 1.0)

@mcp.tool()
async def get_fee_description(amount: float) -> str:
    """Get description for potential fees based on amount."""
    if amount > 1000:
        return "Possible 3% international transaction fee"
    return "Standard transaction fee"

if __name__ == "__main__":
    mcp.run(transport="stdio")
