from pydantic import BaseModel

class MonthlySalesData(BaseModel):
    month: str
    sales_volume_units: int
    revenue_million_usd: float
    average_price_per_unit_usd: float

class RegionalSalesData(BaseModel):
    region: str
    sales_volume_units: int
    revenue_million_usd: float
    market_share_percentage: float

class CustomerSegmentSalesData(BaseModel):
    customer_segment: str
    sales_volume_units: int
    revenue_million_usd: float
    average_order_value_usd: float
    customer_retention_rate_percentage: float

class SalesPipelineData(BaseModel):
    stage: str
    number_of_deals: int
    potential_revenue_million_usd: float
    conversion_rate_percentage: float

class SalesRepPerformanceData(BaseModel):
    sales_rep_id: str
    sales_rep_name: str
    sales_volume_units: int
    revenue_million_usd: float
    quota_achievement_percentage: float
    commission_earned_usd: float

class YearlySalesSummary(BaseModel):
    year: int
    total_sales_volume_units: int
    total_revenue_billion_usd: float
    net_profit_billion_usd: float
    year_over_year_growth_percentage: float

