class PolyOracleError(Exception):
    pass


class ConfigError(PolyOracleError):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"Configuration error: {self.message}"


class LLMError(PolyOracleError):
    def __init__(self, message: str, model: str, prompt_tokens: int = 0) -> None:
        self.message = message
        self.model = model
        self.prompt_tokens = prompt_tokens
        super().__init__(message)

    def __str__(self) -> str:
        return f"LLM error with {self.model} ({self.prompt_tokens} tokens): {self.message}"


class DataFetchError(PolyOracleError):
    def __init__(self, message: str, source: str, url: str = "") -> None:
        self.message = message
        self.source = source
        self.url = url
        super().__init__(message)

    def __str__(self) -> str:
        url_info = f" from {self.url}" if self.url else ""
        return f"Failed to fetch data from {self.source}{url_info}: {self.message}"


class MarketNotFoundError(PolyOracleError):
    def __init__(self, market_id: str) -> None:
        self.market_id = market_id
        super().__init__(f"Market not found: {market_id}")

    def __str__(self) -> str:
        return f"Market {self.market_id} not found"


class InsufficientLiquidityError(PolyOracleError):
    def __init__(self, market_id: str, required: float, available: float) -> None:
        self.market_id = market_id
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient liquidity in market {market_id}: required {required}, available {available}"
        )

    def __str__(self) -> str:
        return (
            f"Market {self.market_id} has insufficient liquidity: "
            f"required ${self.required:.2f}, available ${self.available:.2f}"
        )


class EdgeTooSmallError(PolyOracleError):
    def __init__(self, edge: float, min_edge: float) -> None:
        self.edge = edge
        self.min_edge = min_edge
        super().__init__(f"Edge {edge} below minimum {min_edge}")

    def __str__(self) -> str:
        return f"Edge too small: {self.edge:.4f} < {self.min_edge:.4f}"


class RiskLimitError(PolyOracleError):
    def __init__(self, violation: str) -> None:
        self.violation = violation
        super().__init__(violation)

    def __str__(self) -> str:
        return f"Risk limit violated: {self.violation}"


class ExecutionError(PolyOracleError):
    def __init__(self, message: str, order_id: str = "", reason: str = "") -> None:
        self.message = message
        self.order_id = order_id
        self.reason = reason
        super().__init__(message)

    def __str__(self) -> str:
        order_info = f" (order {self.order_id})" if self.order_id else ""
        reason_info = f": {self.reason}" if self.reason else ""
        return f"Execution failed{order_info}{reason_info}"
