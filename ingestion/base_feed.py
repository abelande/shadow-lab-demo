"""Abstract base class for all order book data feeds."""
from __future__ import annotations

import abc
from typing import Literal, Optional

from ..models import OrderBookSnapshot


class BaseFeed(abc.ABC):
    """Abstract base class for order book data feeds.

    All feeds (synthetic, replay, live) implement this interface so the
    pipeline and backtest runner can consume them interchangeably.
    """

    def __init__(self, symbol: str, data_level: Literal["L1", "L2", "L3"] = "L2"):
        self._symbol = symbol
        self._data_level = data_level
        self._connected = False

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def data_level(self) -> Literal["L1", "L2", "L3"]:
        return self._data_level

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Open the feed connection / initialize resources."""
        self._connected = True

    @abc.abstractmethod
    async def next(self) -> Optional[OrderBookSnapshot]:
        """Return the next snapshot, or None when exhausted."""
        ...

    async def disconnect(self) -> None:
        """Close the feed connection / release resources."""
        self._connected = False
