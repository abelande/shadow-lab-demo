# Staircase Terminal — TradingView Integration

Visual signal overlay for TradingView charts, powered by the Staircase Terminal engine.

## Setup

### 1. Add the Pine Script Indicator

1. Open TradingView and navigate to your chart
2. Click **Pine Editor** (bottom panel)
3. Delete the default code
4. Copy the contents of `staircase_alerts.pine` and paste it in
5. Click **Add to Chart**
6. The indicator appears as "Staircase Terminal Alerts" in your chart's indicator list

### 2. Configure the Indicator Inputs

Click the ⚙️ gear icon on the indicator to customize:

| Input | Default | Description |
|-------|---------|-------------|
| Webhook URL | `https://your-server.com/webhook` | Display only — reminder of your endpoint |
| Min Confidence | 0.5 | Minimum confidence to show signals (0.0–1.0) |
| Show Regime Background | ✅ | Color chart background by detected regime |
| Show Confidence Labels | ✅ | Show confidence % next to signal markers |

### 3. Configure TradingView Alerts

1. Click **Alerts** (🔔) in TradingView
2. Create a new alert:
   - **Condition:** Select "Staircase Terminal Alerts" → "Staircase BUY" or "Staircase SELL" or "Staircase Any Signal"
   - **Webhook URL:** Enter your Staircase Terminal webhook endpoint
   - **Message:** Use the default JSON template or customize
3. Set expiration and notification preferences

### 4. Connect to Staircase Terminal

The signal flow:

```
┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Staircase Engine │────▶│  Webhook Server  │────▶│   TradingView    │
│  (P6 Pipeline)   │     │  (your endpoint) │     │   Alert Webhook  │
└──────────────────┘     └─────────────────┘     └──────────────────┘
         │                                                │
         │  OrderBookSnapshot → DepthIndicatorFrame       │
         │  → AggregatedSignal                            │
         │                                                │
         └──── direction, confidence, regime ─────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Pine Script      │
                    │   Visual Overlay   │
                    │   🟢 BUY / 🔴 SELL  │
                    └────────────────────┘
```

**In your Staircase Terminal config**, set the webhook URL to push signals:

```python
# Example: push signals to TradingView via your relay server
WEBHOOK_URL = "https://your-server.com/api/tradingview/signal"
```

Your relay server receives the Staircase signal and forwards it as a TradingView alert payload.

## Signal Colors

| Element | Meaning |
|---------|---------|
| 🟢 Green triangle (below bar) | BUY signal |
| 🔴 Red triangle (above bar) | SELL signal |
| Blue background | Trending regime |
| Gray background | Ranging regime |
| Orange background | Volatile regime |

## Notes

- TradingView Pine Script cannot receive webhooks directly — it only *sends* them via alerts
- For real-time signal display, you need a relay that updates TradingView inputs via the TradingView API, or use the alert-based approach where signals trigger Pine alert conditions
- The confidence threshold slider filters out low-confidence noise
- Regime background gives at-a-glance context for signal quality
