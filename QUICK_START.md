# Quick Start Guide

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your position files in the `input/` directory:
   - `1.XTP.xlsx` (or `.csv`) - XTP positions
   - `2.JPM.xlsx` (or `.csv`) - JPM positions

3. Configure multipliers (if needed):
   - Edit `config/strike_multipliers.csv`
   - Edit `config/settle_multipliers.csv`

## Run Reconciliation

```bash
python reconcile.py
```

## Output

Results will be saved to:
- `output/reconciliation_YYYYMMDD_HHMMSS.xlsx` - Excel report with multiple sheets
- `logs/reconcile_YYYYMMDD_HHMMSS.log` - Detailed log file

## Input File Format Examples

### XTP File Columns
- **JPM Account** (exact name preferred)
- **JPM GMI Code** (exact name preferred)
- **Strike** (exact name preferred)
- **SettlementPrice** (exact name preferred)
- **Quantity** (flexible: "Qty" or "Quantity" also work)

### JPM File Columns
- **JPM ID / Broker ID** (exact name preferred)
- **Product / Exchange Ticker** (exact name preferred)
- **Strike** (exact name preferred)
- **Settlement_Price** (exact name preferred)
- **Quantity** (flexible: "Qty" or "Quantity" also work)

**Note**: The tool prioritizes exact column names but falls back to flexible case-insensitive matching if exact names are not found.

## Custom File Paths

```bash
python reconcile.py --xtp-file path/to/xtp.xlsx --jpm-file path/to/jpm.xlsx
```

## Tolerance Settings

Allow small quantity differences:
```bash
python reconcile.py --qty-tolerance 1 --qty-percent-tolerance 0.001
```
