# Position Reconciliation Tool: XTP vs JPM

A comprehensive Python tool for reconciling position files between XTP and JPM trading systems.

## Features

- **Flexible Input Formats**: Supports both Excel (.xlsx, .xls) and CSV files
- **External Multiplier Configuration**: Strike and settlement price multipliers loaded from CSV files
- **Intelligent Matching**: Matches positions by Account, Product Code, and adjusted prices
- **Quantity Aggregation**: Automatically aggregates quantities for matching groups
- **Comprehensive Reporting**: Generates detailed Excel reports with multiple sheets
- **Tolerance Settings**: Configurable tolerances for price and quantity comparisons
- **Detailed Logging**: Complete audit trail of reconciliation process

## Installation

1. Install Python 3.8 or higher
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Directory Structure

```
/reconciliation
├── reconcile.py (main script)
├── config/
│   ├── strike_multipliers.csv
│   └── settle_multipliers.csv
├── input/
│   ├── 1.XTP.xlsx
│   └── 2.JPM.xlsx
├── output/
│   └── reconciliation_YYYYMMDD_HHMMSS.xlsx
└── logs/
    └── reconcile_YYYYMMDD_HHMMSS.log
```

## Usage

### Basic Usage

```bash
python reconcile.py
```

This will use default file paths:
- XTP file: `input/1.XTP.xlsx`
- JPM file: `input/2.JPM.xlsx`
- Strike multipliers: `config/strike_multipliers.csv`
- Settlement multipliers: `config/settle_multipliers.csv`

### Custom File Paths

```bash
python reconcile.py --xtp-file path/to/xtp.xlsx --jpm-file path/to/jpm.xlsx
```

### Full Options

```bash
python reconcile.py \
    --xtp-file input/1.XTP.xlsx \
    --jpm-file input/2.JPM.xlsx \
    --strike-multiplier-file config/strike_multipliers.csv \
    --settle-multiplier-file config/settle_multipliers.csv \
    --output-dir output \
    --config-dir config \
    --log-dir logs \
    --strike-tolerance 0.01 \
    --qty-tolerance 0.0 \
    --qty-percent-tolerance 0.0
```

## Input File Requirements

### XTP Position File

Required columns (exact names preferred, with fallback to flexible matching):
- **JPM Account**: Account number
- **JPM GMI Code**: Product code (falls back to "Clearing Code" or "Exchange Clearing Code" if not found)
- **Strike**: Strike price value
- **SettlementPrice**: Settlement price value
- **Quantity**: Position quantity (flexible matching for "Qty" or "Quantity")

### JPM Position File

Required columns (exact names preferred, with fallback to flexible matching):
- **JPM ID / Broker ID**: Account number
- **Product / Exchange Ticker**: Product code
- **Strike**: Strike price value
- **Settlement_Price**: Settlement price value
- **Quantity**: Position quantity (flexible matching for "Qty" or "Quantity")

## Multiplier Configuration

You can apply **strike** and **settlement** price multipliers **independently for XTP and JPM**.

### Option 1: Combined file (independent XTP vs JPM)

Use a single CSV with four multiplier columns so strike and settle can differ by source:

**File**: `config/multipliers.csv` (or pass `--multipliers-file path/to/multipliers.csv`)

```csv
Product_Code,Strike_Mult_XTP,Strike_Mult_JPM,Settle_Mult_XTP,Settle_Mult_JPM
ES,1,1,1,1
NQ,0.01,0.01,0.01,0.01
GC,0.1,0.1,0.1,0.1
CL,1,1,1,1
```

- **Strike_Mult_XTP** / **Strike_Mult_JPM**: applied to XTP and JPM strike prices respectively.
- **Settle_Mult_XTP** / **Settle_Mult_JPM**: applied to XTP and JPM settlement prices respectively.

If `config/multipliers.csv` exists (or you pass `--multipliers-file`), this file is used and the legacy strike/settle files below are ignored.

A template is in `config/multipliers.example.csv`; copy to `config/multipliers.csv` to use it.

### Option 2: Legacy (same multiplier for both sides)

If the combined file is not used, the tool falls back to two files. The **same** multiplier is applied to both XTP and JPM for that price type.

**strike_multipliers.csv**

```csv
Product_Code,Multiplier
ES,1
NQ,0.01
GC,0.1
CL,1
```

**settle_multipliers.csv**

```csv
Product_Code,Multiplier
ES,1
NQ,0.01
GC,0.1
CL,1
```

**Note**: If a product code is not found in the multiplier files, a default multiplier of 1.0 will be used and a warning will be logged.

## Matching Logic

Positions are matched based on:
1. **Account**: 
   - XTP: Uses "JPM Account"
   - JPM: Uses "JPM ID / Broker ID"
2. **Product Code**: 
   - XTP: Uses "JPM GMI Code" (falls back to "Clearing Code" or "Exchange Clearing Code" if not found)
   - JPM: Uses "Product / Exchange Ticker"
3. **Strike Price**: Adjusted using strike multiplier
   - XTP: Uses "Strike" column
   - JPM: Uses "Strike" column
4. **Settlement Price**: Adjusted using settlement multiplier
   - XTP: Uses "SettlementPrice" column
   - JPM: Uses "Settlement_Price" column
5. **Quantity**: Sum of quantities for all matching records

## Output Files

### Excel Report Structure

The reconciliation report is saved as an Excel file with the following sheets:

1. **Matched**: Perfectly matched positions with quantity differences
2. **Unmatched_All**: Combined view of:
   - XTP Only positions
   - JPM Only positions
   - Quantity mismatches
3. **Detail_Breakdown**: Individual records for mismatched groups
4. **Summary**: Reconciliation statistics
5. **Config_Used**: Multiplier configuration used (audit trail)

### Color Coding

- **XTP Only**: XTP Quantity column highlighted in light red
- **JPM Only**: JPM Quantity column highlighted in light blue
- **Qty Mismatch**: Difference column highlighted in light orange

## Tolerance Settings

- **--strike-tolerance**: Tolerance for strike price comparisons (default: 0.01)
- **--qty-tolerance**: Absolute tolerance for quantity differences (default: 0.0)
- **--qty-percent-tolerance**: Percentage tolerance for quantity differences (default: 0.0)

Example with tolerances:
```bash
python reconcile.py --qty-tolerance 1 --qty-percent-tolerance 0.001
```

## Logging

All reconciliation activities are logged to:
- Console output (stdout)
- Log file: `logs/reconcile_YYYYMMDD_HHMMSS.log`

Logs include:
- Multiplier files loaded
- Products with missing multipliers (warnings)
- Data validation warnings
- Reconciliation statistics

## Example Scenario

**XTP Records:**
- Account: 123, Product: ES, Strike: 5000, Settle: 5010, Qty: 10
- Account: 123, Product: ES, Strike: 5000, Settle: 5010, Qty: 5

**JPM Records:**
- Account: 123, Product: ES, Strike: 5000, Settle: 5010, Qty: 15

**Multipliers:**
- ES Strike Multiplier: 1
- ES Settle Multiplier: 1

**Result**: ✅ MATCH (XTP total: 15, JPM total: 15)

## Error Handling

The tool includes comprehensive error handling:
- Validates input file existence
- Checks for required columns
- Validates multiplier file structure
- Handles missing/null values gracefully
- Provides clear error messages

## Maintenance

### Updating Multipliers

Simply edit the CSV files in the `config/` directory:
- No code changes required
- Changes take effect on next run
- Template files are created automatically if missing

### Adding New Products

1. Add product code and multiplier to `config/strike_multipliers.csv`
2. Add product code and multiplier to `config/settle_multipliers.csv`
3. Run reconciliation - no code changes needed

## Troubleshooting

### "Missing required columns" error
- Check that your input files contain the required column names (case-insensitive matching is supported)
- Ensure column names contain keywords like "account", "strike", "settle", "qty"

### "Product code not found in multipliers" warning
- Add the product code to both multiplier CSV files
- Use default multiplier of 1.0 if appropriate

### Empty results
- Verify input files are not empty
- Check that data types are correct (numeric for prices and quantities)
- Review log file for data validation warnings

## Support

For issues or questions, please review the log files in the `logs/` directory for detailed error messages and reconciliation steps.
