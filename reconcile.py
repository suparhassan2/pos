#!/usr/bin/env python3
"""
Position Reconciliation Tool: XTP vs JPM

This script reconciles position files between XTP and JPM systems by:
1. Loading position data from both systems
2. Applying strike and settlement price multipliers
3. Aggregating quantities by Account + Product Code + Adjusted Prices
4. Comparing matched and unmatched positions
5. Generating detailed reconciliation reports
"""

import pandas as pd
import numpy as np
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class ReconciliationTool:
    """Main reconciliation tool class"""
    
    def __init__(self, config_dir: str = "config", log_dir: str = "logs", 
                 strike_tolerance: float = 0.01, qty_tolerance: float = 0.0,
                 qty_percent_tolerance: float = 0.0):
        """
        Initialize the reconciliation tool
        
        Args:
            config_dir: Directory containing multiplier CSV files
            log_dir: Directory for log files
            strike_tolerance: Tolerance for price comparisons
            qty_tolerance: Absolute tolerance for quantity differences
            qty_percent_tolerance: Percentage tolerance for quantity differences
        """
        self.config_dir = Path(config_dir)
        self.log_dir = Path(log_dir)
        self.strike_tolerance = strike_tolerance
        self.qty_tolerance = qty_tolerance
        self.qty_percent_tolerance = qty_percent_tolerance
        
        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Load multipliers
        self.strike_multipliers = {}
        self.settle_multipliers = {}
        self.products_with_default_multipliers = set()
        
        self.logger.info("Reconciliation Tool initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"reconcile_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized: {log_file}")
    
    def load_multipliers(self, strike_file: Optional[str] = None, 
                        settle_file: Optional[str] = None) -> Tuple[Dict, Dict]:
        """
        Load multiplier tables from CSV files
        
        Args:
            strike_file: Path to strike multipliers CSV (default: config/strike_multipliers.csv)
            settle_file: Path to settlement multipliers CSV (default: config/settle_multipliers.csv)
        
        Returns:
            Tuple of (strike_multipliers dict, settle_multipliers dict)
        """
        strike_file = strike_file or self.config_dir / "strike_multipliers.csv"
        settle_file = settle_file or self.config_dir / "settle_multipliers.csv"
        
        # Create template files if they don't exist
        if not Path(strike_file).exists():
            self.logger.warning(f"Strike multipliers file not found: {strike_file}")
            self.logger.info("Creating template strike_multipliers.csv")
            self._create_template_multiplier_file(strike_file, "strike")
        
        if not Path(settle_file).exists():
            self.logger.warning(f"Settlement multipliers file not found: {settle_file}")
            self.logger.info("Creating template settle_multipliers.csv")
            self._create_template_multiplier_file(settle_file, "settlement")
        
        # Load strike multipliers
        try:
            strike_df = pd.read_csv(strike_file)
            self._validate_multiplier_file(strike_df, "strike")
            self.strike_multipliers = dict(zip(
                strike_df['Product_Code'].astype(str).str.upper(),
                pd.to_numeric(strike_df['Multiplier'], errors='coerce')
            ))
            self.logger.info(f"Loaded {len(self.strike_multipliers)} strike multipliers from {strike_file}")
        except Exception as e:
            self.logger.error(f"Error loading strike multipliers: {e}")
            raise
        
        # Load settlement multipliers
        try:
            settle_df = pd.read_csv(settle_file)
            self._validate_multiplier_file(settle_df, "settlement")
            self.settle_multipliers = dict(zip(
                settle_df['Product_Code'].astype(str).str.upper(),
                pd.to_numeric(settle_df['Multiplier'], errors='coerce')
            ))
            self.logger.info(f"Loaded {len(self.settle_multipliers)} settlement multipliers from {settle_file}")
        except Exception as e:
            self.logger.error(f"Error loading settlement multipliers: {e}")
            raise
        
        return self.strike_multipliers, self.settle_multipliers
    
    def _create_template_multiplier_file(self, filepath: Path, multiplier_type: str):
        """Create a template multiplier CSV file"""
        template_data = {
            'Product_Code': ['ES', 'NQ', 'GC', 'CL'],
            'Multiplier': [1, 0.01, 0.1, 1]
        }
        df = pd.DataFrame(template_data)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Created template {multiplier_type} multiplier file: {filepath}")
    
    def _validate_multiplier_file(self, df: pd.DataFrame, file_type: str):
        """Validate multiplier file structure and data"""
        required_columns = ['Product_Code', 'Multiplier']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"{file_type} multiplier file missing required columns: {missing_columns}")
        
        # Check for invalid numeric values
        invalid_multipliers = df[~pd.to_numeric(df['Multiplier'], errors='coerce').notna()]
        if len(invalid_multipliers) > 0:
            self.logger.warning(f"Found {len(invalid_multipliers)} invalid multiplier values in {file_type} file")
        
        self.logger.info(f"Validated {file_type} multiplier file: {len(df)} products")
    
    def get_multiplier(self, product_code: str, multiplier_type: str = 'strike') -> float:
        """
        Get multiplier for a product code
        
        Args:
            product_code: Product code to look up
            multiplier_type: 'strike' or 'settle'
        
        Returns:
            Multiplier value (default: 1.0 if not found)
        """
        product_code = str(product_code).upper().strip()
        multipliers = self.strike_multipliers if multiplier_type == 'strike' else self.settle_multipliers
        
        if product_code not in multipliers:
            if product_code not in self.products_with_default_multipliers:
                self.logger.warning(
                    f"Product code '{product_code}' not found in {multiplier_type} multipliers. "
                    f"Using default multiplier of 1.0"
                )
                self.products_with_default_multipliers.add(product_code)
            return 1.0
        
        return multipliers[product_code]
    
    def load_position_file(self, filepath: str, file_type: str) -> pd.DataFrame:
        """
        Load position file (Excel or CSV)
        
        Args:
            filepath: Path to position file
            file_type: 'XTP' or 'JPM'
        
        Returns:
            DataFrame with position data
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"{file_type} position file not found: {filepath}")
        
        self.logger.info(f"Loading {file_type} positions from: {filepath}")
        
        try:
            if filepath.suffix.lower() == '.csv':
                df = pd.read_csv(filepath)
            elif filepath.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
            self.logger.info(f"Loaded {len(df)} records from {file_type} file")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            return df
        except Exception as e:
            self.logger.error(f"Error loading {file_type} file: {e}")
            raise
    
    def prepare_xtp_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare XTP data for reconciliation
        
        Args:
            df: Raw XTP DataFrame
        
        Returns:
            Prepared DataFrame with standardized columns
        """
        self.logger.info("Preparing XTP data...")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Standardize column names - prioritize exact matches, then fall back to flexible matching
        column_mapping = {}
        
        # Account: "JPM Account" (exact match first)
        if 'JPM Account' in df.columns:
            column_mapping['JPM Account'] = 'Account'
        else:
            # Fall back to flexible matching
            for col in df.columns:
                col_lower = col.lower()
                if 'account' in col_lower and 'Account' not in column_mapping.values():
                    column_mapping[col] = 'Account'
        
        # Product Code: "JPM GMI Code" (exact match first)
        product_code_source = None
        if 'JPM GMI Code' in df.columns:
            df['Product_Code'] = df['JPM GMI Code']
            product_code_source = 'JPM GMI Code'
        elif 'Clearing Code' in df.columns:
            df['Product_Code'] = df['Clearing Code']
            product_code_source = 'Clearing Code'
        elif 'Exchange Clearing Code' in df.columns:
            df['Product_Code'] = df['Exchange Clearing Code']
            product_code_source = 'Exchange Clearing Code'
        else:
            # Fall back to flexible matching
            for col in df.columns:
                col_lower = col.lower()
                if ('clearing' in col_lower or 'product' in col_lower) and 'code' in col_lower:
                    df['Product_Code'] = df[col]
                    product_code_source = col
                    break
            if 'Product_Code' not in df.columns:
                raise ValueError("XTP file must contain 'JPM GMI Code' or similar product code column")
        
        # Strike Price: "Strike" (exact match first)
        if 'Strike' in df.columns:
            column_mapping['Strike'] = 'Strike_Price'
        else:
            # Fall back to flexible matching
            for col in df.columns:
                col_lower = col.lower()
                if 'strike' in col_lower and 'Strike_Price' not in column_mapping.values():
                    column_mapping[col] = 'Strike_Price'
        
        # Settlement Price: "SettlementPrice" (exact match first)
        if 'SettlementPrice' in df.columns:
            column_mapping['SettlementPrice'] = 'Settlement_Price'
        else:
            # Fall back to flexible matching
            for col in df.columns:
                col_lower = col.lower()
                if 'settle' in col_lower and 'Settlement_Price' not in column_mapping.values():
                    column_mapping[col] = 'Settlement_Price'
        
        # Quantity: flexible matching
        for col in df.columns:
            col_lower = col.lower()
            if ('qty' in col_lower or 'quantity' in col_lower) and 'Quantity' not in column_mapping.values():
                column_mapping[col] = 'Quantity'
        
        df = df.rename(columns=column_mapping)
        
        # Log which columns were used
        if column_mapping:
            self.logger.info(f"XTP column mappings: {column_mapping}")
        if product_code_source:
            self.logger.info(f"XTP Product_Code mapped from: {product_code_source}")
        
        # Validate required columns
        required_cols = ['Account', 'Product_Code', 'Strike_Price', 'Settlement_Price', 'Quantity']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"XTP file missing required columns: {missing_cols}. Available columns: {list(df.columns)}")
        
        # Clean and convert data types
        df['Account'] = df['Account'].astype(str).str.strip()
        df['Product_Code'] = df['Product_Code'].astype(str).str.strip().str.upper()
        df['Strike_Price'] = pd.to_numeric(df['Strike_Price'], errors='coerce')
        df['Settlement_Price'] = pd.to_numeric(df['Settlement_Price'], errors='coerce')
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        
        # Remove rows with missing critical data
        initial_count = len(df)
        df = df.dropna(subset=['Account', 'Product_Code', 'Strike_Price', 'Settlement_Price', 'Quantity'])
        if len(df) < initial_count:
            self.logger.warning(f"Removed {initial_count - len(df)} rows with missing data from XTP")
        
        # Apply multipliers
        df['Strike_Multiplier'] = df['Product_Code'].apply(
            lambda x: self.get_multiplier(x, 'strike')
        )
        df['Settle_Multiplier'] = df['Product_Code'].apply(
            lambda x: self.get_multiplier(x, 'settle')
        )
        
        df['Strike_Price_Adjusted'] = df['Strike_Price'] * df['Strike_Multiplier']
        df['Settlement_Price_Adjusted'] = df['Settlement_Price'] * df['Settle_Multiplier']
        
        # Round adjusted prices for matching
        df['Strike_Price_Adjusted'] = df['Strike_Price_Adjusted'].round(2)
        df['Settlement_Price_Adjusted'] = df['Settlement_Price_Adjusted'].round(2)
        
        self.logger.info(f"Prepared {len(df)} XTP records")
        return df
    
    def prepare_jpm_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare JPM data for reconciliation
        
        Args:
            df: Raw JPM DataFrame
        
        Returns:
            Prepared DataFrame with standardized columns
        """
        self.logger.info("Preparing JPM data...")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Standardize column names - prioritize exact matches, then fall back to flexible matching
        column_mapping = {}
        
        # Account: "JPM ID / Broker ID" (exact match first)
        if 'JPM ID / Broker ID' in df.columns:
            column_mapping['JPM ID / Broker ID'] = 'Account'
        else:
            # Fall back to flexible matching
            for col in df.columns:
                col_lower = col.lower()
                if 'account' in col_lower or ('jpm' in col_lower and 'id' in col_lower):
                    if 'Account' not in column_mapping.values():
                        column_mapping[col] = 'Account'
        
        # Product Code: "Product / Exchange Ticker" (exact match first)
        if 'Product / Exchange Ticker' in df.columns:
            column_mapping['Product / Exchange Ticker'] = 'Product_Code'
        else:
            # Fall back to flexible matching
            for col in df.columns:
                col_lower = col.lower()
                if 'product' in col_lower and ('code' in col_lower or 'ticker' in col_lower):
                    if 'Product_Code' not in column_mapping.values():
                        column_mapping[col] = 'Product_Code'
        
        # Strike Price: "Strike" (exact match first)
        if 'Strike' in df.columns:
            column_mapping['Strike'] = 'Strike_Price'
        else:
            # Fall back to flexible matching
            for col in df.columns:
                col_lower = col.lower()
                if 'strike' in col_lower and 'Strike_Price' not in column_mapping.values():
                    column_mapping[col] = 'Strike_Price'
        
        # Settlement Price: "Settlement_Price" (exact match first)
        if 'Settlement_Price' in df.columns:
            column_mapping['Settlement_Price'] = 'Settlement_Price'
        else:
            # Fall back to flexible matching
            for col in df.columns:
                col_lower = col.lower()
                if 'settle' in col_lower and 'Settlement_Price' not in column_mapping.values():
                    column_mapping[col] = 'Settlement_Price'
        
        # Quantity: flexible matching
        for col in df.columns:
            col_lower = col.lower()
            if ('qty' in col_lower or 'quantity' in col_lower) and 'Quantity' not in column_mapping.values():
                column_mapping[col] = 'Quantity'
        
        df = df.rename(columns=column_mapping)
        
        # Log which columns were used
        if column_mapping:
            self.logger.info(f"JPM column mappings: {column_mapping}")
        
        # Validate required columns
        required_cols = ['Account', 'Product_Code', 'Strike_Price', 'Settlement_Price', 'Quantity']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"JPM file missing required columns: {missing_cols}. Available columns: {list(df.columns)}")
        
        # Clean and convert data types
        df['Account'] = df['Account'].astype(str).str.strip()
        df['Product_Code'] = df['Product_Code'].astype(str).str.strip().str.upper()
        df['Strike_Price'] = pd.to_numeric(df['Strike_Price'], errors='coerce')
        df['Settlement_Price'] = pd.to_numeric(df['Settlement_Price'], errors='coerce')
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        
        # Remove rows with missing critical data
        initial_count = len(df)
        df = df.dropna(subset=['Account', 'Product_Code', 'Strike_Price', 'Settlement_Price', 'Quantity'])
        if len(df) < initial_count:
            self.logger.warning(f"Removed {initial_count - len(df)} rows with missing data from JPM")
        
        # Apply multipliers
        df['Strike_Multiplier'] = df['Product_Code'].apply(
            lambda x: self.get_multiplier(x, 'strike')
        )
        df['Settle_Multiplier'] = df['Product_Code'].apply(
            lambda x: self.get_multiplier(x, 'settle')
        )
        
        df['Strike_Price_Adjusted'] = df['Strike_Price'] * df['Strike_Multiplier']
        df['Settlement_Price_Adjusted'] = df['Settlement_Price'] * df['Settle_Multiplier']
        
        # Round adjusted prices for matching
        df['Strike_Price_Adjusted'] = df['Strike_Price_Adjusted'].round(2)
        df['Settlement_Price_Adjusted'] = df['Settlement_Price_Adjusted'].round(2)
        
        self.logger.info(f"Prepared {len(df)} JPM records")
        return df
    
    def aggregate_positions(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Aggregate positions by Account + Product Code + Adjusted Prices
        
        Args:
            df: Prepared position DataFrame
            source: 'XTP' or 'JPM'
        
        Returns:
            Aggregated DataFrame
        """
        self.logger.info(f"Aggregating {source} positions...")
        
        group_cols = ['Account', 'Product_Code', 'Strike_Price_Adjusted', 'Settlement_Price_Adjusted']
        
        # Aggregate quantities
        agg_df = df.groupby(group_cols, as_index=False).agg({
            'Quantity': 'sum',
            'Strike_Price': 'first',  # Keep original strike price for display
            'Settlement_Price': 'first',  # Keep original settlement price for display
            'Strike_Multiplier': 'first',
            'Settle_Multiplier': 'first'
        })
        
        # Store detail records for later breakdown
        agg_df['_detail_records'] = agg_df.apply(
            lambda row: df[
                (df['Account'] == row['Account']) &
                (df['Product_Code'] == row['Product_Code']) &
                (df['Strike_Price_Adjusted'] == row['Strike_Price_Adjusted']) &
                (df['Settlement_Price_Adjusted'] == row['Settlement_Price_Adjusted'])
            ].to_dict('records'),
            axis=1
        )
        
        self.logger.info(f"Aggregated {len(df)} {source} records into {len(agg_df)} groups")
        return agg_df
    
    def reconcile(self, xtp_file: str, jpm_file: str, 
                  strike_mult_file: Optional[str] = None,
                  settle_mult_file: Optional[str] = None) -> Dict:
        """
        Perform reconciliation between XTP and JPM positions
        
        Args:
            xtp_file: Path to XTP position file
            jpm_file: Path to JPM position file
            strike_mult_file: Optional path to strike multipliers file
            settle_mult_file: Optional path to settlement multipliers file
        
        Returns:
            Dictionary containing reconciliation results
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Reconciliation")
        self.logger.info("=" * 80)
        
        # Load multipliers
        self.load_multipliers(strike_mult_file, settle_mult_file)
        
        # Load position files
        xtp_raw = self.load_position_file(xtp_file, 'XTP')
        jpm_raw = self.load_position_file(jpm_file, 'JPM')
        
        # Prepare data
        xtp_df = self.prepare_xtp_data(xtp_raw)
        jpm_df = self.prepare_jpm_data(jpm_raw)
        
        # Aggregate positions
        xtp_agg = self.aggregate_positions(xtp_df, 'XTP')
        jpm_agg = self.aggregate_positions(jpm_df, 'JPM')
        
        # Create matching keys
        xtp_agg['Match_Key'] = (
            xtp_agg['Account'].astype(str) + '|' +
            xtp_agg['Product_Code'].astype(str) + '|' +
            xtp_agg['Strike_Price_Adjusted'].astype(str) + '|' +
            xtp_agg['Settlement_Price_Adjusted'].astype(str)
        )
        
        jpm_agg['Match_Key'] = (
            jpm_agg['Account'].astype(str) + '|' +
            jpm_agg['Product_Code'].astype(str) + '|' +
            jpm_agg['Strike_Price_Adjusted'].astype(str) + '|' +
            jpm_agg['Settlement_Price_Adjusted'].astype(str)
        )
        
        # Merge for comparison
        merged = pd.merge(
            xtp_agg,
            jpm_agg,
            on='Match_Key',
            how='outer',
            suffixes=('_XTP', '_JPM')
        )
        
        # Fill missing quantities with 0
        merged['Quantity_XTP'] = merged['Quantity_XTP'].fillna(0)
        merged['Quantity_JPM'] = merged['Quantity_JPM'].fillna(0)
        
        # Calculate difference
        merged['Quantity_Difference'] = merged['Quantity_XTP'] - merged['Quantity_JPM']
        
        # Determine match status
        def get_status(row):
            if pd.isna(row['Quantity_XTP']) or row['Quantity_XTP'] == 0:
                return 'JPM Only'
            elif pd.isna(row['Quantity_JPM']) or row['Quantity_JPM'] == 0:
                return 'XTP Only'
            else:
                diff = abs(row['Quantity_Difference'])
                if diff <= self.qty_tolerance:
                    return 'Matched'
                elif self.qty_percent_tolerance > 0:
                    max_qty = max(abs(row['Quantity_XTP']), abs(row['Quantity_JPM']))
                    if max_qty > 0 and (diff / max_qty) <= self.qty_percent_tolerance:
                        return 'Matched'
                return 'Qty Mismatch'
        
        merged['Status'] = merged.apply(get_status, axis=1)
        
        # Extract common columns for display
        merged['Account'] = merged['Account_XTP'].fillna(merged['Account_JPM'])
        merged['Product_Code'] = merged['Product_Code_XTP'].fillna(merged['Product_Code_JPM'])
        merged['Strike_Price_Adjusted'] = merged['Strike_Price_Adjusted_XTP'].fillna(
            merged['Strike_Price_Adjusted_JPM']
        )
        merged['Settlement_Price_Adjusted'] = merged['Settlement_Price_Adjusted_XTP'].fillna(
            merged['Settlement_Price_Adjusted_JPM']
        )
        
        # Prepare results
        results = {
            'merged': merged,
            'xtp_detail': xtp_df,
            'jpm_detail': jpm_df,
            'xtp_agg': xtp_agg,
            'jpm_agg': jpm_agg,
            'strike_multipliers': self.strike_multipliers,
            'settle_multipliers': self.settle_multipliers,
            'products_with_default_multipliers': self.products_with_default_multipliers
        }
        
        self.logger.info("Reconciliation completed")
        return results
    
    def generate_summary_stats(self, results: Dict) -> pd.DataFrame:
        """Generate summary statistics"""
        merged = results['merged']
        
        stats = {
            'Metric': [
                'Total XTP Records',
                'Total JPM Records',
                'Total Unique Groups',
                'Matched Groups',
                'Quantity Mismatch Groups',
                'XTP Only Groups',
                'JPM Only Groups',
                'Overall Match Percentage',
                'Products Using Default Multipliers'
            ],
            'Value': [
                len(results['xtp_detail']),
                len(results['jpm_detail']),
                len(merged),
                len(merged[merged['Status'] == 'Matched']),
                len(merged[merged['Status'] == 'Qty Mismatch']),
                len(merged[merged['Status'] == 'XTP Only']),
                len(merged[merged['Status'] == 'JPM Only']),
                f"{(len(merged[merged['Status'] == 'Matched']) / len(merged) * 100):.2f}%" if len(merged) > 0 else "0%",
                len(results['products_with_default_multipliers'])
            ]
        }
        
        return pd.DataFrame(stats)
    
    def export_results(self, results: Dict, output_file: str):
        """
        Export reconciliation results to Excel
        
        Args:
            results: Reconciliation results dictionary
            output_file: Path to output Excel file
        """
        self.logger.info(f"Exporting results to: {output_file}")
        
        merged = results['merged']
        
        # Prepare matched positions
        matched = merged[merged['Status'] == 'Matched'].copy()
        matched_display = matched[[
            'Account', 'Product_Code', 'Strike_Price_XTP', 'Strike_Price_Adjusted',
            'Settlement_Price_XTP', 'Settlement_Price_Adjusted',
            'Quantity_XTP', 'Quantity_JPM', 'Quantity_Difference', 'Status'
        ]].copy()
        matched_display.columns = [
            'Account', 'Product Code', 'Strike Price (Original)', 'Strike Price (Adjusted)',
            'Settlement Price (Original)', 'Settlement Price (Adjusted)',
            'XTP Quantity', 'JPM Quantity', 'Quantity Difference', 'Match Status'
        ]
        
        # Prepare unmatched positions (combined view)
        unmatched = merged[merged['Status'] != 'Matched'].copy()
        
        # Sort unmatched by Status, then by absolute Difference (descending)
        unmatched['Abs_Difference'] = unmatched['Quantity_Difference'].abs()
        unmatched = unmatched.sort_values(
            by=['Status', 'Abs_Difference'],
            ascending=[True, False]
        )
        
        unmatched_display = unmatched[[
            'Account', 'Product_Code', 'Strike_Price_Adjusted', 'Settlement_Price_Adjusted',
            'Quantity_XTP', 'Quantity_JPM', 'Quantity_Difference', 'Status'
        ]].copy()
        unmatched_display.columns = [
            'Account', 'Product Code', 'Strike Price (Adjusted)', 'Settlement Price (Adjusted)',
            'XTP Quantity', 'JPM Quantity', 'Difference', 'Status'
        ]
        
        # Replace 0 with blank for better readability
        unmatched_display['XTP Quantity'] = unmatched_display['XTP Quantity'].replace(0, '')
        unmatched_display['JPM Quantity'] = unmatched_display['JPM Quantity'].replace(0, '')
        
        # Prepare detail breakdown for mismatches
        detail_breakdown = []
        for idx, row in unmatched.iterrows():
            # Safely extract detail records
            xtp_details = []
            jpm_details = []
            
            # Extract XTP detail records
            try:
                if '_detail_records_XTP' in unmatched.columns:
                    detail_val = row.get('_detail_records_XTP', None)
                    if detail_val is not None:
                        # Handle NaN values (pandas converts some values to float NaN)
                        if isinstance(detail_val, float) and pd.isna(detail_val):
                            xtp_details = []
                        elif isinstance(detail_val, list):
                            xtp_details = detail_val
            except Exception as e:
                self.logger.debug(f"Error extracting XTP detail records: {e}")
            
            # Extract JPM detail records
            try:
                if '_detail_records_JPM' in unmatched.columns:
                    detail_val = row.get('_detail_records_JPM', None)
                    if detail_val is not None:
                        # Handle NaN values (pandas converts some values to float NaN)
                        if isinstance(detail_val, float) and pd.isna(detail_val):
                            jpm_details = []
                        elif isinstance(detail_val, list):
                            jpm_details = detail_val
            except Exception as e:
                self.logger.debug(f"Error extracting JPM detail records: {e}")
            
            for detail in xtp_details:
                if isinstance(detail, dict):
                    detail_breakdown.append({
                        'Account': row.get('Account', ''),
                        'Product_Code': row.get('Product_Code', ''),
                        'Strike_Price_Adjusted': row.get('Strike_Price_Adjusted', ''),
                        'Settlement_Price_Adjusted': row.get('Settlement_Price_Adjusted', ''),
                        'Source': 'XTP',
                        'Strike_Price': detail.get('Strike_Price', ''),
                        'Settlement_Price': detail.get('Settlement_Price', ''),
                        'Quantity': detail.get('Quantity', '')
                    })
            
            for detail in jpm_details:
                if isinstance(detail, dict):
                    detail_breakdown.append({
                        'Account': row.get('Account', ''),
                        'Product_Code': row.get('Product_Code', ''),
                        'Strike_Price_Adjusted': row.get('Strike_Price_Adjusted', ''),
                        'Settlement_Price_Adjusted': row.get('Settlement_Price_Adjusted', ''),
                        'Source': 'JPM',
                        'Strike_Price': detail.get('Strike_Price', ''),
                        'Settlement_Price': detail.get('Settlement_Price', ''),
                        'Quantity': detail.get('Quantity', '')
                    })
        
        detail_df = pd.DataFrame(detail_breakdown) if detail_breakdown else pd.DataFrame()
        
        # Generate summary statistics
        summary_stats = self.generate_summary_stats(results)
        
        # Prepare multiplier config for audit trail
        strike_mult_df = pd.DataFrame({
            'Product_Code': list(results['strike_multipliers'].keys()),
            'Multiplier': list(results['strike_multipliers'].values())
        })
        settle_mult_df = pd.DataFrame({
            'Product_Code': list(results['settle_multipliers'].keys()),
            'Multiplier': list(results['settle_multipliers'].values())
        })
        config_used = pd.concat([
            strike_mult_df.assign(Multiplier_Type='Strike'),
            settle_mult_df.assign(Multiplier_Type='Settlement')
        ], ignore_index=True)
        config_used = config_used[['Product_Code', 'Multiplier_Type', 'Multiplier']]
        
        # Export to Excel with multiple sheets
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            matched_display.to_excel(writer, sheet_name='Matched', index=False)
            unmatched_display.to_excel(writer, sheet_name='Unmatched_All', index=False)
            if len(detail_df) > 0:
                detail_df.to_excel(writer, sheet_name='Detail_Breakdown', index=False)
            summary_stats.to_excel(writer, sheet_name='Summary', index=False)
            config_used.to_excel(writer, sheet_name='Config_Used', index=False)
        
        # Apply formatting to Excel file
        try:
            from openpyxl import load_workbook
            from openpyxl.styles import PatternFill, Font
            
            wb = load_workbook(output_file)
            
            # Format Unmatched_All sheet
            if 'Unmatched_All' in wb.sheetnames:
                ws = wb['Unmatched_All']
                header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                header_font = Font(bold=True, color="FFFFFF")
                
                # Format header
                for cell in ws[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                
                # Color code rows
                for row_idx, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row), start=2):
                    status_cell = row[7]  # Status column
                    status = status_cell.value
                    
                    if status == 'XTP Only':
                        row[4].fill = PatternFill(start_color="FFE6E6", end_color="FFE6E6", fill_type="solid")
                    elif status == 'JPM Only':
                        row[5].fill = PatternFill(start_color="E6F3FF", end_color="E6F3FF", fill_type="solid")
                    elif status == 'Qty Mismatch':
                        row[6].fill = PatternFill(start_color="FFF4E6", end_color="FFF4E6", fill_type="solid")
            
            wb.save(output_file)
        except Exception as e:
            self.logger.warning(f"Could not apply Excel formatting: {e}")
        
        self.logger.info(f"Results exported successfully to: {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Position Reconciliation Tool: XTP vs JPM',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--xtp-file',
        type=str,
        default='input/1.XTP.xlsx',
        help='Path to XTP position file (default: input/1.XTP.xlsx)'
    )
    
    parser.add_argument(
        '--jpm-file',
        type=str,
        default='input/2.JPM.xlsx',
        help='Path to JPM position file (default: input/2.JPM.xlsx)'
    )
    
    parser.add_argument(
        '--strike-multiplier-file',
        type=str,
        default=None,
        help='Path to strike multipliers CSV (default: config/strike_multipliers.csv)'
    )
    
    parser.add_argument(
        '--settle-multiplier-file',
        type=str,
        default=None,
        help='Path to settlement multipliers CSV (default: config/settle_multipliers.csv)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    
    parser.add_argument(
        '--config-dir',
        type=str,
        default='config',
        help='Configuration directory (default: config)'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Log directory (default: logs)'
    )
    
    parser.add_argument(
        '--strike-tolerance',
        type=float,
        default=0.01,
        help='Tolerance for strike price comparisons (default: 0.01)'
    )
    
    parser.add_argument(
        '--qty-tolerance',
        type=float,
        default=0.0,
        help='Absolute tolerance for quantity differences (default: 0.0)'
    )
    
    parser.add_argument(
        '--qty-percent-tolerance',
        type=float,
        default=0.0,
        help='Percentage tolerance for quantity differences (default: 0.0)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize reconciliation tool
    tool = ReconciliationTool(
        config_dir=args.config_dir,
        log_dir=args.log_dir,
        strike_tolerance=args.strike_tolerance,
        qty_tolerance=args.qty_tolerance,
        qty_percent_tolerance=args.qty_percent_tolerance
    )
    
    try:
        # Perform reconciliation
        results = tool.reconcile(
            xtp_file=args.xtp_file,
            jpm_file=args.jpm_file,
            strike_mult_file=args.strike_multiplier_file,
            settle_mult_file=args.settle_multiplier_file
        )
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"reconciliation_{timestamp}.xlsx"
        
        # Export results
        tool.export_results(results, str(output_file))
        
        # Print summary
        summary = tool.generate_summary_stats(results)
        print("\n" + "=" * 80)
        print("RECONCILIATION SUMMARY")
        print("=" * 80)
        print(summary.to_string(index=False))
        print("\n" + "=" * 80)
        
        if results['products_with_default_multipliers']:
            print(f"\nWARNING: Products using default multipliers: {results['products_with_default_multipliers']}")
            print("Please update multiplier CSV files to include these products.")
        
        print(f"\nResults exported to: {output_file}")
        print("=" * 80)
        
    except Exception as e:
        tool.logger.error(f"Reconciliation failed: {e}", exc_info=True)
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
