#!/usr/bin/env python3
"""
Transaction Data Acquisition Script for Bicycle Implementation Engineer Technical Assignment

This script:
1. Downloads daily transaction data from GCP bucket
2. Implements checkpointing to resume interrupted downloads
3. Validates downloaded files
4. Provides a data summary for each file

Requirements:
- Google Cloud SDK installed and configured
- Python 3.7+
- Required packages: pandas, tqdm
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import subprocess
from tqdm import tqdm
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='data_acquisition.log',
    filemode='a'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Constants
PROJECT_ID = "pivotal-canto-171605"
BUCKET_NAME = "implemention_engineer_interviews"
LOCAL_DATA_DIR = "./transaction_data"
CHECKPOINT_FILE = "download_checkpoint.json"

# Define a date range that covers 30 days of data
# We'll assume the data is for April 2023, but this can be adjusted
START_DATE = "2025-04-07"
END_DATE = "2025-05-07"
gsutil_path = "C:\\Users\\LENOVO\\AppData\\Local\\Google\\Cloud SDK\\google-cloud-sdk\\bin\\gsutil.cmd"

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def generate_date_range(start_date, end_date):
    """Generate a list of date strings in format YYYY-MM-DD between the given dates"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_list = []
    
    current = start
    while current <= end:
        date_list.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    return date_list

def load_checkpoint():
    """Load checkpoint information from file"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logging.warning("Checkpoint file is missing or corrupted. Starting fresh.")
            
    return {"completed": [], "validated": []}

def save_checkpoint(checkpoint):
    """Save checkpoint information to file"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of a file for validation"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_reference_files():
    """
    Download reference files from the bucket:
    - merchants_reference.csv
    - generation_summary.json
    """
    
    # Create a reference directory
    reference_dir = os.path.join(LOCAL_DATA_DIR, "reference")
    ensure_directory(reference_dir)
    
    # Download merchants_reference.csv
    merchants_source = f"gs://{BUCKET_NAME}/transaction_data/merchants_reference.csv"
    merchants_dest = os.path.join(reference_dir, "merchants_reference.csv")
    
    try:
        logging.info(f"Downloading merchants reference file from {merchants_source}")
        subprocess.run([gsutil_path, "cp", merchants_source, merchants_dest], check=True)
        logging.info("Successfully downloaded merchants reference file")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download merchants reference file: {e}")
    
    # Download generation_summary.json
    summary_source = f"gs://{BUCKET_NAME}/transaction_data/generation_summary.json"
    summary_dest = os.path.join(reference_dir, "generation_summary.json")
    
    try:
        logging.info(f"Downloading generation summary file from {summary_source}")
        subprocess.run([gsutil_path, "cp", summary_source, summary_dest], check=True)
        logging.info("Successfully downloaded generation summary file")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download generation summary file: {e}")

def download_file(date):
    """
    Download transaction file for a specific date
    Data is organized by date: /yyyy-mm-dd/transactions.csv in the GCP bucket
    """
    source_path = f"gs://{BUCKET_NAME}/transaction_data/{date}/transactions.csv"
    dest_dir = os.path.join(LOCAL_DATA_DIR, date)
    dest_path = os.path.join(dest_dir, "transactions.csv")
    
    ensure_directory(dest_dir)
    
    # Use gsutil to download the file from GCP bucket
    
    transactions_success = False
    try:
        logging.info(f"Downloading transactions file from {source_path}")
        subprocess.run([gsutil_path, "cp", source_path, dest_path], check=True)
        logging.info(f"Successfully downloaded transactions data for {date}")
        transactions_success = True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download transactions data for {date}: {e}")
    
    # Download metadata
    metadata_source = f"gs://{BUCKET_NAME}/transaction_data/{date}/metadata.json"
    metadata_dest = os.path.join(dest_dir, "metadata.json")
    
    try:
        logging.info(f"Downloading metadata file from {metadata_source}")
        subprocess.run([gsutil_path, "cp", metadata_source, metadata_dest], check=True)
        logging.info(f"Successfully downloaded metadata for {date}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download metadata for {date}: {e}")
    
    # Return True only if transaction data was successfully downloaded
    # Since that's what the validation function checks
    return transactions_success

def validate_file(date):
    """Validate that the downloaded file is complete and usable"""
    file_path = os.path.join(LOCAL_DATA_DIR, date, "transactions.csv")
    
    if not os.path.exists(file_path):
        logging.error(f"File does not exist: {file_path}")
        return False
        
    file_size = os.path.getsize(file_path)
    if file_size < 1000:  # Files should be ~130-140MB according to the assignment
        logging.error(f"File is too small ({file_size} bytes): {file_path}")
        return False
    
    try:
        # Try to read a few rows to ensure the file is valid CSV
        df = pd.read_csv(file_path, nrows=5)
        if len(df) > 0:
            file_hash = calculate_file_hash(file_path)
            logging.info(f"Validated file for {date} (Hash: {file_hash[:8]}...)")
            return True
        else:
            logging.error(f"File contains no data rows: {file_path}")
            return False
    except Exception as e:
        logging.error(f"File validation failed for {date}: {e}")
        return False

def analyze_file(date):
    """
    Analyze the downloaded file and print summary
    The summary includes:
    - rows per file
    - data type for each column
    - number of distinct values vs total values for each column
    - Top & Bottom 10 values by count
    - Top & Bottom 10 values lexicographically along with their counts
    """
    file_path = os.path.join(LOCAL_DATA_DIR, date, "transactions.csv")
    
    try:
        logging.info(f"Analyzing file: {file_path}")
        
        # Count rows without loading entire file into memory
        row_count = 0
        for chunk in pd.read_csv(file_path, chunksize=100000):
            row_count += len(chunk)
        
        # Load sample of data to analyze columns
        sample_df = pd.read_csv(file_path, nrows=100000)
        
        print(f"\n{'='*80}")
        print(f"DATA SUMMARY FOR {date}")
        print(f"{'='*80}")
        print(f"Total rows: {row_count}")
        print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        
        # Column analysis using the sample
        for column in sample_df.columns:
            print(f"\nColumn: {column}")
            print(f"  Data type: {sample_df[column].dtype}")
            
            # Calculate distinct vs total values
            distinct_values = sample_df[column].nunique()
            total_values = len(sample_df[column].dropna())
            print(f"  Distinct values: {distinct_values} / {total_values} ({distinct_values/total_values*100:.2f}% unique)")
            
            # Value counts analysis for object and category types
            if sample_df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(sample_df[column]):
                value_counts = sample_df[column].value_counts()
                
                print(f"  Top 10 values by count:")
                for val, count in value_counts.head(10).items():
                    print(f"    {val}: {count}")
                
                print(f"  Bottom 10 values by count:")
                for val, count in value_counts.tail(10).items():
                    print(f"    {val}: {count}")
                
                # Sort values lexicographically
                lex_values = pd.Series(value_counts.index).sort_values()
                lex_counts = [value_counts[val] for val in lex_values]
                
                print(f"  Top 10 values lexicographically:")
                for val, count in zip(lex_values[:10], lex_counts[:10]):
                    print(f"    {val}: {value_counts[val]}")
                
                print(f"  Bottom 10 values lexicographically:")
                for val, count in zip(lex_values[-10:], lex_counts[-10:]):
                    print(f"    {val}: {value_counts[val]}")
            
            # For numeric columns, show min, max, mean, median
            elif pd.api.types.is_numeric_dtype(sample_df[column]):
                print(f"  Numeric statistics:")
                print(f"    Min: {sample_df[column].min()}")
                print(f"    Max: {sample_df[column].max()}")
                print(f"    Mean: {sample_df[column].mean()}")
                print(f"    Median: {sample_df[column].median()}")
                    
        logging.info(f"Completed analysis for {date}")
        return True
        
    except Exception as e:
        logging.error(f"Analysis failed for {date}: {e}")
        return False

def main():
    """Main execution function"""
    logging.info("Starting transaction data acquisition process")
    
    ensure_directory(LOCAL_DATA_DIR)
    
    download_reference_files()

    checkpoint = load_checkpoint()
    date_list = generate_date_range(START_DATE, END_DATE)
    
    # Download missing files concurrently
    with ThreadPoolExecutor() as executor:
        download_futures = {
            executor.submit(download_file, date): date
            for date in date_list if date not in checkpoint["completed"]
        }
        for future in tqdm(as_completed(download_futures), total=len(download_futures), desc="Downloading transaction data"):
            date = download_futures[future]
            try:
                if future.result():
                    checkpoint["completed"].append(date)
                    save_checkpoint(checkpoint)
            except Exception as e:
                logging.error(f"Error downloading data for {date}: {e}")
    
    # Validate files concurrently
    with ThreadPoolExecutor() as executor:
        validate_futures = {
            executor.submit(validate_file, date): date
            for date in date_list if date not in checkpoint["validated"] and date in checkpoint["completed"]
        }
        for future in tqdm(as_completed(validate_futures), total=len(validate_futures), desc="Validating downloaded files"):
            date = validate_futures[future]
            try:
                if future.result():
                    checkpoint["validated"].append(date)
                    save_checkpoint(checkpoint)
            except Exception as e:
                logging.error(f"Error validating data for {date}: {e}")
    
    # Analyze validated files concurrently
    with ThreadPoolExecutor() as executor:
        analyze_futures = {
            executor.submit(analyze_file, date): date
            for date in checkpoint["validated"]
        }
        for future in tqdm(as_completed(analyze_futures), total=len(analyze_futures), desc="Analyzing validated files"):
            date = analyze_futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error analyzing data for {date}: {e}")
    
    logging.info("Data acquisition process completed")

if __name__ == "__main__":
    main()