import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import gc
from datetime import datetime
import warnings
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dask.dataframe as dd
import dask.array as da
import pyarrow as pa
import pyarrow.parquet as pq
import multiprocessing
import psutil
import swifter

# Set some display options
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Set chunk size based on available memory
def get_optimal_chunk_size():
    """Determine optimal chunk size based on available system memory"""
    mem_info = psutil.virtual_memory()
    available_memory_gb = mem_info.available / (1024 ** 3)
    # Use a fraction of available memory (30%) for each chunk
    chunk_size = int(available_memory_gb * 0.3 * 1e6)
    # Ensure chunk size is at least 100,000 rows and at most 5 million rows
    return max(100_000, min(5_000_000, chunk_size))

def load_parquet_files(directory, use_dask=True):
    """Load all parquet files from a directory efficiently"""
    print(f"Loading parquet files from {directory}...")
    
    # Get list of all parquet files
    parquet_files = glob(os.path.join(directory, '*.parquet'))
    print(f"Found {len(parquet_files)} parquet files")
    
    if len(parquet_files) == 0:
        print("No parquet files found")
        return None
    
    if use_dask:
        # Use Dask for out-of-memory processing
        print("Using Dask for distributed processing...")
        # Automatically determine optimal number of partitions
        cpu_count = multiprocessing.cpu_count()
        mem_info = psutil.virtual_memory()
        available_memory_gb = mem_info.available / (1024 ** 3)
        
        # Adjust partitions based on memory and CPU count
        n_partitions = max(cpu_count, min(len(parquet_files) * 2, int(cpu_count * 2)))
        
        try:
            # Try to infer schema to detect potential issues
            schema = pq.read_schema(parquet_files[0])
            print(f"Inferred schema with {len(schema.names)} columns")
            
            # Load data with dask
            ddf = dd.read_parquet(
                parquet_files,
                engine='pyarrow',
                calculate_divisions=False,
                split_row_groups=True
            )
            
            print(f"Loaded Dask DataFrame with {len(ddf.columns)} columns")
            print(f"Estimated size: {ddf.memory_usage(deep=True).sum().compute() / 1e9:.2f} GB")
            return ddf
            
        except Exception as e:
            print(f"Error loading with Dask: {str(e)}")
            print("Falling back to chunked pandas loading...")
            use_dask = False
    
    if not use_dask:
        # Use chunked pandas loading when Dask fails
        chunk_size = get_optimal_chunk_size()
        print(f"Using pandas with chunk size: {chunk_size:,} rows")
        
        # Initialize empty list for schema validation
        schemas = []
        
        # Check the schema of the first few files
        for file in parquet_files[:min(5, len(parquet_files))]:
            try:
                schema = pq.read_schema(file)
                schemas.append(set(schema.names))
            except Exception as e:
                print(f"Error reading schema from {file}: {str(e)}")
        
        # Find common columns across all schemas
        if schemas:
            common_columns = set.intersection(*schemas) if schemas else set()
            print(f"Found {len(common_columns)} common columns across sampled files")
        else:
            common_columns = None
        
        # Load files in chunks
        chunk_frames = []
        processed_rows = 0
        
        for file in tqdm(parquet_files, desc="Loading parquet files"):
            try:
                # Use PyArrow to read parquet and convert to pandas in chunks
                parquet_file = pq.ParquetFile(file)
                
                for batch in parquet_file.iter_batches(batch_size=chunk_size):
                    chunk = batch.to_pandas()
                    
                    # Filter to common columns if specified
                    if common_columns:
                        available_cols = set(chunk.columns) & common_columns
                        chunk = chunk[list(available_cols)]
                    
                    chunk_frames.append(chunk)
                    processed_rows += len(chunk)
                    
                    # Process in chunks to avoid memory issues
                    if len(chunk_frames) >= 5:
                        mini_df = pd.concat(chunk_frames, ignore_index=True)
                        chunk_frames = [mini_df]
                        
                        # Force garbage collection
                        gc.collect()
                        
                del batch
                del parquet_file
                gc.collect()
                
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
            
            # Progress update
            if processed_rows > 0 and processed_rows % (chunk_size * 10) == 0:
                print(f"Processed {processed_rows:,} rows so far...")
        
        # Combine all chunks
        if chunk_frames:
            try:
                combined_df = pd.concat(chunk_frames, ignore_index=True)
                print(f"Successfully loaded {len(combined_df):,} transactions")
                return combined_df
            except Exception as e:
                print(f"Error combining DataFrames: {str(e)}")
                return None
        else:
            print("No data loaded")
            return None

def check_data_quality(df, sample_size=None):
    """Check data quality by examining a sample of the data"""
    print("\n=== DATA QUALITY REPORT ===")
    
    if hasattr(df, 'dask'):
        # For Dask DataFrame
        print("Using Dask for data quality check")
        # Basic shape information
        print(f"Data shape: {df.shape[0].compute()} rows x {df.shape[1]} columns")
        
        # Get sample for detailed analysis
        if sample_size is None:
            sample_size = min(1_000_000, int(df.shape[0].compute() * 0.01))
        
        print(f"Taking a sample of {sample_size:,} rows for detailed analysis")
        sample_df = df.sample(frac=sample_size/df.shape[0].compute()).compute()
        
    else:
        # For Pandas DataFrame
        print(f"Data shape: {df.shape}")
        
        # Get sample for detailed analysis
        if sample_size is None:
            sample_size = min(1_000_000, int(len(df) * 0.01))
            
        print(f"Taking a sample of {sample_size:,} rows for detailed analysis")
        sample_df = df.sample(n=min(sample_size, len(df)))
    
    # Check data types
    print("\nData types:")
    print(sample_df.dtypes)
    
    # Check missing values
    print("\nMissing values (from sample):")
    missing = sample_df.isnull().sum()
    missing_percent = (sample_df.isnull().sum() / len(sample_df)) * 100
    missing_info = pd.DataFrame({
        'Count': missing,
        'Percentage': missing_percent
    }).sort_values('Percentage', ascending=False)
    print(missing_info[missing_info['Count'] > 0])
    
    # Check unique values for categorical columns (limited to sample)
    categorical_cols = sample_df.select_dtypes(include=['object', 'category']).columns
    print("\nUnique values for categorical columns (from sample):")
    for col in categorical_cols:
        unique_count = sample_df[col].nunique()
        print(f"{col}: {unique_count} unique values")
    
    # Check for duplicates in transaction_id (from sample)
    if 'transaction_id' in sample_df.columns:
        duplicate_count = sample_df.duplicated(subset=['transaction_id']).sum()
        print(f"\nDuplicate transaction IDs in sample: {duplicate_count} ({duplicate_count/len(sample_df)*100:.2f}%)")
    
    # Check for outliers in amount column (from sample)
    if 'amount' in sample_df.columns:
        print("\nAmount statistics (from sample):")
        print(sample_df['amount'].describe())
        
        # Calculate IQR for outlier detection
        q1 = sample_df['amount'].quantile(0.25)
        q3 = sample_df['amount'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = sample_df[(sample_df['amount'] < lower_bound) | (sample_df['amount'] > upper_bound)]
        print(f"Amount outliers in sample: {len(outliers)} ({len(outliers)/len(sample_df)*100:.2f}%)")
    
    del sample_df
    gc.collect()
    
    return missing_info[missing_info['Count'] > 0]

def clean_data(df, use_dask=True):
    """Clean the data by handling missing values, inconsistent formatting, etc."""
    print("\n=== CLEANING DATA ===")
    
    if use_dask and hasattr(df, 'dask'):
        print("Using Dask for data cleaning")
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = dd.to_datetime(df['timestamp'], errors='coerce')
            except:
                print("Could not convert timestamp column to datetime")
        
        # Handle missing values
        print("Handling missing values...")
        
        # For categorical columns, fill with 'unknown'
        str_cols = ['merchant_name', 'merchant_category', 'currency', 'location_country', 
                    'location_city', 'device_type', 'status']
        for col in str_cols:
            if col in df.columns:
                try:
                    # Convert to string type first to avoid categorical issues
                    df[col] = df[col].astype(str).fillna('unknown')
                except Exception as e:
                    print(f"Error processing column {col}: {str(e)}")
        
        # Handle missing numeric values
        if 'amount' in df.columns:
            # Flag missing amounts
            df['missing_amount'] = df['amount'].isna()
            # Fill missing amounts with 0
            df['amount'] = df['amount'].fillna(0)
        
        # Handle missing boolean values
        if 'is_online' in df.columns:
            df['is_online'] = df['is_online'].fillna(False)
        
        # Remove duplicates - note: this can be expensive in Dask and might trigger computation
        if 'transaction_id' in df.columns:
            print("Removing duplicate transactions...")
            # Use Dask's drop_duplicates which can be computed lazily
            df = df.drop_duplicates(subset=['transaction_id'], keep='first')
            
        # Handle outliers in amount - flag them but don't modify
        if 'amount' in df.columns:
            # Calculate quantiles
            q1 = df['amount'].quantile(0.25).compute()
            q3 = df['amount'].quantile(0.75).compute()
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Flag outliers
            df['is_amount_outlier'] = ((df['amount'] < lower_bound) | (df['amount'] > upper_bound))
            
            # Add capped amount column
            df['amount_capped'] = df['amount'].clip(lower=lower_bound, upper=upper_bound)
        
        # Repartition the data for better performance if needed
        # This reorganizes the data into more balanced partitions
        num_partitions = max(multiprocessing.cpu_count(), 16)
        df = df.repartition(npartitions=num_partitions)
        
    else:
        print("Using pandas for data cleaning")
        # Work with a copy to avoid modifying the original
        df = df.copy() if not hasattr(df, 'dask') else df.compute()
        
        # Process in chunks for large pandas DataFrames
        chunk_size = get_optimal_chunk_size()
        total_rows = len(df)
        chunks = np.array_split(df.index, max(1, total_rows // chunk_size))
        
        # Helper function to clean strings efficiently
        def clean_str_column(series):
            if series.dtype == 'object' or series.dtype == 'string':
                mask = series.notna()
                if mask.any():
                    series.loc[mask] = series.loc[mask].str.strip().str.lower()
            return series
        
        # Use parallel processing with swifter for string cleaning
        # Only process columns that exist in the DataFrame
        str_cols = ['merchant_name', 'merchant_category', 'currency', 'location_country', 
                    'location_city', 'device_type', 'status']
        existing_str_cols = [col for col in str_cols if col in df.columns]
        
        # Process string columns in smaller batches to avoid memory issues
        for col in existing_str_cols:
            print(f"Cleaning column: {col}")
            df[col] = clean_str_column(df[col])
            # Fill missing values
            df[col] = df[col].fillna('unknown')
        
        # Handle missing numeric values
        if 'amount' in df.columns:
            # Flag missing amounts
            df['missing_amount'] = df['amount'].isna()
            # Fill missing amounts with 0
            df['amount'] = df['amount'].fillna(0)
        
        # Handle missing boolean values
        if 'is_online' in df.columns:
            df['is_online'] = df['is_online'].fillna(False)
        
        # Remove duplicates - this can be memory intensive
        if 'transaction_id' in df.columns:
            print("Removing duplicate transactions...")
            before_drop = len(df)
            # For large DataFrames, we need to handle duplicates more carefully
            if before_drop > 10_000_000:  # If more than 10M rows
                # Create a flag for duplicates without creating an intermediate DataFrame
                dup_mask = df.duplicated(subset=['transaction_id'], keep='first')
                df = df[~dup_mask]
                del dup_mask
                gc.collect()
            else:
                df = df.drop_duplicates(subset=['transaction_id'], keep='first')
            
            after_drop = len(df)
            print(f"Removed {before_drop - after_drop} duplicate transactions")
        
        # Handle outliers in amount
        if 'amount' in df.columns:
            # Calculate IQR for outlier detection
            q1 = df['amount'].quantile(0.25)
            q3 = df['amount'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Flag outliers
            df['is_amount_outlier'] = (df['amount'] < lower_bound) | (df['amount'] > upper_bound)
            
            # For analysis, cap outliers rather than removing them
            print("Capping amount outliers...")
            df['amount_capped'] = df['amount'].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"Data cleaning complete")
    return df

def add_features(df, use_dask=True):
    """Add derived features to the dataframe"""
    print("\n=== ADDING DERIVED FEATURES ===")
    
    if use_dask and hasattr(df, 'dask'):
        print("Using Dask for feature generation")
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = dd.to_datetime(df['timestamp'], errors='coerce')
                
                # Extract time-based features
                print("Extracting time-based features...")
                df['date'] = df['timestamp'].dt.date
                df['year'] = df['timestamp'].dt.year
                df['month'] = df['timestamp'].dt.month
                df['day'] = df['timestamp'].dt.day
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['hour'] = df['timestamp'].dt.hour
                df['is_weekend'] = df['day_of_week'].isin([5, 6])
                
                # Create time period categories
                df['time_period'] = 'unknown'  # Default value
                
                # This approach is more efficient in Dask
                df['time_period'] = df['time_period'].mask(df['hour'] < 6, 'night')
                df['time_period'] = df['time_period'].mask((df['hour'] >= 6) & (df['hour'] < 12), 'morning')
                df['time_period'] = df['time_period'].mask((df['hour'] >= 12) & (df['hour'] < 18), 'afternoon')
                df['time_period'] = df['time_period'].mask(df['hour'] >= 18, 'evening')
                
                # Days since transaction - need a current date reference
                current_date_str = datetime.now().date().isoformat()
                df['days_since_transaction'] = (pd.Timestamp(current_date_str) - df['timestamp'].dt.floor('D')).dt.days
            except Exception as e:
                print(f"Error processing timestamp: {str(e)}")
        
        # Flag high-value transactions
        if 'amount' in df.columns:
            # Flag high-value transactions based on absolute threshold
            df['high_value'] = df['amount'] > 1000
            
            # Amount categories - use when() for better Dask performance
            df['amount_category'] = 'unknown'  # Default value
            df['amount_category'] = df['amount_category'].mask(df['amount'] <= 10, 'very_small')
            df['amount_category'] = df['amount_category'].mask((df['amount'] > 10) & (df['amount'] <= 50), 'small')
            df['amount_category'] = df['amount_category'].mask((df['amount'] > 50) & (df['amount'] <= 100), 'medium')
            df['amount_category'] = df['amount_category'].mask((df['amount'] > 100) & (df['amount'] <= 500), 'large')
            df['amount_category'] = df['amount_category'].mask((df['amount'] > 500) & (df['amount'] <= 1000), 'very_large')
            df['amount_category'] = df['amount_category'].mask(df['amount'] > 1000, 'huge')
            
            # Flag high-value online transactions
            if 'is_online' in df.columns:
                df['high_value_online'] = (df['amount'] > 1000) & df['is_online']
        
        # Note: For user-specific features like unusual_location, we'll need to compute these later
        # as they require aggregations which trigger computation in Dask
        
    else:
        print("Using pandas for feature generation")
        # Convert to pandas if it's a Dask DataFrame
        df = df.compute() if hasattr(df, 'dask') else df
        
        # Process in chunks for large pandas DataFrames
        chunk_size = get_optimal_chunk_size()
        total_rows = len(df)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Extract time-based features
            print("Extracting time-based features...")
            df['date'] = df['timestamp'].dt.date
            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['hour'] = df['timestamp'].dt.hour
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
            
            # Create time period categories
            conditions = [
                (df['hour'] < 6),
                (df['hour'] >= 6) & (df['hour'] < 12),
                (df['hour'] >= 12) & (df['hour'] < 18),
                (df['hour'] >= 18)
            ]
            choices = ['night', 'morning', 'afternoon', 'evening']
            df['time_period'] = np.select(conditions, choices, default='unknown')
            
            # Calculate transaction recency (days since transaction)
            current_date = pd.Timestamp(datetime.now().date())
            df['days_since_transaction'] = (current_date - pd.to_datetime(df['date'])).dt.days
        
        # Flag potential anomalies
        print("Flagging potential anomalies...")
        if 'amount' in df.columns and 'is_online' in df.columns:
            # Flag high-value online transactions
            df['high_value_online'] = (df['amount'] > 1000) & df['is_online']
            
            # Flag high-value transactions based on percentile
            # For large datasets, use approximate quantile
            if len(df) > 10_000_000:
                # Use sampling for large datasets
                sample_size = min(5_000_000, len(df))
                high_threshold = df['amount'].sample(sample_size).quantile(0.95)
            else:
                high_threshold = df['amount'].quantile(0.95)
                
            df['high_value'] = df['amount'] > high_threshold
            
            # Flag transactions with unusual locations
            if 'user_id' in df.columns and 'location_country' in df.columns:
                print("Calculating unusual locations...")
                
                # For large datasets, optimize this calculation
                if len(df) > 10_000_000:
                    # Calculate most common country by user more efficiently
                    country_counts = df.groupby(['user_id', 'location_country']).size().reset_index(name='count')
                    user_country = country_counts.sort_values(['user_id', 'count'], ascending=[True, False])
                    user_country = user_country.drop_duplicates('user_id')
                    user_country_dict = dict(zip(user_country['user_id'], user_country['location_country']))
                    
                    # Apply mapping to original dataframe in chunks
                    df['usual_country'] = df['user_id'].map(user_country_dict).fillna('unknown')
                    df['unusual_location'] = df['location_country'] != df['usual_country']
                    df.drop('usual_country', axis=1, inplace=True)
                else:
                    # For smaller datasets, use the original approach
                    user_country = df.groupby('user_id')['location_country'].agg(
                        lambda x: x.value_counts().index[0] if len(x) > 0 else 'unknown'
                    ).to_dict()
                    
                    # Apply function efficiently
                    df['unusual_location'] = df.apply(
                        lambda row: row['location_country'] != user_country.get(row['user_id'], 'unknown'),
                        axis=1
                    )
        
        # Add transaction value categories
        if 'amount' in df.columns:
            # Create amount categories more efficiently
            bins = [float('-inf'), 10, 50, 100, 500, 1000, float('inf')]
            labels = ['very_small', 'small', 'medium', 'large', 'very_large', 'huge']
            df['amount_category'] = pd.cut(df['amount'], bins=bins, labels=labels)
    
    # For very large DataFrames, we might need to use sampling for some features
    print("Feature generation complete")
    return df

def basic_analysis(df, use_dask=True):
    """Perform basic analysis on the transaction data"""
    print("\n=== BASIC ANALYSIS ===")
    
    # Detect if we have a Dask DataFrame
    is_dask = hasattr(df, 'dask')
    
    # For Dask DataFrames, we need to compute results
    if is_dask and use_dask:
        print("Using Dask for analysis (this may trigger computation)")
        
        # Calculate daily transaction volumes and values
        print("\nCalculating daily transaction metrics...")
        # This will trigger computation in Dask
        daily_metrics = df.groupby('date').agg({
            'transaction_id': 'count',
            'amount': ['sum', 'mean']
        }).compute()
        
        # Flatten the column hierarchy
        daily_metrics.columns = ['transaction_count', 'transaction_value', 'avg_transaction_value']
        daily_metrics = daily_metrics.reset_index()
        
        # For large datasets, limit the number of merchants/categories
        limit = 10
        
        # Identify top merchants by transaction count
        print(f"\nCalculating top {limit} merchants by transaction count...")
        top_merchants_count = df.groupby('merchant_name').size().compute()
        top_merchants_count = pd.Series(top_merchants_count).reset_index(name='transaction_count')
        top_merchants_count = top_merchants_count.nlargest(limit, 'transaction_count')
        
        # Identify top merchants by transaction value
        print(f"\nCalculating top {limit} merchants by transaction value...")
        merchants_value = df.groupby('merchant_name')['amount'].sum().compute()
        top_merchants_value = pd.Series(merchants_value).reset_index(name='total_value')
        top_merchants_value = top_merchants_value.nlargest(limit, 'total_value')
        
        # Compute hourly transaction patterns
        print("\nCalculating hourly transaction patterns...")
        hourly_pattern = df.groupby('hour').agg({
            'transaction_id': 'count',
            'amount': ['sum', 'mean']
        }).compute()
        
        # Flatten the column hierarchy
        hourly_pattern.columns = ['transaction_count', 'transaction_value', 'avg_transaction_value']
        hourly_pattern = hourly_pattern.reset_index()
        
        # Additional analysis by merchant category
        print("\nCalculating transaction metrics by merchant category...")
        category_metrics = df.groupby('merchant_category').agg({
            'transaction_id': 'count',
            'amount': ['sum', 'mean']
        }).compute()
        
        # Flatten the column hierarchy
        category_metrics.columns = ['transaction_count', 'transaction_value', 'avg_transaction_value']
        category_metrics = category_metrics.reset_index()
        category_metrics = category_metrics.nlargest(limit, 'transaction_count')
        
        # Online vs offline transactions
        print("\nCalculating online vs offline transaction metrics...")
        if 'is_online' in df.columns:
            online_metrics = df.groupby('is_online').agg({
                'transaction_id': 'count',
                'amount': ['sum', 'mean']
            }).compute()
            
            # Flatten the column hierarchy
            online_metrics.columns = ['transaction_count', 'transaction_value', 'avg_transaction_value']
            online_metrics = online_metrics.reset_index()
        else:
            online_metrics = pd.DataFrame()
        
    else:
        # Convert to pandas for analysis if needed
        if is_dask and not use_dask:
            print("Converting Dask DataFrame to pandas for analysis...")
            # Sample the data for analysis if it's too large
            if df.shape[0].compute() > 10_000_000:
                print("Sampling 10M rows for analysis...")
                df = df.sample(frac=10_000_000/df.shape[0].compute()).compute()
            else:
                df = df.compute()
        
        # For large pandas DataFrames, sample if needed
        if not is_dask and len(df) > 10_000_000:
            print("Dataset too large, sampling 10M rows for analysis...")
            df_sample = df.sample(n=10_000_000)
        else:
            df_sample = df
        
        # Calculate daily transaction volumes and values
        print("\nCalculating daily transaction metrics...")
        daily_metrics = df_sample.groupby('date').agg(
            transaction_count=('transaction_id', 'count'),
            transaction_value=('amount', 'sum'),
            avg_transaction_value=('amount', 'mean')
        ).reset_index()
        
        # Identify top 10 merchants by transaction count
        print("\nCalculating top 10 merchants by transaction count...")
        top_merchants_count = df_sample.groupby('merchant_name').size().reset_index(name='transaction_count')
        top_merchants_count = top_merchants_count.sort_values('transaction_count', ascending=False).head(10)
        
        # Identify top 10 merchants by transaction value
        print("\nCalculating top 10 merchants by transaction value...")
        top_merchants_value = df_sample.groupby('merchant_name').agg({'amount': 'sum'}).reset_index()
        top_merchants_value = top_merchants_value.rename(columns={'amount': 'total_value'})
        top_merchants_value = top_merchants_value.sort_values('total_value', ascending=False).head(10)
        
        # Compute hourly transaction patterns
        print("\nCalculating hourly transaction patterns...")
        hourly_pattern = df_sample.groupby('hour').agg(
            transaction_count=('transaction_id', 'count'),
            transaction_value=('amount', 'sum'),
            avg_transaction_value=('amount', 'mean')
        ).reset_index()
        
        # Additional analysis by merchant category
        print("\nCalculating transaction metrics by merchant category...")
        category_metrics = df_sample.groupby('merchant_category').agg(
            transaction_count=('transaction_id', 'count'),
            transaction_value=('amount', 'sum'),
            avg_transaction_value=('amount', 'mean')
        ).reset_index().sort_values('transaction_count', ascending=False).head(10)
        
        # Online vs offline transactions
        print("\nCalculating online vs offline transaction metrics...")
        if 'is_online' in df_sample.columns:
            online_metrics = df_sample.groupby('is_online').agg(
                transaction_count=('transaction_id', 'count'),
                transaction_value=('amount', 'sum'),
                avg_transaction_value=('amount', 'mean')
            ).reset_index()
        else:
            online_metrics = pd.DataFrame()

        # Clear sample from memory if it was created
        if df_sample is not df:
            del df_sample
            gc.collect()
    
    # Return the results for visualization
    return {
        'daily_metrics': daily_metrics,
        'top_merchants_count': top_merchants_count,
        'top_merchants_value': top_merchants_value,
        'hourly_pattern': hourly_pattern,
        'category_metrics': category_metrics,
        'online_metrics': online_metrics
    }

def visualize_results(analysis_results, output_dir="output_visualizations"):
    """Create visualizations based on the analysis results"""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract results from the analysis
    daily_metrics = analysis_results.get('daily_metrics')
    top_merchants_count = analysis_results.get('top_merchants_count')
    top_merchants_value = analysis_results.get('top_merchants_value')
    hourly_pattern = analysis_results.get('hourly_pattern')
    category_metrics = analysis_results.get('category_metrics')
    online_metrics = analysis_results.get('online_metrics')
    
    # Set figure size for all plots
    plt.figure(figsize=(12, 8))
    
    # 1. Daily Transaction Volume and Value
    if daily_metrics is not None and not daily_metrics.empty:
        print("Creating daily transaction volume and value chart...")
        
        # Ensure date column is in proper datetime format
        if not pd.api.types.is_datetime64_any_dtype(daily_metrics['date']):
            daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])
        
        # Sort by date to ensure proper timeline
        daily_metrics = daily_metrics.sort_values('date')
        
        # Create the plot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add transaction count trace
        fig.add_trace(
            go.Scatter(
                x=daily_metrics['date'], 
                y=daily_metrics['transaction_count'],
                name="Transaction Count", 
                mode="lines+markers",
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            secondary_y=False,
        )
        
        # Add transaction value trace
        fig.add_trace(
            go.Scatter(
                x=daily_metrics['date'], 
                y=daily_metrics['transaction_value'],
                name="Transaction Value", 
                mode="lines+markers",
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ),
            secondary_y=True,
        )
        
        # Update layout with more detailed formatting
        fig.update_layout(
            title_text="Daily Transaction Volume and Value",
            title_x=0.5,
            xaxis_title="Date",
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white",
            hovermode="x unified"
        )
        
        # Format axes
        fig.update_yaxes(
            title_text="Transaction Count", 
            secondary_y=False,
            gridcolor='lightgray'
        )
        
        fig.update_yaxes(
            title_text="Transaction Value", 
            secondary_y=True,
            gridcolor='lightgray'
        )
        
        # Improve x-axis date formatting
        fig.update_xaxes(
            tickformat="%b %d, %Y",
            tickangle=-45,
            tickmode="auto",
            nticks=20
        )
        
        # Write to file
        fig.write_html(os.path.join(output_dir, "daily_transactions.html"))
        
    # 2. Top Merchants by Transaction Count
    if top_merchants_count is not None and not top_merchants_count.empty:
        print("Creating top merchants by count chart...")
        fig = px.bar(top_merchants_count, y='merchant_name', x='transaction_count', 
                    title='Top 10 Merchants by Transaction Count',
                    labels={'transaction_count': 'Number of Transactions', 'merchant_name': 'Merchant Name'},
                    orientation='h',
                    height=600)
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        fig.write_html(os.path.join(output_dir, "top_merchants_count.html"))
    
    # 3. Top Merchants by Transaction Value
    if top_merchants_value is not None and not top_merchants_value.empty:
        print("Creating top merchants by value chart...")
        fig = px.bar(top_merchants_value, y='merchant_name', x='total_value', 
                    title='Top 10 Merchants by Transaction Value',
                    labels={'total_value': 'Total Transaction Value', 'merchant_name': 'Merchant Name'},
                    orientation='h',
                    height=600)
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        fig.write_html(os.path.join(output_dir, "top_merchants_value.html"))
    
    # 4. Hourly Transaction Pattern
    if hourly_pattern is not None and not hourly_pattern.empty:
        print("Creating hourly transaction pattern chart...")
        
        # Ensure hour is sorted properly
        hourly_pattern = hourly_pattern.sort_values('hour')
        
        # Create the subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add transaction count bars with improved styling
        fig.add_trace(
            go.Bar(
                x=hourly_pattern['hour'], 
                y=hourly_pattern['transaction_count'],
                name="Transaction Count",
                marker=dict(
                    color='rgba(58, 71, 80, 0.7)',
                    line=dict(color='rgba(58, 71, 80, 1.0)', width=1)
                ),
                hovertemplate='Hour: %{x}<br>Transaction Count: %{y:,.0f}<extra></extra>'
            ),
            secondary_y=False,
        )
        
        # Add average transaction value line with improved styling
        fig.add_trace(
            go.Scatter(
                x=hourly_pattern['hour'], 
                y=hourly_pattern['avg_transaction_value'],
                name="Avg Transaction Value", 
                mode="lines+markers",
                line=dict(color='rgb(200, 30, 30)', width=3),
                marker=dict(size=8, symbol='circle', color='rgb(200, 30, 30)', 
                            line=dict(color='rgb(0, 0, 0)', width=1)),
                hovertemplate='Hour: %{x}<br>Avg Value: $%{y:.2f}<extra></extra>'
            ),
            secondary_y=True,
        )
        
        # Update layout with better styling
        fig.update_layout(
            title={
                'text': "Hourly Transaction Pattern",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24)
            },
            xaxis={
                'title': "Hour of Day",
                'tickmode': 'linear',
                'tick0': 0,
                'dtick': 1,
                'tickangle': 0,
                'gridcolor': 'lightgray'
            },
            height=600,
            width=1000,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        # Update y-axes with better formatting
        fig.update_yaxes(
            title_text="Transaction Count",
            secondary_y=False,
            gridcolor='lightgray',
            tickformat=',d'  # Format large numbers with commas
        )
        
        fig.update_yaxes(
            title_text="Average Transaction Value ($)",
            secondary_y=True,
            gridcolor='lightgray',
            tickprefix='$',  # Add dollar sign prefix
            tickformat='.2f'  # Format to 2 decimal places
        )
        
        # Add annotations for peak times if desired
        peak_hour = hourly_pattern.loc[hourly_pattern['transaction_count'].idxmax()]
        fig.add_annotation(
            x=peak_hour['hour'],
            y=peak_hour['transaction_count'],
            text=f"Peak: {int(peak_hour['transaction_count']):,}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='black',
            ax=20,
            ay=-40,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1,
            borderpad=4,
            font=dict(size=12)
        )
        
        # Write to file
        fig.write_html(os.path.join(output_dir, "hourly_pattern.html"))
        print("Hourly transaction pattern chart created successfully")
    
    # 5. Merchant Category Analysis
    if category_metrics is not None and not category_metrics.empty:
        print("Creating merchant category analysis chart...")
        fig = px.bar(category_metrics, y='merchant_category', x='transaction_count', 
                    title='Top Merchant Categories by Transaction Count',
                    labels={'transaction_count': 'Number of Transactions', 'merchant_category': 'Merchant Category'},
                    orientation='h',
                    height=600)
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        fig.write_html(os.path.join(output_dir, "category_metrics.html"))
        
        # Also create a pie chart for transaction value distribution by category
        fig = px.pie(category_metrics, values='transaction_value', names='merchant_category',
                    title='Transaction Value Distribution by Merchant Category',
                    height=600)
        fig.write_html(os.path.join(output_dir, "category_value_distribution.html"))
    
    # 6. Online vs Offline Transactions
    if online_metrics is not None and not online_metrics.empty and len(online_metrics) > 1:
        print("Creating online vs offline transactions chart...")
        online_metrics['transaction_type'] = online_metrics['is_online'].map({True: 'Online', False: 'Offline'})
        
        # Create bar chart
        fig = px.bar(online_metrics, x='transaction_type', y=['transaction_count', 'transaction_value'],
                    title='Online vs Offline Transactions',
                    barmode='group',
                    height=600)
        fig.write_html(os.path.join(output_dir, "online_offline_comparison.html"))
        
        # Create pie charts
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                            subplot_titles=('Transaction Count', 'Transaction Value'))
        
        fig.add_trace(go.Pie(labels=online_metrics['transaction_type'], values=online_metrics['transaction_count'],
                            name="Transaction Count"), 1, 1)
        
        fig.add_trace(go.Pie(labels=online_metrics['transaction_type'], values=online_metrics['transaction_value'],
                            name="Transaction Value"), 1, 2)
        
        fig.update_layout(title_text="Online vs Offline Transaction Distribution", height=600)
        fig.write_html(os.path.join(output_dir, "online_offline_distribution.html"))
    
    print(f"Visualizations saved to {output_dir} directory")
    return True

def process_transaction_data(input_dir="data", output_dir="processed_chunks", 
                             use_dask=True, visualize=True, save_results=True):
    """Main function to process transaction data end-to-end"""
    print("\n=== STARTING TRANSACTION DATA ANALYSIS ===")
    print(f"Processing data from {input_dir}")
    
    # Start the timer for performance tracking
    start_time = datetime.now()
    print(f"Start time: {start_time}")
    
    # 1. Load data
    df = load_parquet_files(input_dir, use_dask=use_dask)
    
    if df is None:
        print("No data loaded. Process terminated.")
        return None
    
    # 2. Check data quality
    quality_report = check_data_quality(df)
    
    # 3. Clean data
    df_cleaned = clean_data(df, use_dask=use_dask)
    
    # 4. Add features
    df_enriched = add_features(df_cleaned, use_dask=use_dask)
    
    # 5. Basic analysis
    analysis_results = basic_analysis(df_enriched, use_dask=use_dask)
    
    # 6. Visualize results
    if visualize:
        visualize_results(analysis_results, output_dir=os.path.join(output_dir, "visualizations"))

    
    # Calculate execution time
    end_time = datetime.now()
    execution_time = end_time - start_time
    print(f"\nAnalysis completed in {execution_time}")
    print(f"End time: {end_time}")
    
    return analysis_results

if __name__ == "__main__":
    # You can customize these parameters based on your environment
    input_directory = "processed_chunks"  # Directory containing parquet files
    output_directory = "analysis_output"  # Directory to save results
    use_dask_processing = True  # Set to False for smaller datasets or if Dask causes issues
    
    # Parse command line arguments if provided
    import argparse
    parser = argparse.ArgumentParser(description="Transaction Data Analysis")
    parser.add_argument("--input", "-i", default=input_directory, help="Input directory with parquet files")
    parser.add_argument("--output", "-o", default=output_directory, help="Output directory for results")
    parser.add_argument("--no-dask", action="store_true", help="Disable Dask processing")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization generation")
    parser.add_argument("--no-save", action="store_true", help="Don't save processed data")
    
    args = parser.parse_args()
    
    # Execute the processing pipeline
    results = process_transaction_data(
        input_dir=args.input,
        output_dir=args.output,
        use_dask=not args.no_dask,
        visualize=not args.no_viz,
        save_results=not args.no_save
    )
    
    print("Analysis complete!")