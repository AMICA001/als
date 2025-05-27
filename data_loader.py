import pandas as pd
import io

# Default Raw data as a multiline string (can be overridden by passing data_str to the function)
DEFAULT_DATA = """maturity/delta vol	1	10	20	30	40	50	60	70	80	90	99
25-Jul-25	29.67%	24.10%	21.63%	19.72%	17.96%	16.32%	14.83%	13.63%	12.84%	12.57%	15.71%
31-Jul-25	30.02%	24.34%	21.85%	19.92%	18.14%	16.45%	14.92%	13.68%	12.87%	12.60%	15.81%
15-Aug-25	30.68%	24.76%	22.23%	20.27%	18.44%	16.68%	15.05%	13.71%	12.81%	12.51%	15.67%
29-Aug-25	31.10%	25.09%	22.54%	20.55%	18.70%	16.90%	15.20%	13.75%	12.74%	12.39%	15.70%
19-Sep-25	31.72%	25.53%	22.95%	20.94%	19.05%	17.19%	15.40%	13.85%	12.72%	12.38%	15.76%
30-Sep-25	31.88%	25.64%	23.05%	21.04%	19.16%	17.29%	15.46%	13.84%	12.67%	12.29%	15.78%
17-Oct-25	32.20%	25.82%	23.21%	21.21%	19.33%	17.44%	15.55%	13.85%	12.62%	12.25%	15.67%
31-Oct-25	32.41%	25.98%	23.34%	21.34%	19.46%	17.57%	15.64%	13.87%	12.59%	12.24%	15.74%
21-Nov-25	32.65%	26.13%	23.47%	21.45%	19.57%	17.68%	15.71%	13.88%	12.53%	12.10%	15.72%
19-Dec-25	32.87%	26.28%	23.59%	21.56%	19.67%	17.78%	15.80%	13.93%	12.54%	12.06%	15.58%
"""

def load_and_prepare_volatility_data(data_str: str = None) -> pd.DataFrame:
    """
    Loads the options volatility data from a string, processes it, 
    and returns a Pandas DataFrame.

    Args:
        data_str: Optional. A multiline string containing the volatility data.
                  If None, uses DEFAULT_DATA.
    Returns:
        A Pandas DataFrame with processed volatility data.
    """
    input_data = data_str if data_str is not None else DEFAULT_DATA
    
    # Use io.StringIO to read the string data as if it were a file
    data_io = io.StringIO(input_data)
    
    # Read the data, using the first column as index
    # Specify engine='python' for robust parsing of tab-separated values
    df = pd.read_csv(data_io, sep='\\t', engine='python', index_col=0)
    
    # Parse the index (maturity dates) as datetime objects
    df.index = pd.to_datetime(df.index, format='%d-%b-%y')
    df.index.name = 'Maturity'
    
    # Convert volatility percentage strings to numerical format (e.g., 0.2967)
    for col in df.columns:
        # Remove '%' and convert to float, then divide by 100
        df[col] = df[col].astype(str).str.rstrip('%').astype('float') / 100.0
        
    # Ensure column names (deltas) are appropriate
    df.columns.name = 'Delta'
    
    return df

if __name__ == "__main__":
    # Test with default data
    print("--- Testing with Default Data ---")
    vol_df_default = load_and_prepare_volatility_data()
    print("DataFrame Head:")
    print(vol_df_default.head())
    print("\nDataFrame Info:")
    vol_df_default.info()

    # Test with custom data string
    print("\n--- Testing with Custom Data String ---")
    custom_data = """maturity/delta vol	1	10
01-Jan-26	10.00%	12.00%
01-Feb-26	11.00%	13.00%
"""
    vol_df_custom = load_and_prepare_volatility_data(data_str=custom_data)
    print("DataFrame Head:")
    print(vol_df_custom.head())
    print("\nDataFrame Info:")
    vol_df_custom.info()
    print("\nSample specific value (01-Jan-26, Delta 1):")
    print(vol_df_custom.loc[pd.to_datetime('2026-01-01'), '1'])

"""
Specific tasks checklist:
1.  Represent the data as a Pandas DataFrame. - Done
2.  The first column 'maturity/delta vol' should be used as the index and parsed as datetime objects. - Done
3.  The header row (1, 10, 20, ..., 99) should be the column names, representing deltas. - Done
4.  All volatility values (e.g., "29.67%") should be converted to numerical format (e.g., 0.2967). - Done
5.  Save the script as 'data_loader.py'. - Done
6.  The script should define a function that performs these loading and preparation steps and returns the DataFrame. - Done
7.  Include a simple test or print statement in the script to display the head and info of the loaded DataFrame to verify correctness. - Done
8.  Modified load_and_prepare_volatility_data to accept data_str argument. - Done
"""
