import numpy as np
from als import impute_missing_values  # Updated import
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge # Though default, useful for explicit testing or if default changes

# Define a common dataset for tests
SAMPLE_DATA = [
    [1, 2, np.nan, 4],
    [np.nan, 5, 6, np.nan],
    [7, np.nan, 9, 10],
    [11, 12, np.nan, 14]
]

SAMPLE_DATA_SIMPLE_ALS = [ # Original data for ALS test had different dimensions
    [1, 2, np.nan],
    [np.nan, 5, 6],
    [7, np.nan, 9]
]

def test_als_imputation_simple():
    """
    Test impute_missing_values with method='ALS' on a simple case.
    Checks shape, absence of NaNs, and approximate preservation of original values.
    """
    data = [row[:] for row in SAMPLE_DATA_SIMPLE_ALS] # Use a copy
    original_data_np = np.array(data)
    expected_shape = original_data_np.shape

    # Run ALS imputation
    # Using a small number of iterations and a higher tolerance for a quick test
    completed_matrix = impute_missing_values(data, method='ALS', rank=2, max_iter=10, tol=0.1)

    assert completed_matrix.shape == expected_shape, \
        f"ALS: Expected shape {expected_shape}, but got {completed_matrix.shape}"

    assert not np.isnan(completed_matrix).any(), \
        "ALS: Completed matrix should not contain NaN values"

    # Check if the original non-NaN values are approximately preserved.
    # ALS reconstructs the entire matrix, so original values might change slightly.
    mask = ~np.isnan(original_data_np)
    assert np.allclose(completed_matrix[mask], original_data_np[mask], atol=1.5), \
        "ALS: Original non-NaN values were not preserved sufficiently."

    # Check that NaN values have been filled
    nan_mask_original = np.isnan(original_data_np)
    assert not np.isnan(completed_matrix[nan_mask_original]).any(), \
        "ALS: NaN values in the original matrix were not all filled."


def test_mice_imputation_default_regressor():
    """
    Test impute_missing_values with method='MICE' using the default BayesianRidge regressor.
    Checks shape, absence of NaNs, and exact preservation of original non-NaN values.
    """
    data = [row[:] for row in SAMPLE_DATA] # Use a copy
    original_data_np = np.array(data)
    expected_shape = original_data_np.shape

    completed_matrix = impute_missing_values(data, method='MICE')

    assert completed_matrix.shape == expected_shape, \
        f"MICE (default): Expected shape {expected_shape}, but got {completed_matrix.shape}"

    assert not np.isnan(completed_matrix).any(), \
        "MICE (default): Completed matrix should not contain NaN values"

    # For MICE, original non-NaN values should be exactly preserved.
    mask = ~np.isnan(original_data_np)
    assert np.array_equal(completed_matrix[mask], original_data_np[mask]), \
        "MICE (default): Original non-NaN values were not exactly preserved."

    # Check that NaN values have been filled
    nan_mask_original = np.isnan(original_data_np)
    assert not np.isnan(completed_matrix[nan_mask_original]).any(), \
        "MICE (default): NaN values in the original matrix were not all filled."


def test_mice_imputation_custom_regressor():
    """
    Test impute_missing_values with method='MICE' using a custom DecisionTreeRegressor.
    Checks shape, absence of NaNs, and exact preservation of original non-NaN values.
    """
    data = [row[:] for row in SAMPLE_DATA] # Use a copy
    original_data_np = np.array(data)
    expected_shape = original_data_np.shape

    # Using DecisionTreeRegressor with random_state for reproducibility
    estimator = DecisionTreeRegressor(random_state=0)
    completed_matrix = impute_missing_values(data, method='MICE', estimator=estimator)

    assert completed_matrix.shape == expected_shape, \
        f"MICE (custom): Expected shape {expected_shape}, but got {completed_matrix.shape}"

    assert not np.isnan(completed_matrix).any(), \
        "MICE (custom): Completed matrix should not contain NaN values"

    # For MICE, original non-NaN values should be exactly preserved.
    mask = ~np.isnan(original_data_np)
    assert np.array_equal(completed_matrix[mask], original_data_np[mask]), \
        "MICE (custom): Original non-NaN values were not exactly preserved."
    
    # Check that NaN values have been filled
    nan_mask_original = np.isnan(original_data_np)
    assert not np.isnan(completed_matrix[nan_mask_original]).any(), \
        "MICE (custom): NaN values in the original matrix were not all filled."

def test_mice_imputation_single_column():
    """
    Test MICE imputation with a single column of data.
    """
    data_single_col = [[1], [np.nan], [3], [4], [np.nan], [6]]
    original_data_np = np.array(data_single_col)
    expected_shape = original_data_np.shape

    completed_matrix = impute_missing_values(data_single_col, method='MICE', estimator=BayesianRidge())

    assert completed_matrix.shape == expected_shape, \
        f"MICE (single_col): Expected shape {expected_shape}, but got {completed_matrix.shape}"
    assert not np.isnan(completed_matrix).any(), \
        "MICE (single_col): Completed matrix should not contain NaN values"
    
    mask = ~np.isnan(original_data_np)
    assert np.array_equal(completed_matrix[mask], original_data_np[mask]), \
        "MICE (single_col): Original non-NaN values were not exactly preserved."

if __name__ == "__main__":
    # This allows running tests directly, e.g., python test_als.py
    # For more comprehensive testing, use pytest.
    np.set_printoptions(precision=4, suppress=True) # For cleaner output if any manual checks are done

    print("Running test_als_imputation_simple...")
    test_als_imputation_simple()
    print("test_als_imputation_simple PASSED")

    print("\nRunning test_mice_imputation_default_regressor...")
    test_mice_imputation_default_regressor()
    print("test_mice_imputation_default_regressor PASSED")

    print("\nRunning test_mice_imputation_custom_regressor...")
    test_mice_imputation_custom_regressor()
    print("test_mice_imputation_custom_regressor PASSED")
    
    print("\nRunning test_mice_imputation_single_column...")
    test_mice_imputation_single_column()
    print("test_mice_imputation_single_column PASSED")

    print("\nAll tests passed successfully!")
