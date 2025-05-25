import numpy as np
from als import ALS_MatrixCompletion  # Assuming als.py is in the same directory

def test_als_matrix_completion_simple():
    """
    Test ALS_MatrixCompletion with a simple case.
    """
    data = [
        [1, 2, np.nan],
        [np.nan, 5, 6],
        [7, np.nan, 9]
    ]
    
    # Expected shape of the result
    expected_shape = (3, 3)

    # Run ALS matrix completion
    # Using a small number of iterations and a higher tolerance for a quick test
    completed_matrix = ALS_MatrixCompletion(data, rank=2, max_iter=10, tol=0.1)

    # Check if the result has the correct shape
    assert completed_matrix.shape == expected_shape,         f"Expected shape {expected_shape}, but got {completed_matrix.shape}"

    # Check that there are no NaN values in the completed matrix
    assert not np.isnan(completed_matrix).any(),         "Completed matrix should not contain NaN values"

    # Check if the original non-NaN values are approximately preserved
    # This is a basic check; more sophisticated checks might involve comparing
    # the reconstructed values with known values if the underlying model is simple.
    original_data = np.array(data)
    mask = ~np.isnan(original_data)
    
    # We can't expect exact preservation for NaN entries, 
    # but original values should be somewhat close.
    # This check is more of a sanity check for this simple test case.
    # For a real-world scenario, you'd have a more robust way to verify correctness.
    
    # For the purpose of this test, we'll just check if the non-NaN original values 
    # are not drastically changed.
    # A more rigorous test would involve a known dataset and expected output.
    
    # Let's check if the values in the original positions (non-NaN) are still there
    # and if the NaN positions are filled.
    for i in range(original_data.shape[0]):
        for j in range(original_data.shape[1]):
            if not np.isnan(original_data[i, j]):
                # Check if original values are somewhat preserved.
                # Due to the nature of ALS, they might not be identical.
                # This is a loose check.
                assert np.isclose(completed_matrix[i, j], original_data[i, j], atol=1.5),                     f"Original value at ({i},{j}) changed significantly. "                     f"Original: {original_data[i,j]}, Completed: {completed_matrix[i,j]}"
            else:
                # Check that NaN values have been filled
                assert not np.isnan(completed_matrix[i,j]),                     f"NaN value at ({i},{j}) was not filled."

    print("ALS Matrix Completion test passed (simple case).")
    print("Original Matrix:")
    print(original_data)
    print("Completed Matrix:")
    print(completed_matrix)

if __name__ == "__main__":
    test_als_matrix_completion_simple()
    # Example of how to run with pytest:
    # Create a virtual environment: python -m venv .venv
    # Activate it: source .venv/bin/activate (Linux/macOS) or .venv\Scripts\activate (Windows)
    # Install pytest and numpy: pip install pytest numpy
    # Run tests: pytest
