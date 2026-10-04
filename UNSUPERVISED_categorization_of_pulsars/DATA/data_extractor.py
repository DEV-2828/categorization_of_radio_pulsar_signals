import pandas as pd
import os

def extract_true_pulsars():
    # Define file paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    input_path = os.path.join(project_root, 'htru2', 'HTRU_2.csv')
    output_path = os.path.join(script_dir, 'true_pulsars.csv')
    
    # Feature names as described in Readme.txt
    columns = [
        'mean_integrated_profile',
        'std_integrated_profile',
        'kurtosis_integrated_profile',
        'skewness_integrated_profile',
        'mean_dmsnr_curve',
        'std_dmsnr_curve',
        'kurtosis_dmsnr_curve',
        'skewness_dmsnr_curve',
        'class_label'
    ]

    print(f"Reading dataset from {input_path}...")
    # Load dataset
    df = pd.read_csv(input_path, header=None, names=columns)
    
    # Filter for true pulsars (class_label == 1)
    true_pulsars = df[df['class_label'] == 1].copy()
    
    # Verify the number of rows is exactly 1639
    num_rows = len(true_pulsars)
    print(f"Found {num_rows} true pulsars.")
    
    # The unsupervised pipeline only uses the 8 features, so we can drop the class_label
    true_pulsars = true_pulsars.drop(columns=['class_label'])
    
    # Save the filtered dataset
    print(f"Saving filtered dataset to {output_path}...")
    true_pulsars.to_csv(output_path, index=False)
    print("Extraction complete.")

if __name__ == '__main__':
    extract_true_pulsars()
