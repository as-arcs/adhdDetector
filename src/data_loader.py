import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')

# The ADHD-200 phenotypic file uses these names for diagnosis codes:
# 0 = Control, 1 = ADHD-Combined, 2 = ADHD-Hyperactive, 3 = ADHD-Inattentive
PHENOTYPIC_FILENAMES = ['phenotypics.tsv', 'phenotypic.tsv', 'phenotypic.csv']


class ADHDDataLoader:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.connectomes_dir = os.path.join(data_dir, 'connectomes')
        self.num_rois = None  # determined dynamically from the first file

    def _find_phenotypic_file(self):
        for name in PHENOTYPIC_FILENAMES:
            path = os.path.join(self.data_dir, name)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"No phenotypic file found in {self.data_dir}. "
            f"Expected one of: {PHENOTYPIC_FILENAMES}"
        )

    def load_phenotypic_data(self):
        """Loads patient labels and IDs from a TSV or CSV phenotypic file."""
        path = self._find_phenotypic_file()

        # Try tab-separated first (ADHD-200 standard), then comma-separated
        for sep in ['\t', ',']:
            try:
                df = pd.read_csv(path, sep=sep)
                if 'DX' in df.columns and 'ScanDir ID' in df.columns:
                    break
            except Exception:
                continue
        else:
            raise ValueError(
                f"Could not parse {path} as a valid phenotypic file. "
                "Ensure it is a TSV/CSV with 'ScanDir ID' and 'DX' columns. "
                "Re-download from the ADHD-200 Preprocessed Connectomes Project if needed."
            )

        df['DX'] = pd.to_numeric(df['DX'], errors='coerce')
        df = df[df['DX'].notna()]
        df = df.astype({'DX': 'int'})
        return df

    def _load_time_series(self, file_path):
        """Load a .1D time-series file (tab-separated, with header and metadata columns)."""
        df = pd.read_csv(file_path, sep='\t')
        roi_cols = [c for c in df.columns if c.startswith('Mean_')]
        return df[roi_cols].values

    def flatten_connectome(self, matrix):
        """Flatten NxN correlation matrix to N*(N-1)/2 unique features (upper triangle)."""
        n = matrix.shape[0]
        if matrix.shape != (n, n):
            return None
        if self.num_rois is None:
            self.num_rois = n
        upper_tri_indices = np.triu_indices(n, k=1)
        return matrix[upper_tri_indices]

    def load_data(self, binary_classification=False):
        print(f"Looking for data in: {self.data_dir}")

        try:
            df = self.load_phenotypic_data()
        except (FileNotFoundError, ValueError) as e:
            print(f"[!] Error: {e}")
            return np.array([]), np.array([]), []

        if not os.path.exists(self.connectomes_dir):
            print(f"[!] Error: Connectomes folder not found at {self.connectomes_dir}")
            return np.array([]), np.array([]), []

        print("Indexing files in connectomes folder...")
        files_in_dir = [f for f in os.listdir(self.connectomes_dir) if f.endswith('.1D')]

        X_list = []
        y_list = []
        valid_ids = []
        skipped = 0

        print(f"Processing {len(df)} subjects...")

        for _, row in df.iterrows():
            subject_id = str(row['ScanDir ID'])
            diagnosis = row['DX']

            matching_files = [f for f in files_in_dir if subject_id in f]

            if not matching_files:
                skipped += 1
                continue

            matching_files.sort()
            file_name = matching_files[0]
            file_path = os.path.join(self.connectomes_dir, file_name)

            try:
                time_series = self._load_time_series(file_path)

                # Pearson correlation → connectivity matrix (ROIs x ROIs)
                matrix = np.corrcoef(time_series, rowvar=False)

                # Check for NaNs (caused by 0 variance regions) and replace with 0
                if np.isnan(matrix).any():
                    matrix = np.nan_to_num(matrix) 

                features = self.flatten_connectome(matrix)
                if features is None:
                    continue

                if binary_classification:
                    label = 0 if diagnosis == 0 else 1
                else:
                    label = diagnosis

                X_list.append(features)
                y_list.append(label)
                valid_ids.append(subject_id)

            except Exception as e:
                print(f"Error processing {subject_id}: {e}")

        if skipped:
            print(f"[i] {skipped} subjects had no matching connectome file.")

        X = np.array(X_list)
        y = np.array(y_list)

        return X, y, valid_ids

    def get_train_test_split(self, X, y, test_size=0.2, random_state=42):
        """
        Standardized split for the whole team.
        Ensures everyone uses the exact same Train/Test sets.
        """
        if len(X) == 0:
            return [], [], [], []
            
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_idx, test_idx in splitter.split(X, y):
            return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        return [], [], [], []

if __name__ == "__main__":
    loader = ADHDDataLoader()
    X, y, ids = loader.load_data()

    if len(X) > 0:
        n_rois = loader.num_rois
        expected_features = n_rois * (n_rois - 1) // 2
        print(f"\nSuccess! Loaded {X.shape[0]} subjects.")
        print(f"ROIs detected: {n_rois}")
        print(f"Feature vector length: {X.shape[1]} (expected {expected_features})")
        print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        X_train, X_test, y_train, y_test = loader.get_train_test_split(X, y)
        print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    else:
        print("No data loaded. Check that phenotypic.tsv exists and is valid.")