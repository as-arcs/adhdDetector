import os
import pandas as pd
import numpy as np
import warnings  # Added to handle suppression

# Logic for pathing remains the same
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

PHENOTYPIC_FILENAME = 'adhd200_preprocessed_phenotypics.tsv'
TRAIN_CONNECTOMES_DIR = 'ADHD200_CC200_TCs_filtfix'
TEST_CONNECTOMES_DIR = 'ADHD200_CC200_TCs_TestRelease'

class ADHDDataLoader:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.train_connectomes_dir = os.path.join(data_dir, TRAIN_CONNECTOMES_DIR)
        self.test_connectomes_dir = os.path.join(data_dir, TEST_CONNECTOMES_DIR)
        self.num_rois = None 

    def load_phenotypic_data(self):
        """Aggressively hunts for labels and ensures IDs match folder naming (7-digit zfill)."""
        path = os.path.join(self.data_dir, PHENOTYPIC_FILENAME)
        all_labels = []

        if os.path.exists(path):
            try:
                df_master = pd.read_csv(path, sep='\t', dtype=str)
                id_col = next((c for c in df_master.columns if c.lower().replace(" ", "") in ['scandirid', 'id']), None)
                dx_col = next((c for c in df_master.columns if c.lower().strip() == 'dx'), None)
                if id_col and dx_col:
                    all_labels.append(df_master[[id_col, dx_col]].rename(columns={id_col: 'ScanDir ID', dx_col: 'DX'}))
            except: pass

        for root, _, files in os.walk(self.data_dir):
            for f in files:
                if 'phenotypic' in f.lower() and f.endswith(('.csv', '.tsv')):
                    try:
                        temp = pd.read_csv(os.path.join(root, f), sep=None, engine='python', dtype=str)
                        id_c = next((c for c in temp.columns if c.lower().replace(" ", "") in ['scandirid', 'id']), None)
                        dx_c = next((c for c in temp.columns if c.lower().strip() in ['dx', 'diagnosis']), None)
                        if id_c and dx_c:
                            subset = temp[[id_c, dx_c]].copy()
                            subset.columns = ['ScanDir ID', 'DX']
                            all_labels.append(subset)
                    except: continue
        
        if not all_labels:
            raise ValueError("No phenotypic data found.")

        df = pd.concat(all_labels, ignore_index=True)
        df['ScanDir ID'] = df['ScanDir ID'].astype(str).str.strip().str.zfill(7)
        df['DX'] = pd.to_numeric(df['DX'], errors='coerce')
        df = df.dropna(subset=['DX']).drop_duplicates(subset=['ScanDir ID'])
        
        print(f"[i] Phenotypic labels found for {len(df)} unique subjects.")
        return df.astype({'DX': 'int'})

    def _index_connectome_files(self, connectomes_dir):
        file_map = {}
        if not os.path.exists(connectomes_dir):
            return file_map

        site_dirs = [d for d in os.listdir(connectomes_dir) if os.path.isdir(os.path.join(connectomes_dir, d))]

        for site in site_dirs:
            site_path = os.path.join(connectomes_dir, site)
            subject_dirs = [d for d in os.listdir(site_path) if os.path.isdir(os.path.join(site_path, d))]
            for subject_id in subject_dirs:
                subject_path = os.path.join(site_path, subject_id)
                for f in os.listdir(subject_path):
                    if f.endswith('.1D'):
                        full_path = os.path.join(subject_path, f)
                        file_map.setdefault(subject_id, []).append(full_path)
        return file_map

    def flatten_connectome(self, matrix):
        n = matrix.shape[0]
        if matrix.shape != (n, n): return None
        if self.num_rois is None: self.num_rois = n
        upper_tri_indices = np.triu_indices(n, k=1)
        return matrix[upper_tri_indices]

    def _process_subjects(self, df, file_map, binary_classification=False):
        """Match labels to files and extract features, suppressing noisy fMRI warnings."""
        X_list, y_list, valid_ids = [], [], []
        available_folder_ids = set(file_map.keys())
        subjects_to_process = df[df['ScanDir ID'].isin(available_folder_ids)]

        for _, row in subjects_to_process.iterrows():
            subject_id = str(row['ScanDir ID'])
            diagnosis = row['DX']
            matching_paths = file_map.get(subject_id)
            filtered = [p for p in matching_paths if 'sfnwmrda' in os.path.basename(p)]
            file_path = sorted(filtered)[0] if filtered else sorted(matching_paths)[0]

            try:
                time_series_df = pd.read_csv(file_path, sep=r'\s+')
                time_series_df.columns = [c.strip() for c in time_series_df.columns]
                roi_cols = [c for c in time_series_df.columns if c.startswith('Mean_')]
                time_series = time_series_df[roi_cols].values

                if self.num_rois is None: self.num_rois = time_series.shape[1]
                if time_series.shape[1] != self.num_rois: continue

                # suppress RuntimeWarnings for empty/constant ROIs
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    matrix = np.corrcoef(time_series, rowvar=False)
                
                # turn NaNs (from zero-variance regions) into 0.0
                matrix = np.nan_to_num(matrix)
                
                features = self.flatten_connectome(matrix)
                if features is None: continue

                label = diagnosis
                if binary_classification:
                    label = 0 if diagnosis == 0 else 1

                X_list.append(features)
                y_list.append(label)
                valid_ids.append(subject_id)

            except Exception:
                continue

        return np.array(X_list), np.array(y_list), valid_ids

    def load_train_test_data(self, binary_classification=False):
        print(f"Looking for data in: {self.data_dir}")
        try:
            df = self.load_phenotypic_data()
        except Exception as e:
            print(f"[!] Error: {e}")
            return (np.array([]),), (np.array([]),)

        train_map = self._index_connectome_files(self.train_connectomes_dir)
        test_map = self._index_connectome_files(self.test_connectomes_dir)

        print("Processing training subjects...")
        X_train, y_train, train_ids = self._process_subjects(df, train_map, binary_classification)

        print("Processing test subjects...")
        X_test, y_test, test_ids = self._process_subjects(df, test_map, binary_classification)

        return (X_train, y_train, train_ids), (X_test, y_test, test_ids)

if __name__ == "__main__":
    loader = ADHDDataLoader()
    (X_train, y_train, train_ids), (X_test, y_test, test_ids) = loader.load_train_test_data()

    if len(X_train) > 0:
        print(f"\nSuccess!")
        print(f"ROIs detected: {loader.num_rois}")
        print(f"Train: {X_train.shape[0]} subjects")
        print(f"Test:  {X_test.shape[0]} subjects")
        print(f"Labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")