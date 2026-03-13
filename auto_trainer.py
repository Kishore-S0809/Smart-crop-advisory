import io
import os
import json
import sys
import traceback
import zipfile
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
REPORTS_DIR = os.path.join(REPO_ROOT, 'reports')
REPORT_JSON = os.path.join(REPORTS_DIR, 'report.json')
TRAIN_LOG = os.path.join(REPORTS_DIR, 'training_logs.txt')

# Defer heavy imports inside functions so the script can at least
# generate reports/logs if some optional deps are missing.

def log(msg: str):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    with open(TRAIN_LOG, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(msg)


def safe_imports():
    modules = {}
    try:
        import pandas as pd
        modules['pd'] = pd
    except Exception as e:
        log(f"Missing pandas: {e}")
    try:
        import numpy as np
        modules['np'] = np
    except Exception as e:
        log(f"Missing numpy: {e}")
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        modules.update({
            'SimpleImputer': SimpleImputer,
            'OneHotEncoder': OneHotEncoder,
            'StandardScaler': StandardScaler,
            'ColumnTransformer': ColumnTransformer,
            'Pipeline': Pipeline,
            'train_test_split': train_test_split,
            'accuracy_score': accuracy_score,
            'f1_score': f1_score,
            'r2_score': r2_score,
            'mean_squared_error': mean_squared_error,
            'RandomForestClassifier': RandomForestClassifier,
            'RandomForestRegressor': RandomForestRegressor,
        })
    except Exception as e:
        log(f"Missing scikit-learn: {e}")
    try:
        import joblib
        modules['joblib'] = joblib
    except Exception as e:
        log(f"Missing joblib: {e}")
    try:
        # Optional time-series libs
        import statsmodels.api as sm
        modules['sm'] = sm
    except Exception as e:
        log(f"statsmodels not available: {e}")
    try:
        # Prophet is optional; if present we may prefer it
        from prophet import Prophet  # type: ignore
        modules['Prophet'] = Prophet
    except Exception as e:
        log(f"Prophet not available: {e}")
    try:
        import tensorflow as tf
        from keras import layers, models, applications
        modules.update({'tf': tf, 'layers': layers, 'models': models, 'applications': applications})
    except Exception as e:
        log(f"TensorFlow not available: {e}")
    return modules


def _scan_root_for_datasets(root_path):
    files = []
    image_roots = []
    for root, dirs, filenames in os.walk(root_path):
        # Detect if current root looks like an image dataset root
        try:
            subdirs = [os.path.join(root, d) for d in dirs]
            has_class_subdirs = False
            for sd in subdirs:
                try:
                    entries = os.listdir(sd)
                except Exception:
                    entries = []
                if any(name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')) for name in entries):
                    has_class_subdirs = True
                    break
            if has_class_subdirs and len(subdirs) >= 2:
                image_roots.append(root)
        except Exception:
            pass
        for d in list(dirs):
            # Heuristic: image dataset root has subfolders as class names containing images
            class_dir = os.path.join(root, d)
            try:
                entries = os.listdir(class_dir)
            except Exception:
                entries = []
            if any(name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')) for name in entries):
                # This is a flat image folder, not multi-class root.
                continue
            if any(os.path.isdir(os.path.join(class_dir, e)) for e in entries):
                # contains subdirs -> likely class folders under image dataset root
                image_roots.append(class_dir)
        for fn in filenames:
            if fn.lower().endswith(('.csv', '.xlsx', '.xls')):
                files.append(os.path.join(root, fn))
            elif fn.lower().endswith('.zip'):
                zip_path = os.path.join(root, fn)
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        for member in zf.namelist():
                            if member.lower().endswith(('.csv', '.xlsx', '.xls')):
                                # Encode as "zip:<zip_path>!<member>" so the caller can read it
                                files.append(f"zip:{zip_path}!{member}")
                except zipfile.BadZipFile:
                    pass
    return files, image_roots


def discover_datasets():
    # Scan both data/ and src/ as requested
    roots = [DATA_DIR, os.path.join(REPO_ROOT, 'src')]
    all_files = []
    all_image_roots = []
    for rp in roots:
        if os.path.isdir(rp):
            files, image_roots = _scan_root_for_datasets(rp)
            all_files.extend(files)
            all_image_roots.extend(image_roots)
    # Deduplicate
    all_files = list(dict.fromkeys(all_files))
    all_image_roots = list(dict.fromkeys(all_image_roots))
    return all_files, all_image_roots


def classify_csv_purpose(pd, df, path):
    name = os.path.basename(path).lower()
    # Basic keyword heuristics
    if any(k in name for k in ['weather', 'rain', 'temperature', 'climate']):
        return 'weather'
    if any(k in name for k in ['fertilizer', 'fertiliser', 'soil', 'nutrient', 'npk']):
        return 'fertilizer_advisory'
    if any(k in name for k in ['recommend', 'crop_recommendation']):
        return 'crop_recommendation'
    # Look into columns
    cols = [c.lower() for c in df.columns]
    if any(c in cols for c in ['rainfall', 'precipitation', 'temperature', 'humidity', 'date', 'time']):
        return 'weather'
    if any(c in cols for c in ['nitrogen', 'phosphorus', 'potassium', 'n', 'p', 'k', 'ph', 'soil_type']):
        return 'fertilizer_advisory'
    return 'crop_recommendation'


def detect_target_column(pd, df):
    # Prefer common label names
    label_candidates = ['label', 'target', 'class', 'crop', 'crop_name', 'yield', 'price']
    for c in df.columns:
        if c.strip().lower() in label_candidates:
            return c
    # Fallback: last column as target
    return df.columns[-1]


def is_classification_target(pd, series):
    unique_cnt = series.nunique(dropna=True)
    # Heuristic: small number of unique values => classification
    if unique_cnt <= 20 and series.dtype.name not in ['float64', 'float32']:
        return True
    return False


def preprocess_csv_and_split(mods, df, target_col):
    pd = mods['pd']
    SimpleImputer = mods['SimpleImputer']
    OneHotEncoder = mods['OneHotEncoder']
    StandardScaler = mods['StandardScaler']
    ColumnTransformer = mods['ColumnTransformer']
    Pipeline = mods['Pipeline']
    train_test_split = mods['train_test_split']

    y = df[target_col]
    X = df.drop(columns=[target_col])

    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return preprocessor, X_train, X_test, y_train, y_test


def train_csv_model(mods, df, purpose, save_basename):
    pd = mods.get('pd')
    if pd is None:
        log('pandas not available, skipping CSV training')
        return None

    # Downsample very large datasets for speed
    try:
        if len(df) > 50000:
            df = df.sample(n=50000, random_state=42)
    except Exception:
        pass

    target_col = detect_target_column(pd, df)
    clf_task = is_classification_target(pd, df[target_col])

    preprocessor, X_train, X_test, y_train, y_test = preprocess_csv_and_split(mods, df, target_col)

    Pipeline = mods['Pipeline']
    RandomForestClassifier = mods['RandomForestClassifier']
    RandomForestRegressor = mods['RandomForestRegressor']
    accuracy_score = mods['accuracy_score']
    f1_score = mods['f1_score']
    r2_score = mods['r2_score']
    mean_squared_error = mods['mean_squared_error']

    if clf_task:
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        pipe = Pipeline(steps=[('preprocess', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
        }
        ext = 'joblib'
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        pipe = Pipeline(steps=[('preprocess', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        metrics = {
            'r2': float(r2_score(y_test, y_pred)),
            'rmse': float(rmse),
        }
        ext = 'joblib'

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{save_basename}.{ext}")
    mods['joblib'].dump(pipe, model_path)
    return {'task': 'classification' if clf_task else 'regression', 'metrics': metrics, 'model_path': model_path}


def _make_datetime_index(pd, df):
    cols_lower = {c.lower(): c for c in df.columns}
    # Direct datetime column
    for key in ['datetime', 'timestamp', 'date', 'ds']:
        if key in cols_lower:
            col = cols_lower[key]
            return pd.to_datetime(df[col], errors='coerce')
    # Year / Month / Day
    year_col = cols_lower.get('year')
    month_col = cols_lower.get('month')
    day_col = cols_lower.get('day')
    if year_col is not None:
        y = df[year_col].astype(int)
        m = df[month_col].astype(int) if month_col is not None else 1
        d = df[day_col].astype(int) if day_col is not None else 1
        return pd.to_datetime({'year': y, 'month': m, 'day': d}, errors='coerce')
    return None


def train_time_series_model(mods, df, save_basename):
    pd = mods.get('pd')
    if pd is None:
        log('pandas not available, skipping time-series training')
        return None

    # Downsample long series for speed
    if len(df) > 100000:
        df = df.tail(100000)

    # Choose target: prefer typical weather variables
    preferred = ['rainfall', 'precipitation', 'temperature', 'humidity']
    target_col = None
    for c in df.columns:
        if c.lower() in preferred:
            target_col = c
            break
    if target_col is None:
        # last numeric column
        num_cols = [c for c in df.columns if df[c].dtype != 'object']
        target_col = num_cols[-1] if num_cols else df.columns[-1]

    ds = _make_datetime_index(pd, df)
    if ds is None:
        # Fallback: try parsing first column as date
        ds = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    ts = pd.DataFrame({'ds': ds, 'y': pd.to_numeric(df[target_col], errors='coerce')}).dropna()
    ts = ts.sort_values('ds')
    if len(ts) < 20:
        log('Time series too short, skipping')
        return None

    # Train/test split by time (last 20% test)
    split_idx = int(len(ts) * 0.8)
    train, test = ts.iloc[:split_idx], ts.iloc[split_idx:]

    result = None
    model_info = {}

    # Prefer Prophet if available
    Prophet = mods.get('Prophet')
    if Prophet is not None:
        try:
            m = Prophet()
            m.fit(train.rename(columns={'ds': 'ds', 'y': 'y'}))
            forecast = m.predict(test[['ds']])
            y_pred = forecast['yhat'].values
            y_true = test['y'].values
            import numpy as np
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            model_info = {'model': 'prophet', 'mae': mae, 'rmse': rmse}
            # Save
            model_path = os.path.join(MODELS_DIR, f"{save_basename}_prophet.joblib")
            mods['joblib'].dump(m, model_path)
            result = {'task': 'time_series_forecast', 'metrics': {'mae': mae, 'rmse': rmse}, 'model_path': model_path}
            return result
        except Exception:
            log('Prophet failed, falling back to ARIMA')

    # Fallback to ARIMA(1,1,1)
    sm = mods.get('sm')
    if sm is None:
        log('statsmodels not available, skipping time-series training')
        return None
    try:
        model = sm.tsa.statespace.SARIMAX(train['y'], order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False)
        y_pred = fitted.forecast(steps=len(test))
        import numpy as np
        y_true = test['y'].values
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        model_path = os.path.join(MODELS_DIR, f"{save_basename}_arima.pkl")
        mods['joblib'].dump(fitted, model_path)
        return {'task': 'time_series_forecast', 'metrics': {'mae': mae, 'rmse': rmse}, 'model_path': model_path}
    except Exception:
        log('ARIMA training failed')
        return None


def prepare_image_data(mods, image_root, img_size=(224, 224), batch_size=32):
    tf = mods.get('tf')
    if tf is None:
        log('TensorFlow not available, skipping image training')
        return None
    # First split: train vs (val+test)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_root,
        validation_split=0.3,
        subset='training',
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )
    valtest_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_root,
        validation_split=0.3,
        subset='validation',
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )
    # Split valtest into val and test
    card = tf.data.experimental.cardinality(valtest_ds)
    val_size = card // 2
    val_ds = valtest_ds.take(val_size)
    test_ds = valtest_ds.skip(val_size)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    class_names = train_ds.class_names
    return train_ds, val_ds, test_ds, class_names


def build_transfer_model(mods, num_classes, img_size=(224, 224)):
    tf = mods['tf']
    applications = mods['applications']
    layers = mods['layers']
    base = applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(img_size[0], img_size[1], 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
    x = applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_image_model(mods, image_root, save_basename):
    tf = mods.get('tf')
    if tf is None:
        log('TensorFlow not available, skipping image training')
        return None
    prepared = prepare_image_data(mods, image_root)
    if prepared is None:
        return None
    train_ds, val_ds, test_ds, class_names = prepared
    model = build_transfer_model(mods, num_classes=len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)
    val_acc = history.history.get('val_accuracy', [None])[-1]
    # Evaluate on test
    test_metrics = model.evaluate(test_ds, verbose=0)
    test_acc = float(test_metrics[1]) if len(test_metrics) > 1 else None
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{save_basename}.h5")
    model.save(model_path)
    return {'task': 'image_classification', 'metrics': {'val_accuracy': float(val_acc) if val_acc is not None else None, 'test_accuracy': test_acc}, 'model_path': model_path, 'classes': class_names}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    mods = safe_imports()
    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'files_scanned': [],
        'datasets': [],
    }

    try:
        csv_files, image_roots = discover_datasets()
        report['files_scanned'] = csv_files + image_roots
        log(f"Discovered {len(csv_files)} tabular files and {len(image_roots)} image roots")

        # Process CSV files
        pd = mods.get('pd')
        for csv_path in csv_files:
            try:
                if csv_path.startswith('zip:'):
                    # Format: zip:<zip_file_path>!<member>
                    rest = csv_path[len('zip:'):]
                    if '!' not in rest:
                        log(f"Skipping malformed zip path (expected 'zip:<file>!<member>'): {csv_path}")
                        continue
                    zip_file_path, member = rest.split('!', 1)
                    with zipfile.ZipFile(zip_file_path, 'r') as zf:
                        with zf.open(member) as fh:
                            data = io.BytesIO(fh.read())
                    file_ext = os.path.splitext(member)[1].lower()
                    if file_ext == '.csv':
                        df = pd.read_csv(data) if pd is not None else None
                    else:
                        df = pd.read_excel(data) if pd is not None else None
                    display_path = f"{os.path.basename(zip_file_path)}!{member}"
                    member_name = os.path.splitext(os.path.basename(member))[0]
                elif csv_path.lower().endswith('.csv'):
                    df = pd.read_csv(csv_path) if pd is not None else None
                    display_path = csv_path
                    member_name = None
                    file_ext = '.csv'
                else:
                    df = pd.read_excel(csv_path) if pd is not None else None
                    display_path = csv_path
                    member_name = None
                    file_ext = os.path.splitext(csv_path)[1].lower()
                if df is None:
                    log(f"Skipping {display_path}: pandas unavailable")
                    continue
                purpose = classify_csv_purpose(pd, df, csv_path)
                base = member_name if member_name else os.path.splitext(os.path.basename(csv_path))[0]
                save_basename = f"{base}_{purpose}"
                # Route weather-like datasets to time-series model if possible
                if purpose == 'weather':
                    res = train_time_series_model(mods, df, save_basename)
                    preprocessing_desc = 'time-indexing, ARIMA/Prophet forecasting'
                    model_name = 'time_series_forecast' if res else None
                else:
                    res = train_csv_model(mods, df, purpose, save_basename)
                    preprocessing_desc = 'impute + scale numeric, impute + onehot categorical'
                    model_name = res['task'] if res else None
                entry = {
                    'path': display_path,
                    'type': 'csv',
                    'purpose': purpose,
                    'preprocessing': preprocessing_desc,
                    'model': model_name,
                    'metrics': res['metrics'] if res else None,
                    'saved_model': res['model_path'] if res else None,
                    'format': file_ext.lstrip('.'),
                }
                report['datasets'].append(entry)
                log(f"Finished CSV: {display_path}")
            except Exception:
                log(f"Error processing {csv_path}:\n{traceback.format_exc()}")

        # Process image datasets
        for image_root in image_roots:
            try:
                base = os.path.basename(image_root.rstrip(os.sep)).replace(' ', '_').lower()
                save_basename = f"{base}_images"
                res = train_image_model(mods, image_root, save_basename)
                entry = {
                    'path': image_root,
                    'type': 'images',
                    'purpose': 'image_classification',
                    'preprocessing': 'resize 224x224, normalize, train/val/test split',
                    'model': 'transfer_learning_efficientnet_b0' if res else None,
                    'metrics': res['metrics'] if res else None,
                    'saved_model': res['model_path'] if res else None,
                    'classes': res['classes'] if res and 'classes' in res else None,
                }
                report['datasets'].append(entry)
                log(f"Finished images: {image_root}")
            except Exception:
                log(f"Error processing images {image_root}:\n{traceback.format_exc()}")

    finally:
        with open(REPORT_JSON, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        log(f"Wrote report to {REPORT_JSON}")


if __name__ == '__main__':
    main()
