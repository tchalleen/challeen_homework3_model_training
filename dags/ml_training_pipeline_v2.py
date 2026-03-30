from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import sys, os
import json

# Ensure src/ is on path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.model import train_model
from ml_pipeline.evaluation import evaluate_model
from ml_pipeline.versioning import generate_model_version, save_metadata
from ml_pipeline.s3_utils import S3ModelUploader

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="ml_training_pipeline_v2",
    default_args=default_args,
    description="Train, evaluate, and promote ML model with versioning and S3 storage",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    def train_model_wrapper(**context):
        """Train model and return model object and test data."""
        # Load breast cancer dataset
        data = load_breast_cancer()
        X = data.data
        y = data.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create DataFrame for train_model function
        import pandas as pd
        train_df = pd.DataFrame(X_train, columns=data.feature_names)
        train_df['target'] = y_train
        
        # Generate model version
        model_version = generate_model_version(context.get('execution_date'))
        model_path = f"models/model_{model_version}.pkl"
        
        # Train model
        accuracy = train_model(train_df, model_path)
        
        # Store test data and model for evaluation
        ti = context['task_instance']
        ti.xcom_push(key='X_test', value=X_test.tolist())
        ti.xcom_push(key='y_test', value=y_test.tolist())
        ti.xcom_push(key='model_path', value=model_path)
        ti.xcom_push(key='model_version', value=model_version)
        ti.xcom_push(key='feature_names', value=data.feature_names.tolist())
        
        return {"model_path": model_path, "accuracy": accuracy, "model_version": model_version}

    def evaluate_model_wrapper(**context):
        """Evaluate model and save metrics."""
        ti = context['task_instance']
        
        # Get data from previous task
        model_path = ti.xcom_pull(task_ids='train_model', key='model_path')
        X_test_list = ti.xcom_pull(task_ids='train_model', key='X_test')
        y_test_list = ti.xcom_pull(task_ids='train_model', key='y_test')
        feature_names = ti.xcom_pull(task_ids='train_model', key='feature_names')
        model_version = ti.xcom_pull(task_ids='train_model', key='model_version')
        
        # Convert back to arrays
        import pandas as pd
        import numpy as np
        X_test = np.array(X_test_list)
        y_test = np.array(y_test_list)
        
        # Load model
        import joblib
        model = joblib.load(model_path)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, "models/metrics.json")
        
        # Save metadata
        metadata = save_metadata(
            model_version, 
            "breast_cancer", 
            "logistic_regression", 
            metrics["accuracy"],
            "models/metadata.json"
        )
        
        return {"accuracy": metrics["accuracy"], "model_version": model_version}

    def promote_model_wrapper(**context):
        """Promote model to S3 if it meets quality threshold."""
        ti = context['task_instance']
        
        # Get evaluation results
        results = ti.xcom_pull(task_ids='evaluate_model')
        accuracy = results["accuracy"]
        model_version = results["model_version"]
        
        # Quality threshold
        THRESHOLD = 0.94
        
        if accuracy >= THRESHOLD:
            print(f"[promote_model] Model accuracy {accuracy:.4f} meets threshold {THRESHOLD}")
            
            # Rename model file to standard name for S3 upload
            import shutil
            model_path = f"models/model_{model_version}.pkl"
            standard_model_path = "models/model.pkl"
            shutil.copy2(model_path, standard_model_path)
            
            # Upload to S3 (if credentials available)
            bucket_identifier = os.environ.get('S3_BUCKET_ARN', os.environ.get('S3_BUCKET_NAME', 'your-bucket-name'))
            uploader = S3ModelUploader(bucket_identifier)
            
            print(f"[promote_model] Checking AWS credentials for bucket: {bucket_identifier}")
            if uploader.check_credentials():
                print(f"[promote_model] AWS credentials found, attempting upload...")
                success = uploader.upload_model_artifacts(model_version)
                if success:
                    print(f"[promote_model] Successfully uploaded model {model_version} to S3")
                else:
                    print(f"[promote_model] Failed to upload model {model_version} to S3")
                    raise Exception("S3 upload failed")
            else:
                print("[promote_model] No AWS credentials found - skipping S3 upload")
                print("[promote_model] Model artifacts are available locally in models/ directory")
            
            return {"promoted": True, "model_version": model_version, "accuracy": accuracy}
        else:
            print(f"[promote_model] Model accuracy {accuracy:.4f} below threshold {THRESHOLD}")
            raise Exception(f"Model accuracy {accuracy:.4f} below threshold {THRESHOLD}")

    # Define tasks
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model_wrapper,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model_wrapper,
    )

    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model_wrapper,
    )

    # Define task dependencies
    train_task >> evaluate_task >> promote_task