import os
import json
import boto3
import re
from typing import Dict, Any
from botocore.exceptions import NoCredentialsError, ClientError

class S3ModelUploader:
    def __init__(self, bucket_identifier: str):
        """
        Initialize with either bucket name or bucket ARN.
        
        Args:
            bucket_identifier: Either bucket name (e.g., 'my-bucket') 
                              or bucket ARN (e.g., 'arn:aws:s3:::my-bucket')
        """
        self.bucket_arn = bucket_identifier
        # Extract bucket name from ARN if ARN is provided
        if bucket_identifier.startswith('arn:aws:s3:::'):
            self.bucket_name = bucket_identifier.split(':::')[-1]
        else:
            self.bucket_name = bucket_identifier
        
        self.s3_client = boto3.client('s3')
        
    def upload_file(self, file_path: str, s3_key: str) -> bool:
        """Upload a file to S3."""
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            print(f"[ml_pipeline.s3_utils] Uploaded {file_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except (NoCredentialsError, ClientError) as e:
            print(f"[ml_pipeline.s3_utils] Error uploading {file_path}: {e}")
            return False
    
    def upload_model_artifacts(self, model_version: str, artifacts_dir: str = "models") -> bool:
        """Upload all model artifacts for a given version to S3."""
        artifacts = ["model.pkl", "metrics.json", "metadata.json"]
        s3_prefix = f"models/{model_version}"
        
        success = True
        for artifact in artifacts:
            local_path = os.path.join(artifacts_dir, artifact)
            if os.path.exists(local_path):
                s3_key = f"{s3_prefix}/{artifact}"
                if not self.upload_file(local_path, s3_key):
                    success = False
            else:
                print(f"[ml_pipeline.s3_utils] Warning: {local_path} not found")
        
        return success
    
    def check_credentials(self) -> bool:
        """Check if AWS credentials are available."""
        try:
            self.s3_client.list_buckets()
            return True
        except NoCredentialsError:
            print("[ml_pipeline.s3_utils] AWS credentials not found")
            return False
        except Exception as e:
            print(f"[ml_pipeline.s3_utils] Error checking AWS credentials: {e}")
            return False
