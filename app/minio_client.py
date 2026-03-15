import io

from minio import Minio

from config import (
    MINIO_HOST,
    MINIO_PORT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_BUCKET,
)


class MinIOClient:
    def __init__(self):
        self.client = Minio(
            f"{MINIO_HOST}:{MINIO_PORT}",
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False,
        )
        if not self.client.bucket_exists(MINIO_BUCKET):
            self.client.make_bucket(MINIO_BUCKET)

    def store_document(
        self,
        doc_id: str,
        content: bytes,
        filename: str,
        content_type: str = "application/json",
    ) -> str:
        object_name = f"{doc_id}/{filename}"
        self.client.put_object(
            MINIO_BUCKET,
            object_name,
            io.BytesIO(content),
            len(content),
            content_type=content_type,
        )
        return object_name

    def get_document(self, doc_id: str, filename: str) -> bytes:
        object_name = f"{doc_id}/{filename}"
        response = self.client.get_object(MINIO_BUCKET, object_name)
        return response.read()

    def delete_document(self, doc_id: str):
        objects = self.client.list_objects(MINIO_BUCKET, prefix=f"{doc_id}/")
        for obj in objects:
            self.client.remove_object(MINIO_BUCKET, obj.object_name)

    def list_documents(self) -> list:
        objects = self.client.list_objects(MINIO_BUCKET, recursive=False)
        return [obj.object_name.rstrip("/") for obj in objects if obj.is_dir]

    def health_check(self) -> bool:
        try:
            self.client.bucket_exists(MINIO_BUCKET)
            return True
        except Exception:
            return False
