from supabase import create_client, Client
import os

def get_supabase_client() -> Client:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "https://vyndvwdwqyrnzxjdcakf.supabase.co")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZ5bmR2d2R3cXlybnp4amRjYWtmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDY2NTc0NTksImV4cCI6MjA2MjIzMzQ1OX0._HhABMUYPP_86KJxq5Tnj6-KxU1XK06nUpkV10avMyg")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_to_supabase(client: Client, bucket_name: str, filename: str, file_data: bytes):
    return client.storage.from_(bucket_name).upload(
        path=filename,
        file=file_data,
        file_options={"content-type": "audio/wav"}
    )
def insert_annotation(client: Client, annotation: dict):
    response = client.table("annotations").insert(annotation).execute()
    return response

import tempfile

def list_wav_files(client: Client, bucket_name: str = "nfc-uploads"):
    response = client.storage.from_(bucket_name).list()
    if isinstance(response, list):
        return [f["name"] for f in response if f["name"].endswith(".wav")]
    return []

def download_wav_file(client: Client, filename: str, bucket_name: str = "nfc-uploads"):
    data = client.storage.from_(bucket_name).download(filename)
    if isinstance(data, bytes):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(data)
        temp_file.close()
        return temp_file.name
    return None

def get_annotated_filenames(client: Client):
    response = client.table("annotations").select("filename").execute()
    if response.data:
        return list(set([row["filename"] for row in response.data]))
    return []

