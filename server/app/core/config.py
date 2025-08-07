from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SERPAI_KEY:str
    LOCATION:str
    DATABASE_URL: str
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    IMAGE_EMBEDDING_MODEL: str = "clip-ViT-B-32"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()