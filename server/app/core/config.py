from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SERPAI_KEY:str
    LOCATION:str

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()