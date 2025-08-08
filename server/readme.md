```
pip install git+https://github.com/openai/CLIP.git
```

`.env`
```
DATABASE_URL=postgresql+asyncpg://username:pass@localhost:5432/dbname
```

### Start Postgres and log in-

```
sudo service postgresql start
psql -U postgres -d postgres -h localhost
```

### Table schema
```
CREATE TABLE product_results (
    id SERIAL PRIMARY KEY,
    query VARCHAR,
    description VARCHAR,
    provider VARCHAR,
    title VARCHAR,
    price VARCHAR,
    url VARCHAR,
    image_url VARCHAR,
    text_embedding vector(384),
    image_embedding vector(512),
    product_metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```