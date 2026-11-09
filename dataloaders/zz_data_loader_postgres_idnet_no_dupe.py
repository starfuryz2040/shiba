import urllib

import pandas as pd
from sqlalchemy import create_engine


password = urllib.parse.quote_plus("postgres")
db_connection_str = f'postgresql+psycopg2://postgres:{password}@localhost:5434/postgres'
# 1. Initialize the PostgreSQL connection engine
# Format: postgresql://username:password@host:port/database_name
engine = create_engine(db_connection_str)

# 2. Read the Parquet file into memory
df = pd.read_parquet("../silver/idnet_no_dupe.parquet", engine="pyarrow")

# 3. Append or create the table inside PostgreSQL
df.to_sql(
    name="idnet_no_dupe",
    con=engine,
    if_exists="replace",  # Options: 'fail', 'replace', 'append'
    index=False          # Discard the DataFrame's numeric index row
)
