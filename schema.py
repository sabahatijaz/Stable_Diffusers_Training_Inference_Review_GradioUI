import sqlite3

def create_schema(connection):
    cursor = connection.cursor()

    # Create the models table with the Time and approval_status columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            model_id INTEGER PRIMARY KEY,
            model_name TEXT UNIQUE NOT NULL,
            Time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            approval_status TEXT DEFAULT 'Incomplete'
        )
    ''')

    # Create the checkpoints table with a foreign key reference to models
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS checkpoints (
            checkpoint_id INTEGER PRIMARY KEY,
            model_id INTEGER NOT NULL,
            checkpoint_name TEXT NOT NULL,
            FOREIGN KEY (model_id) REFERENCES models(model_id)
        )
    ''')

    connection.commit()
def check_database_schema(database_path, expected_schema):
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Get the current schema from the database
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
        existing_schema = [row[0] for row in cursor.fetchall()]

        # Check if the current schema matches the expected schema
        if set(existing_schema) == set(expected_schema):
            print("Database schema is as expected.")
            return 1
        else:
            print("Database schema does not match the expected schema.")
            return 0

        conn.close()
    except sqlite3.Error as e:
        print("Error:", e)