import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('models.db')
cursor = conn.cursor()

try:
    # Delete all entries in the 'checkpoints' table
    cursor.execute("DELETE FROM checkpoints")

    # Delete all entries in the 'models' table
    cursor.execute("DELETE FROM models")

    # Commit the changes to the database
    conn.commit()

    print("All entries deleted successfully.")

except sqlite3.Error as e:
    print("Error:", e)

finally:
    # Close the database connection
    conn.close()
