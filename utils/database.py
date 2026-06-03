import sqlite3

conn = sqlite3.connect(
    "traffic.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS traffic_data(
id INTEGER PRIMARY KEY AUTOINCREMENT,
vehicle_id INTEGER,
timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

def insert_data(vehicle_id):

    cursor.execute(
        """
        INSERT INTO traffic_data(vehicle_id)
        VALUES(?)
        """,
        (vehicle_id,)
    )

    conn.commit()