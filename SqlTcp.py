import socket
import mysql.connector
from mysql.connector import Error

# Database connection setup
def connect_to_database():
    try:
        db = mysql.connector.connect(
            host="database-ranking.cwtvfmjokt9c.us-west-1.rds.amazonaws.com",
            user="admin",
            password="mobimobi",
            database="sushitech_schema"
        )
        return db
    except Error as e:
        print("Error while connecting to MySQL", e)
        return None

# Function to insert data into the database
def insert_into_database(db, data):
    try:
        mycursor = db.cursor()
        sql = "INSERT INTO escape_game_test (date, time, nick_name, gender_code, age_range_code, score) VALUES (%s, %s, %s, %s, %s, %s)"
        mycursor.execute(sql, data)
        db.commit()
        print("Data inserted successfully")
    except Error as e:
        print("Error:", e)
        db.rollback()
    finally:
        if db.is_connected():
            mycursor.close()

# Function to start a client that connects to a server
def start_client(server_ip, server_port):
    # Create a TCP/IP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the server's port
    server_address = (server_ip, server_port)
    print(f"Connecting to {server_ip} port {server_port}")
    client_socket.connect(server_address)

    try:
        # Receive the data sent by the server
        while True:
            data = client_socket.recv(1024)
            if data:
                message = data.decode('utf-8')
                print(f"Received data: {message}")

                # Parse the data
                data_parts = message.split(',')
                if len(data_parts) == 6:
                    # Insert data into the database
                    db = connect_to_database()
                    if db is not None:
                        insert_into_database(db, tuple(data_parts))
                        db.close()
                else:
                    print("Received data is in an incorrect format.")
            else:
                break

    except Error as e:
        print(f"An error occurred: {e}")
    finally:
        client_socket.close()
        print("Connection closed.")

# Replace 'server_ip' and 'server_port' with the actual server IP and port
server_ip = '192.168.0.189'  # The server's IP address
server_port = 8199  # The server's port

start_client(server_ip, server_port)