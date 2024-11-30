import socket

ESP32_IP = "192.168.155.27"  # Replace with the ESP32's IP address
ESP32_PORT = 80             # Port the ESP32 server is listening on

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    # Connect to the ESP32 server
    client_socket.connect((ESP32_IP, ESP32_PORT))
    print("Connected to ESP32")

    # Send a string
    message = "Hello from Python this is a test please let me know if it works!\n"  # '\n' marks the end of the message
    client_socket.send(message.encode())
    print("Message sent:", message)

    # Optional: Receive acknowledgment
    response = client_socket.recv(1024).decode()
    print("Response from ESP32:", response)

except Exception as e:
    print("Error:", e)

finally:
    client_socket.close()
    print("Connection closed")
