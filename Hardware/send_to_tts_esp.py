import socket

import subprocess
import time
import os

output_folder = './responds_vm'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def transfer_file_when_ready(server, user, remote_file_path, local_file_path, check_interval=0.5):
    while True:
        try:
            # Check if the file exists on the server
            result = subprocess.run(
                ["ssh", f"{user}@{server}", f"test -f {remote_file_path}"],
                capture_output=True
            )
            if result.returncode == 0:  # File exists
                # Use scp to transfer the file
                # os.system('rm -r sharpest_frame.txt')
                subprocess.run(
                    ["scp", f"{user}@{server}:{remote_file_path}", local_file_path],
                    check=True
                )
                print(f"File transferred to {local_file_path}")
                break
            else:
                print("File not found. Retrying...")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
        time.sleep(check_interval)

# Example Usage
server = "195.242.13.247"
user = "ubuntu"
remote_file_path = "/home/ubuntu/meta_llama_hacks/responds/sharpest_frame.txt"
local_file_path = "./responds_vm/sharpest_frame.txt"

transfer_file_when_ready(server, user, remote_file_path, local_file_path)

with open('./responds_vm/sharpest_frame.txt', 'r') as f:
    message = f.read()

print(message)



ESP32_IP = "192.168.155.27"  # Replace with the ESP32's IP address
ESP32_PORT = 80             # Port the ESP32 server is listening on

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    # Connect to the ESP32 server
    client_socket.connect((ESP32_IP, ESP32_PORT))
    print("Connected to ESP32")

    # Send a string
    # message = "Hello from Python!\n"  # '\n' marks the end of the message
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
