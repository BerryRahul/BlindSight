from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os
import ollama
from PIL import Image

def ensure_image_compatibility(image_path):
    try:
        with Image.open(image_path) as img:
            compatible_path = image_path
            img.save(compatible_path, format='PNG') 
            print(f"Image re-saved as compatible PNG: {compatible_path}")
            return compatible_path
    except Exception as e:
        print(f"Error ensuring image compatibility: {e}")
        return None

def wait_for_file(file_path, timeout=5):
    initial_size = -1
    for _ in range(timeout):
        current_size = os.path.getsize(file_path)
        if current_size == initial_size:
            return True 
        initial_size = current_size
        time.sleep(1)
    return False

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.png', '.jpg', '.jpeg')):
            full_path = event.src_path
            filename = os.path.basename(full_path)
            print(f"New file detected: {filename}")
            
            if wait_for_file(full_path):
                compatible_path = ensure_image_compatibility(full_path)
                if compatible_path:
                    try:
                        response = ollama.chat(
                            model='llama3.2-vision',
                            messages=[{
                                'role': 'user',
                                'content': 'Act as an assistant for me and act as if I am blind. \
                                            The dist value is the distance from the blind person to the object. Tell me all the objects in the image \
                                            Your response should only include the format object: distance and direction (right, left, front only) \
                                            for all the objects in the frame and dont include any markdown only the text and dont include anything else',
                                'images': [compatible_path]
                            }]
                        )
                        
                        with open(f"./responds/sharpest_frame.txt", "w") as text_file:
                            text_file.write(response.message.content)

                        os.system('rm ./inputs/sharpest_frame.png')

                        print(f"Ollama Response: {response}")
                    except Exception as e:
                        print(f"Error sending file to Ollama: {e}")
                else:
                    print(f"Failed to ensure compatibility for: {filename}")
            else:
                print(f"File {filename} is not stable. Skipping.")

folder_to_monitor = "/home/ubuntu/meta_llama_hacks/inputs"

event_handler = NewFileHandler()
observer = Observer()

observer.schedule(event_handler, folder_to_monitor, recursive=False)
observer.start()

print(f"Monitoring folder: {folder_to_monitor}")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
