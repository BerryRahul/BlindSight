#include <WiFi.h>

const char* ssid = "RB";
const char* password = "password";

WiFiServer server(80); // Create a server on port 80

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  
  Serial.println("Connected to WiFi");
  Serial.println(WiFi.localIP()); // Print the ESP32's IP address
  
  server.begin(); // Start the server
}

void loop() {
  WiFiClient client = server.available(); // Check for incoming clients
  
  if (client) {
    Serial.println("New Client Connected");
    while (client.connected()) {
      if (client.available()) {
        String data = client.readStringUntil('\n'); // Read data from the client
        Serial.println("Received: " + data);
        client.println("Message received!"); // Optional acknowledgment to the client
      }
    }
    client.stop(); // Close the connection
    Serial.println("Client Disconnected");
  }
}
