#include <WiFi.h>
#include <HTTPClient.h>
#include "Audio.h"

// Wi-Fi credentials
const char* ssid = "RB";
const char* password = "password";

// Flask server endpoint
const char* serverName = "http://192.168.155.144:5001/button";

// Button setup
const int buttonPin = 14;  
int buttonState = 0;       
int lastButtonState = 0;   

// TCP server setup
WiFiServer tcpServer(80);

// Speaker setup
#define I2S_DOUT 25
#define I2S_BCLK 27
#define I2S_LRC  26
Audio audio;

void setup() {
  Serial.begin(115200);
  pinMode(buttonPin, INPUT);

  // Connect to Wi-Fi
  Serial.println("Connecting to Wi-Fi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nConnected to Wi-Fi!");
  Serial.print("ESP32 IP address: ");
  Serial.println(WiFi.localIP());

  // Start TCP server
  tcpServer.begin();

  // Initialize speaker
  audio.setPinout(I2S_BCLK, I2S_LRC, I2S_DOUT);
  audio.setVolume(1); // Set volume level
}

void loop() {
  // Handle button press
  buttonState = digitalRead(buttonPin);

  if (buttonState == HIGH && lastButtonState == LOW) {
    Serial.println("Button pressed!");

    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      Serial.print("Notifying Flask server: ");
      Serial.println(serverName);

      http.begin(serverName);
      int httpResponseCode = http.GET();

      if (httpResponseCode > 0) {
        Serial.print("HTTP Response code: ");
        Serial.println(httpResponseCode);
      } else {
        Serial.print("Error code: ");
        Serial.println(http.errorToString(httpResponseCode).c_str());
      }

      http.end();
    } else {
      Serial.println("Wi-Fi disconnected!");
    }
  }

  lastButtonState = buttonState;

  // Handle incoming TCP connections
  WiFiClient client = tcpServer.available(); // Check for incoming clients

  if (client) {
    Serial.println("New Client Connected");
    while (client.connected()) {
      if (client.available()) {
        String receivedData = client.readStringUntil('\n'); // Read data from the client
        Serial.println("Received: " + receivedData);

        // Play the received string on the spefaker using TTS
        audio.connecttospeech(receivedData.c_str(), "en"); // Google TTS
        while (audio.isRunning()) {
          audio.loop();
        }

        client.println("Message received and played!"); // Acknowledge to the client
      }
    }
    client.stop(); // Close the connection
    Serial.println("Client Disconnected");
  }

  delay(50); // Debounce delay
}
