#include <WiFi.h>
#include <HTTPClient.h>

// Wi-Fi credentials
const char* ssid = "RB";
const char* password = "password";

// Flask server URL
const char* serverUrl = "http://192.168.155.144:5001/button";

// Button pin
const int buttonPin = 14;
bool buttonPressed = false;

void setup() {
  Serial.begin(115200);

  // Initialize the button pin as input
  pinMode(buttonPin, INPUT_PULLUP);

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("\nConnected to Wi-Fi");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  // Read the button state
  int buttonState = digitalRead(buttonPin);

  // If the button is pressed and was not already detected
  if (buttonState == LOW && !buttonPressed) {
    buttonPressed = true;
    Serial.println("Button Pressed!");

    // Send GET request to Flask server
    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      http.begin(serverUrl);
      int httpResponseCode = http.GET();
      if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.println("Server Response:");
        Serial.println(response);
      } else {
        Serial.print("Error on sending GET: ");
        Serial.println(httpResponseCode);
      }
      http.end();
    }
  } else if (buttonState == HIGH) {
    // Reset buttonPressed when the button is released
    buttonPressed = false;
  }
}
