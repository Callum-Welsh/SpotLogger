// Simple Arduino sketch to send "TRIGGER" over serial when a pin goes HIGH

const int triggerPin = 2; // Pin to listen for TTL trigger (use interrupt pin if possible)
volatile bool triggered = false;

void setup() {
  Serial.begin(9600); // Match baud rate in Python
  pinMode(triggerPin, INPUT_PULLUP); // Use INPUT_PULLUP if trigger pulls LOW, or INPUT if trigger drives HIGH
  // Using an interrupt is more reliable for fast triggers:
  // attachInterrupt(digitalPinToInterrupt(triggerPin), handleTrigger, RISING); // Or FALLING
  Serial.println("Arduino Ready");
}

// // --- Interrupt Service Routine (if using attachInterrupt) ---
// void handleTrigger() {
//   triggered = true;
// }
// // ----------------------------------------------------------

void loop() {
  // --- Polling method (simpler, might miss fast triggers) ---
  if (digitalRead(triggerPin) == HIGH) { // Adjust HIGH/LOW based on your trigger signal
     Serial.println("TRIGGER");
     delay(100); // Basic debounce / prevent rapid re-triggering
  }
  // ----------------------------------------------------------

  // // --- Interrupt method (use with attachInterrupt) ---
  // if (triggered) {
  //   Serial.println("TRIGGER");
  //   triggered = false; // Reset flag
  //   // Optional: add delay or disable interrupt briefly for debouncing
  // }
  // // --------------------------------------------------

  delay(10); // Small delay to prevent overwhelming serial port if polling
}