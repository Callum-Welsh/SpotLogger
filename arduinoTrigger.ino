// --- Arduino Trigger Detector Sketch ---

const byte INTERRUPT_PIN = 2; // Using Digital Pin 2 for external interrupt
volatile boolean triggerFlag = false; // Flag set by ISR, 'volatile' is important!

void setup() {
  // Start Serial communication at a reasonably fast speed
  Serial.begin(115200);
  while (!Serial); // Wait for Serial port to connect (needed for some boards like Leonardo, not strictly for Uno but good practice)

  // Configure the interrupt pin as input
  // No pullup needed if the TTL signal is actively driven high and low.
  pinMode(INTERRUPT_PIN, INPUT);

  // Attach the interrupt:
  // digitalPinToInterrupt(INTERRUPT_PIN) -> Maps pin 2 to the correct internal interrupt number
  // handleTrigger -> The function to call when the interrupt occurs (ISR)
  // RISING -> Trigger on the rising edge (change to FALLING or CHANGE if needed)
  attachInterrupt(digitalPinToInterrupt(INTERRUPT_PIN), handleTrigger, RISING);

  pinMode(LED_BUILTIN, OUTPUT); // Use onboard LED for feedback
  digitalWrite(LED_BUILTIN, LOW); // Start with LED off

  Serial.println("Arduino Trigger Detector Ready. Waiting for rising edge on Pin 2...");
}

void loop() {
  // Check if the ISR has set the flag
  if (triggerFlag) {
    // Send the trigger character over Serial to the PC
    Serial.write('!');


    // Reset the flag *immediately* after processing
    triggerFlag = false;
    Serial.println("");
    Serial.println("WAIT");

    // Optional: Quick blink of the onboard LED to show the trigger was detected and sent
    digitalWrite(LED_BUILTIN, HIGH);
    delay(20); // Keep delays short
    digitalWrite(LED_BUILTIN, LOW);
  }

  // The main loop can do other non-blocking things here if needed,
  // or just stay empty. The interrupt handles the trigger detection.
}

// --- Interrupt Service Routine (ISR) ---
// This function is called automatically by the hardware when the RISING edge occurs on Pin 2
// Keep ISRs as short and fast as possible! Avoid delays or complex operations here.
void handleTrigger() {
  // Just set the flag. The main loop() will handle the Serial communication.
  // This is safer than putting Serial.write directly inside the ISR.
  triggerFlag = true;
}