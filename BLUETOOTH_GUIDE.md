# Bluetooth Classic Guide - ESP32 to PC (Wireless)

## 🔵 **Overview**

**Bluetooth Classic** adalah cara paling mudah untuk komunikasi wireless antara ESP32 dan PC, karena bekerja seperti **Serial USB biasa** tapi tanpa kabel!

---

## ✅ **Keuntungan Bluetooth Classic:**

| Feature | Serial USB | Bluetooth Classic |
|---------|-----------|-------------------|
| **Kabel** | ❌ Perlu USB | ✅ **Wireless** |
| **Jarak** | ~3m | ~10m |
| **Latency** | ~1ms | ~5-10ms |
| **Setup** | ✅ Plug & play | ⚠️ Pairing sekali |
| **Code** | ✅ Simple | ✅ **Same as Serial!** |
| **Reliability** | ✅ Very stable | ✅ Stable |

**Key Point:** Bluetooth Classic = Serial port wireless! Code Python **TIDAK PERLU DIUBAH**.

---

## 📋 **Setup Instructions**

---

## 🪟 **Windows Setup**

### **Step 1: ESP32 Code (Arduino)**

```cpp
#include "BluetoothSerial.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to enable it
#endif

BluetoothSerial SerialBT;

const int BUTTON_PIN = 4;

int lastButtonState = HIGH;
int currentButtonState;
unsigned long lastDebounceTime = 0;
unsigned long debounceDelay = 50;

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // Initialize Bluetooth with device name
  SerialBT.begin("ESP32_Button"); // Nama yang muncul di PC
  Serial.println("Bluetooth Started! Device name: ESP32_Button");
  Serial.println("Waiting for connection...");
}

void loop() {
  int reading = digitalRead(BUTTON_PIN);
  
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }
  
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != currentButtonState) {
      currentButtonState = reading;
      
      if (currentButtonState == LOW) {
        // Send via Bluetooth (same as Serial!)
        SerialBT.println("disable_image");
        Serial.println("Button pressed - sent via Bluetooth");
      }
    }
  }
  
  lastButtonState = reading;
}
```

**Key Points:**
- ✅ `BluetoothSerial` library sudah built-in di ESP32
- ✅ `SerialBT.println()` sama seperti `Serial.println()`
- ✅ Device name: `ESP32_Button` (bisa diganti)

---

### **Step 2: Pairing ESP32 dengan Windows**

1. **Upload code ke ESP32**
2. **Buka Settings → Bluetooth & devices**
3. **Add device → Bluetooth**
4. **Pilih "ESP32_Button"**
5. **Pair** (PIN biasanya `1234` atau tidak perlu PIN)

**Setelah paired, ESP32 akan muncul sebagai COM port baru!**

#### **Cek COM Port:**

```cmd
# Windows Device Manager
Win + X → Device Manager → Ports (COM & LPT)
```

Anda akan lihat:
```
Standard Serial over Bluetooth link (COM10)  ← ESP32 Bluetooth
```

---

### **Step 3: Configure Python (OPTIONAL)**

Jika Anda ingin **hardcode** Bluetooth COM port di code:

**Edit `main.py` line ~1143:**

```python
if self.button_mode:
    print(">>> BUTTON_MODE enabled")
    # Hardcode Bluetooth COM port
    self.serial_thread = SerialReaderThread(port='COM10')  # Ganti COM10
    self.serial_thread.start()
```

**Atau biarkan auto-detect** (akan detect USB atau Bluetooth):

```python
if self.button_mode:
    print(">>> BUTTON_MODE enabled")
    self.serial_thread = SerialReaderThread()  # Auto-detect
    self.serial_thread.start()
```

---

### **Step 4: Configure .env**

```env
BUTTON_MODE=true
```

**Tidak perlu setting tambahan!** Bluetooth Classic menggunakan Serial port biasa.

---

### **Step 5: Run Application**

```bash
python main.py
```

**Expected output:**

```
>>> BUTTON_MODE enabled - Image will show for 5 seconds then hide
>>> Using Serial communication
>>> Found potential ESP32 at COM10: Standard Serial over Bluetooth link
>>> Serial connected to COM10 at 115200 baud
>>> Serial reader thread started
```

---

## 🧪 **Testing**

### **Test 1: Check Connection**

**ESP32 Serial Monitor:**
```
Bluetooth Started! Device name: ESP32_Button
Waiting for connection...
```

**PC (after pairing):**
```
>>> Serial connected to COM10 at 115200 baud
```

### **Test 2: Button Press**

**ESP32:**
```
Button pressed - sent via Bluetooth
```

**PC:**
```
>>> Serial received: disable_image
>>> Button pressed - Image displayed (will hide after 5 seconds)
```

---

## 🐧 **Linux (Ubuntu) Setup**

### **Step 1: Install Dependencies**

```bash
# Install Bluetooth tools
sudo apt update
sudo apt install bluetooth bluez bluez-tools rfkill

# Check Bluetooth status
sudo systemctl status bluetooth

# Unblock Bluetooth (if blocked)
sudo rfkill unblock bluetooth
```

---

### **Step 2: ESP32 Code (Same as Windows)**

Upload the same ESP32 code as Windows (see above).

---

### **Step 3: Pairing ESP32 dengan Ubuntu**

#### **Method 1: GUI (Easiest)**

1. **Upload code ke ESP32**
2. **Open Settings → Bluetooth**
3. **Turn ON Bluetooth**
4. **Find "ESP32_Button"**
5. **Click "Connect" or "Pair"**
6. **PIN:** Usually `1234` or no PIN needed

#### **Method 2: Command Line (bluetoothctl)**

```bash
# Start Bluetooth service
sudo systemctl start bluetooth
sudo systemctl enable bluetooth

# Open bluetoothctl
bluetoothctl

# Inside bluetoothctl:
power on
agent on
default-agent
scan on

# Wait for ESP32_Button to appear
# Note the MAC address, e.g., 30:AE:A4:XX:XX:XX

# Pair with ESP32
pair 30:AE:A4:XX:XX:XX
trust 30:AE:A4:XX:XX:XX
connect 30:AE:A4:XX:XX:XX

# Exit
exit
```

---

### **Step 4: Bind Bluetooth to Serial Port**

After pairing, bind ESP32 to serial port:

```bash
# Check ESP32 MAC address
hcitool scan

# Output:
# Scanning ...
#   30:AE:A4:XX:XX:XX  ESP32_Button

# Bind to RFCOMM (serial port)
sudo rfcomm bind /dev/rfcomm0 30:AE:A4:XX:XX:XX 1

# Check if successful
ls -l /dev/rfcomm0

# Output:
# crw-rw---- 1 root dialout 216, 0 Oct 20 18:21 /dev/rfcomm0
```

**Bluetooth serial port:** `/dev/rfcomm0`

---

### **Step 5: Give Permission**

```bash
# Add user to dialout group (for serial port access)
sudo usermod -a -G dialout $USER

# Logout and login again, or:
newgrp dialout

# Set permission for rfcomm0
sudo chmod 666 /dev/rfcomm0
```

---

### **Step 6: Auto-Bind on Boot (Optional)**

Create systemd service to auto-bind on boot:

```bash
# Create service file
sudo nano /etc/systemd/system/rfcomm-bind.service
```

Add this content (replace MAC address):

```ini
[Unit]
Description=RFCOMM Bind ESP32 Bluetooth
After=bluetooth.service

[Service]
Type=oneshot
ExecStart=/usr/bin/rfcomm bind /dev/rfcomm0 30:AE:A4:XX:XX:XX 1
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Enable service:

```bash
sudo systemctl enable rfcomm-bind.service
sudo systemctl start rfcomm-bind.service
```

---

### **Step 7: Run Application**

```bash
python main.py
```

**Expected output:**

```
>>> BUTTON_MODE enabled - Image will show for 5 seconds then hide
>>> Using Serial communication
>>> Found Bluetooth RFCOMM at /dev/rfcomm0
>>> Serial connected to /dev/rfcomm0 at 115200 baud
>>> Serial reader thread started
```

---

### **Step 8: Test Connection**

```bash
# Test serial communication
echo "test" > /dev/rfcomm0

# Monitor serial data
cat /dev/rfcomm0
```

---

## 🐛 **Troubleshooting**

### **Problem 1: ESP32 tidak muncul di Bluetooth**

**Solutions:**
1. ✅ Check ESP32 sudah upload code dengan `BluetoothSerial`
2. ✅ Restart ESP32 (tekan tombol RST)
3. ✅ Check Bluetooth PC sudah ON
4. ✅ Jarak ESP32 dan PC < 10m

---

### **Problem 2: Pairing failed**

**Solutions:**
1. ✅ Remove device dari PC, lalu pair ulang
2. ✅ Restart Bluetooth di PC
3. ✅ Coba pairing dari ESP32 ke PC (bukan sebaliknya)

---

### **Problem 3: COM port tidak muncul**

**Symptoms:**
```
>>> No ESP32 device found
```

**Solutions:**
1. ✅ Check pairing berhasil (lihat di Bluetooth settings)
2. ✅ Check Device Manager → Ports (COM & LPT)
3. ✅ Restart PC setelah pairing
4. ✅ Install Bluetooth driver jika perlu

---

### **Problem 4: Connection unstable**

**Solutions:**
1. ✅ Jarak ESP32 dan PC terlalu jauh (max ~10m)
2. ✅ Interference dari WiFi/device lain
3. ✅ Battery ESP32 lemah (gunakan power supply)
4. ✅ Restart ESP32 dan re-pair

---

### **Problem 5: Linux - /dev/rfcomm0 not found**

**Solutions:**

```bash
# Check if Bluetooth is running
sudo systemctl status bluetooth

# Check paired devices
bluetoothctl devices

# Manually bind
sudo rfcomm bind /dev/rfcomm0 30:AE:A4:XX:XX:XX 1

# Check binding
sudo rfcomm show all
```

---

### **Problem 6: Linux - Permission denied**

**Solutions:**

```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER

# Logout and login

# Or set permission manually
sudo chmod 666 /dev/rfcomm0

# Check permission
ls -l /dev/rfcomm0
```

---

### **Problem 7: Linux - Bluetooth blocked**

**Solutions:**

```bash
# Check if blocked
rfkill list

# Unblock all
sudo rfkill unblock all

# Restart Bluetooth
sudo systemctl restart bluetooth
```

---

### **Problem 8: Linux - Connection drops**

**Solutions:**

```bash
# Keep connection alive with auto-reconnect script
nano ~/reconnect_bt.sh
```

Add this content:

```bash
#!/bin/bash
MAC="30:AE:A4:XX:XX:XX"  # Replace with your ESP32 MAC

while true; do
    if ! rfcomm show /dev/rfcomm0 &> /dev/null; then
        echo "Reconnecting..."
        sudo rfcomm bind /dev/rfcomm0 $MAC 1
        sudo chmod 666 /dev/rfcomm0
    fi
    sleep 5
done
```

Make executable and run:

```bash
chmod +x ~/reconnect_bt.sh
./reconnect_bt.sh &
```

---

## 📊 **Comparison: USB vs Bluetooth**

### **Serial USB**

```python
# Auto-detect USB
self.serial_thread = SerialReaderThread()
```

**Pros:**
- ✅ Lower latency (~1ms)
- ✅ More stable
- ✅ No pairing needed

**Cons:**
- ❌ Need USB cable
- ❌ Limited range (~3m)

---

### **Bluetooth Classic**

```python
# Auto-detect Bluetooth (or hardcode port)
self.serial_thread = SerialReaderThread(port='COM10')
```

**Pros:**
- ✅ **Wireless** (no cable!)
- ✅ Longer range (~10m)
- ✅ **Same code as Serial!**
- ✅ Easy setup (pair once)

**Cons:**
- ⚠️ Slightly higher latency (~5-10ms)
- ⚠️ Need pairing first
- ⚠️ Limited to 1 device

---

## 🔧 **Advanced: Multiple ESP32**

Untuk multiple ESP32 dengan Bluetooth:

**Option 1: Multiple Bluetooth adapters di PC**
- Setiap ESP32 pair ke adapter berbeda
- Masing-masing dapat COM port berbeda

**Option 2: Switch between devices**
- Unpair device 1, pair device 2
- Ganti COM port di code

**Option 3: Use WiFi WebSocket instead**
- Lebih cocok untuk multiple devices

---

## 🎯 **Recommended Use Cases**

| Scenario | Recommended |
|----------|-------------|
| **Lab testing (dekat PC)** | Serial USB |
| **Demo (jarak dekat)** | Bluetooth Classic ⭐ |
| **Production (fixed)** | Serial USB |
| **Production (mobile)** | Bluetooth Classic ⭐ |
| **Multiple buttons** | WiFi WebSocket |
| **Outdoor/Long range** | WiFi WebSocket |

---

## 💡 **Tips & Tricks**

### **Tip 1: Auto-reconnect**

ESP32 Bluetooth akan auto-reconnect jika koneksi terputus:

```cpp
void loop() {
  if (!SerialBT.connected()) {
    Serial.println("Bluetooth disconnected, waiting...");
    delay(1000);
    return;
  }
  
  // Normal code here
}
```

### **Tip 2: Check Connection Status**

```cpp
if (SerialBT.hasClient()) {
  SerialBT.println("disable_image");
} else {
  Serial.println("No Bluetooth client connected");
}
```

### **Tip 3: Custom Device Name**

```cpp
// Ganti nama device untuk identifikasi lebih mudah
SerialBT.begin("BDT_Button_1");  // Muncul sebagai "BDT_Button_1"
```

### **Tip 4: Battery Indicator**

```cpp
void setup() {
  SerialBT.begin("ESP32_Button");
  
  // Send battery level via Bluetooth
  int batteryLevel = analogRead(35); // Assuming battery on pin 35
  SerialBT.printf("Battery: %d%%\n", batteryLevel);
}
```

---

## ✅ **Summary**

| Feature | Status |
|---------|--------|
| Wireless communication | ✅ Bluetooth Classic |
| No code change needed | ✅ Same as Serial |
| Easy setup | ✅ Pair once |
| Range | ✅ ~10m |
| Latency | ✅ ~5-10ms |
| Stability | ✅ Good |

**Bluetooth Classic adalah pilihan terbaik untuk:**
- ✅ Wireless tanpa kompleksitas WiFi
- ✅ Jarak dekat (~10m)
- ✅ Single device
- ✅ Code yang sama dengan Serial USB

**Cara pakai:**
1. Upload ESP32 code dengan `BluetoothSerial`
2. Pair ESP32 dengan PC (sekali saja)
3. Run `python main.py` (auto-detect Bluetooth COM port)
4. Done! 🎉

---

## 📝 **Complete ESP32 Code**

```cpp
#include "BluetoothSerial.h"

BluetoothSerial SerialBT;
const int BUTTON_PIN = 4;

int lastButtonState = HIGH;
int currentButtonState;
unsigned long lastDebounceTime = 0;
unsigned long debounceDelay = 50;

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  SerialBT.begin("ESP32_Button");
  Serial.println("Bluetooth Started!");
}

void loop() {
  int reading = digitalRead(BUTTON_PIN);
  
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }
  
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != currentButtonState) {
      currentButtonState = reading;
      
      if (currentButtonState == LOW) {
        SerialBT.println("disable_image");
        Serial.println("Button pressed");
      }
    }
  }
  
  lastButtonState = reading;
}
```

**That's it!** Code Python tidak perlu diubah sama sekali. 🚀
