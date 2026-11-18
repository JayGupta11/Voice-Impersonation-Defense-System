# 🛡️ AI Voice Impersonation Defense System (SIH 2026)

A real-time, AI-powered voice deepfake and impersonation defense system built for **Smart India Hackathon (SIH) 2026**. The system pairs a custom Google Dialer-style mobile application (React Native & Expo) with a high-performance FastAPI backend powered by an **XGBoost + ECAPA-TDNN** audio analysis pipeline.

---

## ✨ Key Features

* **Real-Time Call Stream Analysis:** Processes live incoming audio streams via WebSockets in 500ms chunks for low-latency deepfake detection.
* **Custom Mobile Dialer UI:** Modern dark-mode dialer featuring a keypad, device contact integration, search bar, and an animated active call alert screen.
* **Offline Media Analyzer (Scan Tab):** Upload recorded calls or voice notes (`.wav`, `.mp3`) directly from device storage for forensic AI authenticity verification.
* **Dynamic Risk Scoring & Alerts:** Real-time risk percentage display with visual pulse animations and automated severity classification (**Safe**, **Suspicious**, **Critical**).
* **Built-in Demo Speed Dials:** Instant stage-ready test simulations using `*888` (Authentic Human Voice) and `*999` (AI Scam / Cloned Voice).

---

## 🏗️ Tech Stack

* **Mobile App (Frontend):** React Native, Expo, TypeScript, `expo-audio`, `expo-contacts`, `expo-document-picker`, `expo-file-system/legacy`.
* **API Server (Backend):** Python 3.9+, FastAPI, Uvicorn, WebSockets, Pydantic.
* **AI & Audio Processing:** XGBoost, ECAPA-TDNN speaker embeddings, Librosa (16 kHz mono resampling), NumPy.

---

## 📂 Project Structure

```text
Voice_Impersonation_Defense/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI server (WebSockets + REST endpoints)
│   └── core/
│       └── pipeline.py          # XGBoost / ECAPA-TDNN inference pipeline
├── data/
│   ├── human_call.wav           # Server-side demo audio (*888)
│   └── scam_call.wav            # Server-side demo audio (*999)
├── mobile-dialer/
│   ├── assets/
│   │   ├── human_call.wav       # Local playback audio (*888)
│   │   └── scam_call.wav        # Local playback audio (*999)
│   ├── App.tsx                  # Main React Native dialer & scanner app
│   ├── app.json                 # Expo configuration
│   └── package.json             # Node dependencies
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 📋 Prerequisites

Ensure the following tools are installed before starting:

* **Python 3.9 or higher**
* **Node.js (v18 LTS or higher)** and **npm**
* **Git**
* **Expo Go** installed on an Android or iOS smartphone
* **USB Cable** (Recommended for USB Tethering during demos to ensure stable, zero-latency local networking)

---

## ⚙️ Step 1: Backend Setup (FastAPI + AI Pipeline)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JayGupta11/Voice-Impersonation-Defense-System.git
   cd Voice_Impersonation_Defense
   ```

2. **Create and activate a Python virtual environment:**
   * **Windows (PowerShell):**
     ```powershell
     python -m venv videnv
     .\videnv\Scripts\activate
     ```
   * **macOS / Linux:**
     ```bash
     python3 -m venv videnv
     source videnv/bin/activate
     ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is not yet generated, install the core server packages manually:)*
   ```bash
   pip install fastapi uvicorn numpy librosa soundfile python-multipart pydantic websockets
   ```

4. **Verify demo audio files:**
   Make sure the `data/` directory at the project root contains:
   * `data/human_call.wav`
   * `data/scam_call.wav`

5. **Start the FastAPI backend server:**
   Run the server from the **project root directory** so module imports resolve cleanly:
   ```bash
   python -m backend.api.main
   ```
   You should see:
   ```text
   INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
   ```

---

## 📱 Step 2: Frontend Setup (React Native / Expo)

1. **Open a second terminal** and navigate to the mobile app folder:
   ```bash
   cd mobile-dialer
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```
   Ensure the required Expo native modules are installed:
   ```bash
   npx expo install expo-file-system expo-document-picker expo-contacts expo-audio
   ```

3. **Allow local HTTP traffic on Android:**
   Open `mobile-dialer/app.json` and ensure `usesCleartextTraffic` is enabled inside the `android` section:
   ```json
   {
     "expo": {
       "android": {
         "usesCleartextTraffic": true
       }
     }
   }
   ```

4. **Verify frontend audio assets:**
   Ensure copies of the demo `.wav` files are placed in:
   * `mobile-dialer/assets/human_call.wav`
   * `mobile-dialer/assets/scam_call.wav`

5. **Configure the backend IP address in `App.tsx`:**
   * Connect your phone and laptop to the same network (or enable **USB Tethering** on your Android phone).
   * Find your laptop's local IPv4 address:
     * **Windows:** Run `ipconfig` and copy the **IPv4 Address** under your active Wi-Fi or USB Ethernet/RNDIS adapter (e.g., `192.168.42.129` or `10.246.53.232`).
     * **macOS / Linux:** Run `ifconfig` or `ip a`.
   * Open `mobile-dialer/App.tsx` and update the connection variables using **plain `ws://` and `http://`** with port `8000`:
     ```tsx
     const LOCAL_HOST = 'YOUR_LAPTOP_IPV4:8000'; // e.g., 192.168.42.129:8000
     const NGROK_WS = `ws://${LOCAL_HOST}`;
     const NGROK_HTTP = `http://${LOCAL_HOST}`;
     ```

6. **Start the Expo development server:**
   ```bash
   npx expo start -c
   ```
   *(The `-c` flag clears the Metro bundler cache.)*

7. **Launch on your phone:**
   Open the **Expo Go** app on your smartphone and scan the QR code displayed in the terminal.

---

## 🌐 Alternative: Public Tunnel Setup (Two-Phone / Remote Testing)

If you need to test across different networks without sharing a local IP or USB tether, use **Cloudflare Tunnel** instead of Ngrok's free tier (Ngrok's free tier serves an HTML interstitial warning page that causes `403 Forbidden` errors on WebSocket handshakes).

1. Run Cloudflare Tunnel against port `8000`:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
2. Copy the generated `*.trycloudflare.com` domain into `App.tsx` using secure schemes (`wss://` and `https://`):
   ```tsx
   const TUNNEL_DOMAIN = 'your-generated-subdomain.trycloudflare.com';
   const NGROK_WS = `wss://${TUNNEL_DOMAIN}`;
   const NGROK_HTTP = `https://${TUNNEL_DOMAIN}`;
   ```

---

## 🔌 API Endpoints

| Method / Protocol | Endpoint | Description |
| :--- | :--- | :--- |
| `WebSocket` | `/ws/stream/{call_id}` | Bi-directional real-time audio byte routing and JSON risk score broadcasting. Supports `{"action": "start_demo", "type": "scam" \| "human"}` commands. |
| `WebSocket` | `/ws/stream/frontend_monitor` | Connects an external monitoring dashboard to receive live broadcast risk scores across calls. |
| `POST` | `/api/analyze-audio` | Multipart form-data endpoint for uploading `.wav`/`.mp3` files for chunked AI analysis. |
| `POST` | `/api/analyze-audio-base64` | JSON endpoint accepting `{ "filename": string, "file_data": base64_string }` to reliably bypass Android scoped-storage upload locks. |

---

## 🧪 Running a Live Demo

### 1. Register on the Dialer
* Open the app and enter any identity number (4+ digits) on the verification keypad, then tap **Continue**.

### 2. Real-Time Call Simulation (`Keypad` Tab)
* **Safe Call Test:** Dial `*888` and tap the green call button. The app plays the authentic human voice sample while streaming synchronized 500ms inference scores from FastAPI (**Voice Biometrics Verified**).
* **Deepfake Scam Test:** Dial `*999` and tap the call button. As the AI-cloned voice plays, the risk score spikes above 50–75%, turning the call screen red with a pulsing **Voice Impersonation Detected** warning.

### 3. Live Voice / Recorded File Verification (`Scan` Tab)
* Record a short voice clip on your phone's default Voice Recorder app (or use an existing `.mp3`/`.wav` file).
* Open the **Scan** tab, tap the upload box, and select the audio file.
* Tap **Run AI Analysis** to run the full file through the backend XGBoost/ECAPA-TDNN pipeline and view the calculated risk score.

---

## 🛠️ Troubleshooting

| Error / Issue | Cause | Solution |
| :--- | :--- | :--- |
| `WARNING: Invalid HTTP request received.` in Uvicorn | The mobile app is sending encrypted `wss://` or `https://` requests to a plain local Uvicorn server. | Switch `App.tsx` URLs to plain `ws://` and `http://` when pointing to a local IPv4 address. |
| `403 Forbidden` on WebSocket via Ngrok | Ngrok free tier intercepts requests with an anti-phishing HTML warning screen. | Switch to a local USB Tethering IPv4 address or use `cloudflared tunnel --url http://localhost:8000`. |
| `java.io.IOException: Location ... isn't readable` | Android scoped storage / Expo Go cache restrictions block direct binary multipart reads from `DocumentPicker`. | Use `FileSystem.readAsStringAsync(..., { encoding: FileSystem.EncodingType.Base64 })` and POST to `/api/analyze-audio-base64`. |
| `Method uploadAsync imported from "expo-file-system" is deprecated` | Newer Expo SDKs moved legacy methods to a dedicated subpath. | Change import to `import * as FileSystem from 'expo-file-system/legacy';`. |
| `ExpoAsset.downloadAsync has been rejected` (Font crash) | Phone cannot fetch `.ttf` icon fonts from Metro bundler over a restricted router or slow tunnel. | Connect via Mobile Hotspot or USB Tethering, force-close Expo Go, and restart Metro with `npx expo start -c`. |

---

## 🚀 Future Roadmap (Native VoIP)

Expo Go restricts microphone access to file-based recordings rather than continuous raw PCM byte streams. To enable true live two-way VoIP microphone streaming between two physical devices:

1. Eject from Expo Go to a **Custom Development Build** (`npx expo run:android`).
2. Integrate `react-native-live-audio-stream` (or WebRTC) to capture continuous 16 kHz PCM audio chunks directly from Android's native audio layer and stream them over `/ws/stream/{call_id}`.
