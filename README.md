# Custom Video Broadcaster

A professional, AI-powered video broadcasting platform for real-time background effects, person segmentation, and seamless virtual camera output. Built with FastAPI, OpenCV, Ultralytics YOLO, and pyvirtualcam, this solution is designed for reliability, flexibility, and ease of use in demanding environments.

---

![App Interface](static/Screenshot%202025-07-06%20022333.png)

---

## ✨ Features
- **Real-time person segmentation** using YOLOv8 for accurate subject detection.
- **Multiple background effects:** blur, black, custom image, or virtual background.
- **Virtual camera output** for integration with Zoom, Teams, OBS, Google Meet, and more.
- **Modern web-based control panel** for device selection and effect configuration.
- **FastAPI backend** for robust, scalable, and easy integration.
- **Cross-platform support** (Windows, macOS, Linux with compatible drivers).

---

## 🚀 Quick Start
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the server:**
   ```bash
   python main.py
   ```
3. **Open your browser:**
   Go to [http://localhost:8002](http://localhost:8002) and start broadcasting!

---

## 🏢 Professional Use Cases & Industries
This platform is ideal for:
- **Remote Work & Video Conferencing:**
  - Corporate professionals, executives, and teams who need privacy, branding, or a distraction-free environment in Zoom, Teams, Meet, and more.
- **Education & Online Training:**
  - Teachers, lecturers, and trainers delivering online classes with custom or blurred backgrounds for privacy and engagement.
- **Content Creation & Streaming:**
  - YouTubers, Twitch streamers, podcasters, and influencers who want dynamic, branded, or themed backgrounds.
- **Broadcasting & News:**
  - News anchors, reporters, and studios using virtual sets or privacy backgrounds for live or recorded segments.
- **Healthcare & Telemedicine:**
  - Doctors, therapists, and healthcare professionals ensuring patient privacy and a professional appearance during virtual consultations.
- **Customer Service & Call Centers:**
  - Support teams and agents with branded or neutral backgrounds for a consistent, professional look.
- **Events, Webinars & Online Conferences:**
  - Hosts, speakers, and panelists with themed, sponsor, or event-specific backgrounds.
- **Recruitment & HR:**
  - Interviewers and candidates maintaining privacy and professionalism during virtual interviews.
- **Legal & Financial Services:**
  - Lawyers, consultants, and advisors ensuring confidentiality and a polished presence.

---

## 📦 Project Structure
- `main.py` — FastAPI server and endpoints
- `stream_utils.py` — Streaming logic and device management
- `engine.py` — AI segmentation and background processing
- `static/` — Web UI and assets

---

## 🛠️ Troubleshooting & Tips
- **Camera not detected or not working?**
  - Ensure your webcam is connected and not in use by another app.
  - Try running as administrator or check your system's privacy/camera settings.
  - Update your camera drivers if you see low-level errors in the logs.
  - If you see OpenCV/driver errors, these are now suppressed to only show critical issues.
- **Missing background image?**
  - If the custom background image is missing, the app will use a black background and print a warning.
- **Favicon and UI screenshot:**
  - The web UI includes a favicon and a screenshot for a polished, professional look.
- **Logging and error handling:**
  - The backend suppresses verbose OpenCV logs and provides clear error messages for device and streaming issues.

---

## 🤝 Contributing & Support
- Pull requests and suggestions are welcome! Please open an issue for bugs or feature requests.
- For help, contact the maintainer or open a GitHub issue.

---

## ❤️ Credits
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenCV](https://opencv.org/)

---

## 📄 License
MIT License
