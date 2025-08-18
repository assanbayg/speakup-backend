# SpeakUp
**Features**:
- account deletion
- coqui tts integration (in the future)

**How to run locally**:
```
git clone https://github.com/assanbayg/speakup-backend.git
cd speakup-backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```