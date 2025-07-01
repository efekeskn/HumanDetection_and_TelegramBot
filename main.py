import os
import cv2
import time
import asyncio
import telegram
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CAMERA_URL = os.getenv("IP_CAMERA_URL")

model = YOLO('best.pt') 

NOTIF_COOLDOWN = 60



async def send_alert_async(message: str):
    bot = telegram.Bot(token=BOT_TOKEN)
    async with bot:
        await bot.send_message(chat_id=CHAT_ID, text=message)

async def send_photo_async(caption: str, image: np.ndarray):
    bot = telegram.Bot(token=BOT_TOKEN)
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()

    async with bot:
        await bot.send_photo(chat_id=CHAT_ID, photo=image_bytes, caption=caption)

def send_telegram_alert(message: str):
    print(f"[INFO] Alert gönderiliyor: {message}")
    try:
        asyncio.run(send_alert_async(message))
    except Exception as e:
        print(f"[ERROR] Metin gönderilemedi: {e}")

def send_telegram_photo(message: str, image: np.ndarray):
    print("[INFO] Fotoğraf gönderiliyor.")
    try:
        asyncio.run(send_photo_async(message, image))
    except Exception as e:
        print(f"[ERROR] Fotoğraf gönderilemedi: {e}")



def detect_person(results) -> bool:
    for result in results:
        if len(result.boxes) > 0:
            return True
    return False



def start_detection():
    cap = cv2.VideoCapture(CAMERA_URL)

    if not cap.isOpened():
        print(f"[ERROR] Kamera açılamadı: {CAMERA_URL}")
        return

    person_present = False
    last_notification = 0
    frame_count = 0
    FRAME_SKIP = 3

    print("[INFO] Başlatıldı. 'q' ile çıkabilirsiniz.")

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("[WARNING] Görüntü alınamadı, tekrar denenecek.")
            time.sleep(0.1)
            continue

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        results = model.track(frame, persist=False, conf=0.5, verbose=False)
        person_detected = detect_person(results)
        current_time = time.time()

        annotated_frame = results[0].plot() if results else frame

        if person_detected and not person_present:
            if current_time - last_notification > NOTIF_COOLDOWN:
                send_telegram_photo("🚶 Kişi algılandı.", annotated_frame)
                last_notification = current_time
            person_present = True

        elif not person_detected and person_present:
            person_present = False

        cv2.imshow("Room", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Çıkış yapılıyor.")
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    start_detection()
