from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from tracker import Tracker
import asyncio
import traceback

# Initialize YOLO model and Tracker
model = YOLO('best3.pt')
tracker = Tracker()

entered_ids = set()
initial_passenger_count = 0
with open("coco1.txt", "r") as my_file:
    class_list = my_file.read().splitlines()

app = FastAPI()

async def process_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=404, detail=f"Cannot open video at {video_path}")
    try:
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if count % 2 != 0:
                continue
            frame = cv2.resize(frame, (1020, 500))
            results = model.predict(frame)
            a = results[0].boxes.data
            px = pd.DataFrame(a).astype("float")
            bbox_list = [[int(row[0]), int(row[1]), int(row[2]), int(row[3])] for index, row in px.iterrows()]
            bbox_idx = tracker.update(bbox_list)
            for bbox in bbox_idx:
                x3, y3, x4, y4, id = bbox
                if id not in entered_ids:
                    entered_ids.add(id)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
            total_passenger_count = initial_passenger_count + len(entered_ids)
            cvzone.putTextRect(frame, f'Total Passengers: {total_passenger_count}', (50, 50), 2, 2)
            _, img_encoded = cv2.imencode('.jpg', frame)
            yield img_encoded.tobytes()
            await asyncio.sleep(0)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cap.release()
        cv2.destroyAllWindows()
        final_total_passenger_count = initial_passenger_count + len(entered_ids)
        print(f"Final Total Passengers: {final_total_passenger_count}")

@app.get("/process_video/")
async def process_video_api(video_path: str = Query(..., description="Path to the video file")):
    try:
        async def generate():
            async for frame_bytes in process_video(video_path):
                yield frame_bytes
        return StreamingResponse(generate(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/total_passengers/")
async def total_passengers_api():
    try:
        total_passenger_count = initial_passenger_count + len(entered_ids)
        return JSONResponse(content={"total_passengers": total_passenger_count})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video_frame/")
async def video_frame_api(video_path: str = Query(..., description="Path to the video file")):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=404, detail=f"Cannot open video at {video_path}")
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            _, img_encoded = cv2.imencode('.jpg', frame)
            yield StreamingResponse(content=img_encoded.tobytes(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cap.release()
        cv2.destroyAllWindows()

