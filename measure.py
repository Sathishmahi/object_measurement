from torchvision.ops import box_convert
import numpy as np
from IPython.display import clear_output
from groundingdino.util.inference import load_model, load_image, predict
import torch
import cv2
import os
import wget
from tqdm import tqdm
import subprocess
import shutil

weight_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"

def convert_video(input_video,output_video):
    if not os.path.exists(output_video):
        command = f"ffmpeg -i {input_video} -vcodec libx264 {output_video}"
        try:
            subprocess.run(command, shell=True, check=True)
            print("Video conversion completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Video conversion failed with error: {e}")
    else:print(f"OUTPUT FOUND : {output_video}")

def annotate(image_source: np.ndarray, boxes: torch.Tensor) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(np.int32)
    return xyxy

def download_wight(weight_dir_name):
  os.makedirs(weight_dir_name,exist_ok=True)
  WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
  weight_path = os.path.join("weights", WEIGHTS_NAME)
  if not os.path.exists(weight_path):
    wget.download(weight_url,out = weight_path)
  else:print("WEIGHT FILE ALREADY DOWNLOADED")
  
  return weight_path



model = load_model(CONFIG_PATH, download_wight("weights"))


actual_obj_height_cm,actual_obj_height_pix = 23.5,351
one_pix_cm = 23.5/351
def height_measurement(y_diff):
  return y_diff*one_pix_cm

def detector(video_path,TEXT_PROMPT,
            BOX_TRESHOLD,TEXT_TRESHOLD,out_video_file_path,out_display_video_path):
  global frame_first
  cap =  cv2.VideoCapture(video_path)
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  writer = cv2.VideoWriter(out_video_file_path,fourcc,30,
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
  for c in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    suc,frame = cap.read()
    cv2.imwrite("temp.jpg",frame)


    image_source, image = load_image("temp.jpg")

    # img = cv2.cvtColor(cv2.imread("/content/Om2.jpg"), cv2.COLOR_BGR2RGB)/255.
    boxes, logits, phrases = predict(
        model=model,
        # image=torch.tensor(img,dtype=torch.float32),
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    boxes = annotate(image_source,boxes)
    for bb,lo,ph in zip(boxes, logits, phrases ):
      r,c,_ = frame.shape
      # x1,y1,x2,y2 = [  int(v*c) if not i%2 else int(v*r)   for i,v in enumerate(bb,start=1)]
      x1,y1,x2,y2 = bb
      cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),1)

      # id = tracking.track((x1,y1,x2,y2 ),frame_first)
      frame_first = False
      y_diff = y2-y1
      obj_hei = height_measurement(y_diff)
      cv2.putText(frame,f"# {obj_hei:.1f} cm",(x1,y1-5),cv2.FONT_HERSHEY_COMPLEX,1.2,(0,0,255),3)
    # frame = annotate(image_source,boxes, logits, phrases)
    writer.write(frame)
    clear_output()

  writer.release()
  cap.release()
  convert_video(out_video_file_path,out_display_video_path)

