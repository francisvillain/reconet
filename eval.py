import numpy as np
import cv2
import torch
from model2 import ReCoNetMin
import utils
import time
from PIL import Image

batch_size = 2
fps = 24
frame_size = (640,360)
state_dict_path = "autoportrait/model_min_15000.pth"
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_device = 0
def style_batch(b):
    b = np.array(b)
    for styled_frame in style(b):
        # cv2.imwrite("output/img{}.jpg".format(iteration),styled_frame)
        writer.write(styled_frame)

def resize(frame):
	scale_percent = 50 # percent of original size
	width = int(frame.shape[1] * scale_percent / 100)
	height = int(frame.shape[0] * scale_percent / 100)
	dim = (width, height)
	# resize image
	resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
	return resized

def style(imgs):
    assert imgs.dtype == np.uint8
    assert 3 <= imgs.ndim <= 4

    orig_ndim = imgs.ndim
    if imgs.ndim == 3:
            imgs = imgs[None, ...]
        
    imgs = torch.from_numpy(imgs)
    imgs = utils.nhwc_to_nchw(imgs)
    imgs = imgs.to(torch.float32) / 255

    with device():
        with torch.no_grad():
            imgs = to_device(imgs)
            imgs = utils.preprocess_for_reconet(imgs)
            styled_images = model(imgs)
            styled_images = utils.postprocess_reconet(styled_images)
            styled_images = styled_images.cpu()
            styled_images = torch.clamp(styled_images * 255, 0, 255).to(torch.uint8)
            styled_images = utils.nchw_to_nhwc(styled_images)
            styled_images = styled_images.numpy()
            if orig_ndim == 3:
                styled_images = styled_images[0]
            return styled_images

def to_device(x):
    if torch.cuda.is_available():
        with device():
            return x.cuda()
    else:
        return x

def device():
    if torch.cuda.is_available() and gpu_device is not None:
        return torch.cuda.device(gpu_device)

print(time.ctime())
model = ReCoNetMin()
model.load_state_dict(torch.load(state_dict_path))
model = to_device(model)
model.eval()
print("Model loaded!")

cap = cv2.VideoCapture('input/chinka1.mp4')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')

writer = cv2.VideoWriter('output/autoportrait_output.avi', fourcc, 30.0, frame_size)
iteration = 0
batch = []
while(cap.isOpened()):

    ret, frame = cap.read()
    frame = resize(frame)
    # cv2.imwrite('/content/frame.png',frame)
    # break
    if ret==True:
        batch.append(frame)
        if len(batch) == batch_size:
            style_batch(batch)
            iteration +=len(batch)
            if iteration  % 1000 == 0:
                print("{} frames".format(iteration ))
            if iteration % 2000 == 0:
                print(time.ctime())
            batch = []
        # if cv2.waitKey(1) & 0xFF == ord('q'):
    # else:
    #     break


if len(batch) != 0:
    style_batch(batch)

# Release everything if job is finished
cap.release()
#cv2.destroyAllWindows()
print("Done")
print(time.ctime())