import os, glob
from pathlib import Path
import cv2
import mmcv
import argparse
from mmdet.apis import inference_detector, init_detector

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / 'data'
CONFIG = './configs/detr/detr_r50_8x2_150e_coco.py'
CKT = './checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'

VIDEO_FILE = str(DATA_DIR / 'goats1.mp4')

MODEL_NAME = Path(CONFIG).stem

print(MODEL_NAME)

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('--video', default=VIDEO_FILE, help='Video file')
    parser.add_argument('--config', default=CONFIG, help='Config file')
    parser.add_argument('--checkpoint', default=CKT, help='Checkpoint file')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    args.show = True
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')
    main_video(args)

def main_video(args):
    out_dir = ROOT_DIR / 'output' / MODEL_NAME
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    args.out = out_dir / Path(args.video).name
    args.out = None
    detect_1_video(args)
def detect_1_video(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)

    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    i = -1
    for frame in mmcv.track_iter_progress(video_reader):
        i += 1
        if i%5!=0:
            continue
        result = inference_detector(model, frame)
        frame, nums = model.show_result(frame, result, score_thr=args.score_thr)
        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', args.wait_time)
        if args.out:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

def main_imgs():
    data_dir = ROOT_DIR / 'data'
    out_dir = ROOT_DIR / 'output' / MODEL_NAME
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    img_files = []
    for fmt in ['jpg', 'jpeg', 'png']:
        img_files += glob.glob(str(data_dir/f'*.{fmt}'))
    print(img_files)

    for img_f in img_files:
        out_img = out_dir / Path(img_f).name
        cmd = f'python demo/image_demo.py {img_f} {CONFIG} {CKT} --out-file {out_img} --device cpu'
        print(img_f)
        print(out_img)
        #os.system(cmd)
    pass

if __name__ == '__main__':
    main()