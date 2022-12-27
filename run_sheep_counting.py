import os, glob
from pathlib import Path

ROOT_DIR = Path(__file__).parent
CONFIG = './configs/detr/detr_r50_8x2_150e_coco.py'
CKT = './checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'

MODEL_NAME = Path(CONFIG).stem

print(MODEL_NAME)

def main():

    main_imgs()
    
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
        os.system(cmd)
        break
    pass

if __name__ == '__main__':
    main()