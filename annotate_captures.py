"""Crop the CARLA window chrome from the captured frames and add a legible state banner (state parsed from the file name)."""
import glob
import os
import sys

from PIL import Image, ImageDraw, ImageFont

src = sys.argv[1] if len(sys.argv) > 1 else r"F:\carla_cl_ids\captures"
dst = sys.argv[2] if len(sys.argv) > 2 else r"F:\carla_cl_ids\captures_final"
os.makedirs(dst, exist_ok=True)
font = ImageFont.truetype(r"C:\Windows\Fonts\malgunbd.ttf", 26)
small = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 18)

for p in sorted(glob.glob(os.path.join(src, "*.png"))):
    base = os.path.splitext(os.path.basename(p))[0]
    idx, state = base.split("_", 1)
    can = state.split("_SENSOR-")[0].replace("CAN-", "").replace("_", " ")
    sen = state.split("_SENSOR-")[1].replace("_", " ")
    im = Image.open(p).convert("RGB")
    w, h = im.size
    im = im.crop((8, 32, w, h))
    attack = can != "NORMAL" or sen != "NORMAL"
    bar_h = 78
    out = Image.new("RGB", (im.width, im.height + bar_h), (20, 20, 20))
    out.paste(im, (0, bar_h))
    d = ImageDraw.Draw(out)
    col = (255, 90, 90) if attack else (90, 220, 120)
    d.text((14, 8), f"CAN bus:  {can}", font=font, fill=(255, 120, 120) if can != "NORMAL" else (90, 220, 120))
    d.text((14, 42), f"Sensors:  {sen}", font=font, fill=(255, 120, 120) if sen != "NORMAL" else (90, 220, 120))
    d.text((im.width - 330, 10), f"CARLA 0.9.16  |  Town10HD  |  frame {int(idx) + 1}", font=small, fill=(210, 210, 210))
    out.save(os.path.join(dst, base + ".png"))
    print("wrote", base, out.size)
