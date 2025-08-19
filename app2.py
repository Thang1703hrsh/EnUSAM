# app.py  (no pandas) + legend + Dice/IoU CSV
import io, os, tempfile
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import streamlit as st
from PIL import Image, ImageFile
import segmentation_models_pytorch as smp

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_grad_enabled(False)

# ================= UI =================
st.set_page_config(page_title="En-U-SAMNet", layout="wide")
st.title("En-U-SAMNet: MÔ HÌNH HỌC SÂU KẾT HỢP CHO VIỆC PHÁT HIỆN MỘT SỐ CƠ QUAN ĐƯỜNG TIÊU HOÁ")

with st.sidebar:
    st.header("Model / Inference")
    num_classes = 3
    st.divider()
    thr          = st.slider("Ngưỡng sigmoid", 0.0, 1.0, 0.50, 0.01)
    alpha        = st.slider("Opacity overlay", 0.0, 1.0, 0.45, 0.05)

c1, c2, c3 = st.columns(3)
with c1:
    ckpt_path = st.text_input("Checkpoint path", value="Weight/SAM.bin")
with c2:
    img_path  = st.text_input("Image path", value="Image/case101_day20_slice_0085.npy")
with c3:
    gt_path   = st.text_input("GT mask path (.npy, optional)", value="mask/case101_day20_slice_0085.npy")

ckpt_upload = st.file_uploader("upload checkpoint", type=["bin","pt","pth"])
img_upload  = st.file_uploader("upload ảnh", type=["npy","jpg","jpeg","png"])
gt_upload   = st.file_uploader("Upload GT mask (tuỳ chọn, .npy)", type=["npy"])

# ================= Helpers =================
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def save_upload_to_tmp(upload, suffix):
    path = os.path.join(tempfile.gettempdir(), f"up_{next(tempfile._get_candidate_names())}{suffix}")
    with open(path, "wb") as f: f.write(upload.getvalue())
    return path

def _to_hwc(arr: np.ndarray) -> np.ndarray:
    a = np.squeeze(arr)
    if a.ndim == 2:
        a = np.stack([a]*3, axis=-1)
    elif a.ndim == 3 and a.shape[0] == 3 and a.shape[2] != 3:
        a = np.transpose(a, (1,2,0))
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    return a

def load_image_for_display(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        a = _to_hwc(np.load(path, allow_pickle=False))
        if a.dtype != np.uint8:
            a = a.astype(np.float32)
            if float(np.nanmax(a)) <= 1.5: a *= 255.0
            a = np.clip(a, 0, 255).astype(np.uint8)
        return a
    else:
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def load_image_01(path: str) -> np.ndarray:
    """HWC float32 in [0..1] cho tensor input."""
    if path.lower().endswith(".npy"):
        a = _to_hwc(np.load(path, allow_pickle=False)).astype(np.float32)
    else:
        a = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    mx = float(np.nanmax(a)) if a.size else 0.0
    if mx > 1.5: a = a / 255.0
    return np.clip(a, 0.0, 1.0)

def img01_to_tensor(img01: np.ndarray, imagenet_norm: bool) -> torch.Tensor:
    x = torch.from_numpy(img01).permute(2,0,1).float().unsqueeze(0)  # [1,3,H,W], 0..1
    if imagenet_norm:
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x

def pad_to_multiple(t: torch.Tensor, mult: int):
    _, _, h, w = t.shape
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph == 0 and pw == 0: return t, (0,0)
    return F.pad(t, (0, pw, 0, ph), mode="reflect"), (ph, pw)

def unpad(t: torch.Tensor, pads):
    ph, pw = pads
    if ph == 0 and pw == 0: return t
    return t[..., : t.shape[-2] - ph, : t.shape[-1] - pw]

def chw_bin_to_hwc3(bin_chw: np.ndarray) -> np.ndarray:
    C,H,W = bin_chw.shape
    if C >= 3:
        return (np.transpose(bin_chw[:3], (1,2,0))*255).astype(np.uint8)  # R, G, B
    elif C == 1:
        g = (bin_chw[0]*255).astype(np.uint8)
        return np.stack([g,g,g], axis=-1)
    else:
        reps = (3 + C - 1)//C
        tiled = np.tile(bin_chw, (reps,1,1))[:3]
        return (np.transpose(tiled,(1,2,0))*255).astype(np.uint8)

def dice_iou(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-7):
    """pred_bin, gt_bin: [H,W] 0/1"""
    p = pred_bin.astype(bool); g = gt_bin.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    dice = (2*inter) / (p.sum() + g.sum() + eps)
    iou  = inter / (union + eps)
    return float(dice), float(iou), int(p.sum()), int(g.sum())

# ================= Dataset & Dataloader (no pandas) =================
class SimpleImageDataset(Dataset):
    def __init__(self, paths, imagenet_norm=True):
        self.paths = list(paths)
        self.imagenet_norm = imagenet_norm

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img01 = load_image_01(p)                                   # HWC, 0..1
        x = img01_to_tensor(img01, self.imagenet_norm).squeeze(0)  # [3,H,W]
        vis = (img01*255.0).astype(np.uint8)                       # HWC uint8 để show
        return x, vis, p

# ================= Model =================
def build_model():
    model = smp.Unet(
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
    model.to(device).eval()
    return model

def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def render_metrics_table_md(rows):
    """Render list[dict] thành bảng Markdown, tránh phụ thuộc pyarrow/pandas."""
    headers = ["class", "color_rgb", "dice", "iou", "pred_px>0", "gt_px>0"]
    md = "| " + " | ".join(headers) + " |\n"
    md += "|" + " | ".join(["---"]*len(headers)) + "|\n"
    for r in rows:
        md += f"| {r['class']} | {r['color_rgb']} | {r['dice']} | {r['iou']} | {r['pred_px>0']} | {r['gt_px>0']} |\n"
    st.markdown(md)

# ================= Resolve inputs =================
device = "cuda" if (torch.cuda.is_available()) else "cpu"

# checkpoint
ckpt_bytes = None
if ckpt_upload is not None:
    ckpt_bytes = ckpt_upload.getvalue()
elif ckpt_path and os.path.exists(ckpt_path):
    with open(ckpt_path, "rb") as f: ckpt_bytes = f.read()

# image
if img_upload is not None:
    img_path = save_upload_to_tmp(img_upload, os.path.splitext(img_upload.name)[1])

# gt (optional)
if gt_upload is not None:
    gt_path = save_upload_to_tmp(gt_upload, ".npy")

if not ckpt_bytes:
    st.info("Hãy upload SAM.bin hoặc điền đường dẫn hợp lệ.")
elif not img_path or not os.path.exists(img_path):
    st.info("Hãy upload ảnh test (.npy/.jpg/.png) hoặc điền đường dẫn hợp lệ.")
else:
    with st.spinner("Khởi tạo & nạp checkpoint…"):
        model = load_model(ckpt_path)

    #  DataLoader như test loop (không pandas)
    ds = SimpleImageDataset([img_path], imagenet_norm=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    xs, visses, paths = next(iter(loader))
    vis0 = visses[0] if isinstance(visses, list) else np.array(visses[0])
    xs = xs.to(device, dtype=torch.float)

    xs, pads = pad_to_multiple(xs, 32)
    with torch.no_grad():
        logits = model(xs)                       # [B,C,H',W']
        # logits = unpad(logits, pads)           # (khuyến nghị bật lại nếu muốn khớp biên gốc)
        probs  = torch.sigmoid(logits)           # [B,C,H,W]
        bins   = (probs > float(thr)).float()    # [B,C,H,W]

    probs_1 = probs[0].detach().cpu().numpy()    # [C,H,W]
    bin_1   = bins[0].detach().cpu().numpy()     # [C,H,W]
    Ht, Wt  = vis0.shape[:2]

    # đảm bảo H,W trùng với ảnh hiển thị
    if bin_1.shape[1:] != (Ht, Wt):
        t = torch.from_numpy(bin_1)[None].float()
        t = F.interpolate(t, size=(Ht, Wt), mode="nearest")
        bin_1 = t[0].numpy()

    pred_hwc = chw_bin_to_hwc3(bin_1)

    overlay = (
        vis0.astype(np.float32) * (1 - float(alpha)) +
        pred_hwc.astype(np.float32) * float(alpha)
    ).clip(0,255).astype(np.uint8)

    # ======= Hiển thị 1 hàng: Input | Pred | Overlay =======
    st.subheader("Prediction")
    cA, cB, cC = st.columns(3)
    with cA: st.image(vis0,    caption=f"Input ({Wt}x{Ht})", use_column_width=True)
    with cB: st.image(pred_hwc,caption="Pred mask (HWC)",   use_column_width=True)
    with cC: st.image(overlay, caption="Overlay",           use_column_width=True)

    # ======= (Tuỳ chọn) hiển thị & ĐÁNH GIÁ GT =======
    if gt_path and os.path.exists(gt_path):
        try:
            g = np.load(gt_path, allow_pickle=False)
            g = np.squeeze(g)
            if g.ndim == 2:
                g2 = (g > 0.5).astype(np.uint8)
                gt_chw = g2[None, ...]
                g_rgb  = np.stack([g2*255]*3, axis=-1)
            elif g.ndim == 3 and g.shape[0] in (1,3,int(num_classes)) and g.shape[0] != g.shape[-1]:
                g = np.transpose(g, (1,2,0))  # HWC
                g_rgb = (g > 0.5).astype(np.uint8)*255
                gt_chw = np.transpose((g>0.5).astype(np.uint8), (2,0,1))
            else:
                g_rgb = (g>0.5).astype(np.uint8)*255
                gt_chw = np.transpose((g>0.5).astype(np.uint8), (2,0,1))

            if g_rgb.shape[:2] != (Ht,Wt):
                g_rgb = np.array(Image.fromarray(g_rgb).resize((Wt,Ht), Image.NEAREST))
                gt_chw = torch.from_numpy(gt_chw[None].astype(np.float32))
                gt_chw = F.interpolate(gt_chw, size=(Ht,Wt), mode="nearest")[0].numpy().astype(np.uint8)

            g_overlay = (
                vis0.astype(np.float32) * (1 - float(alpha)) +
                g_rgb.astype(np.float32) * float(alpha)
            ).clip(0,255).astype(np.uint8)

            st.subheader("Ground Truth")
            g0, g1, g2 = st.columns(3)
            with g0: st.image(vis0,    caption=f"Input ({Wt}x{Ht})", use_column_width=True)
            with g1: st.image(g_rgb,   caption="GT mask (HWC 0/255)", use_column_width=True)
            with g2: st.image(g_overlay, caption="GT overlay", use_column_width=True)

            # -------- Legend (kênh → màu) ----------
            st.markdown("**Chú giải màu (Legend)** — mỗi *kênh* → một màu RGB:")
            legend_cols = st.columns(num_classes)
            palette = [(255,0,0), (0,255,0), (0,0,255)]  # ch0: đỏ, ch1: xanh lá, ch2: xanh dương
            class_names = [f"Class {i}" for i in range(num_classes)]
            bophan_names = ["ruột già" , "ruột non", "dạ dày"]
            for i in range(num_classes):
                chip = np.full((36, 36, 3), palette[i], dtype=np.uint8)
                legend_cols[i].image(chip, caption=f"{class_names[i]} — {bophan_names[i]}", use_column_width=False)
                
            # ---------- Dice/IoU table ----------
            C_pred = bin_1.shape[0]
            C_gt   = gt_chw.shape[0]
            K = min(C_pred, C_gt)

            rows = []
            dice_sum = 0.0
            iou_sum  = 0.0
            for c in range(K):
                d, i, pp, gp = dice_iou(bin_1[c], gt_chw[c])
                dice_sum += d; iou_sum += i
                rows.append({
                    "class": c,
                    "color_rgb": f"{bophan_names[c] if c < len(bophan_names) else 'N/A'}",
                    "dice": f"{d:.4f}",
                    "iou": f"{i:.4f}",
                    "pred_px>0": pp,
                    "gt_px>0": gp
                })

            st.subheader("Dice / IoU theo lớp")
            render_metrics_table_md(rows)  # << dùng Markdown, không còn pyarrow

            # ---------- Download CSV ----------
            csv_lines = ["class,color_rgb,dice,iou,pred_px_pos,gt_px_pos"]
            for r in rows:
                csv_lines.append(f"{r['class']},{r['color_rgb']},{r['dice']},{r['iou']},{r['pred_px>0']},{r['gt_px>0']}")
            csv_bytes = ("\n".join(csv_lines)).encode("utf-8")
            

        except Exception as e:
            st.warning(f"Không đọc được GT: {e}")