"""Banana Ripeness Classifier - demo app (Gradio).

Chạy:  py app.py     (đặt file này cạnh thư mục models/ do notebook tạo ra)
Mở:    địa chỉ in ra ở terminal (thường http://127.0.0.1:7860) bằng Chrome/Edge
"""
import os
import inspect
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image
from tensorflow import keras

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = HERE if os.path.exists(os.path.join(HERE, "mobilenetv3.keras")) else os.path.join(HERE, "models")
IMG_SIZE = (224, 224)
CLASSES = ["unripe", "ripe", "overripe"]

INFO = {
    "unripe": {"emoji": "🟢", "title": "Chuối xanh", "en": "Unripe", "color": "#4caf50",
               "advice": "Chuối còn xanh hoặc mới ương. Nên để thêm vài ngày ở nhiệt độ phòng cho chín."},
    "ripe": {"emoji": "🟡", "title": "Chuối chín", "en": "Ripe", "color": "#f2b705",
             "advice": "Chuối chín tới, thời điểm ngon nhất để ăn trực tiếp."},
    "overripe": {"emoji": "🟤", "title": "Chuối quá chín", "en": "Overripe", "color": "#8d5a2b",
                 "advice": "Chuối đã rất chín, nên dùng sớm. Hợp để làm sinh tố, bánh chuối hoặc chuối nghiền."},
}


def pick_model():
    """Chọn model có Macro-F1 cao hơn nếu có results.csv, mặc định MobileNetV3."""
    files = {"CNN Baseline": "cnn_baseline.keras", "MobileNetV3": "mobilenetv3.keras"}
    name, stats = "MobileNetV3", None
    csv = os.path.join(MODEL_DIR, "results.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv).sort_values("Macro-F1", ascending=False)
        row = df.iloc[0]
        name = row["Model"]
        stats = (float(row["Accuracy"]), float(row["Macro-F1"]))
    path = os.path.join(MODEL_DIR, files.get(name, files["MobileNetV3"]))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không thấy model: {path}. Hãy chạy notebook để train và lưu model trước.")
    return name, path, stats


MODEL_NAME, MODEL_PATH, STATS = pick_model()
model = keras.models.load_model(MODEL_PATH)
print(f"Đã nạp model: {MODEL_NAME}")

EMPTY = """
<div class='panel empty'>
  <div class='empty-emoji'>🍌</div>
  <div class='empty-title'>Chưa có ảnh</div>
  <div class='empty-sub'>Tải ảnh lên hoặc chụp bằng camera, kết quả sẽ hiện ở đây.</div>
</div>"""


def render(p):
    top = CLASSES[int(np.argmax(p))]
    t = INFO[top]
    # Vị trí trên thang độ chín: 0 = xanh, 0.5 = chín, 1 = quá chín
    pos = float(p[1] * 0.5 + p[2] * 1.0)
    pos = min(max(pos, 0.03), 0.97) * 100
    bars = ""
    for i, c in enumerate(CLASSES):
        inf = INFO[c]
        pct = p[i] * 100
        bars += f"""
        <div class='row'>
          <div class='row-label'>{inf['emoji']} {inf['title']}</div>
          <div class='track'><div class='fill' style='width:{pct:.1f}%;background:{inf['color']}'></div></div>
          <div class='row-val'>{pct:.1f}%</div>
        </div>"""
    return f"""
    <div class='panel result' style='border-top:6px solid {t['color']}'>
      <div class='badge' style='background:{t['color']}'>{t['en']}</div>
      <div class='big'>{t['emoji']} {t['title']}</div>
      <div class='conf'>Độ tin cậy <b>{p.max() * 100:.1f}%</b></div>
      <p class='advice'>{t['advice']}</p>
      <div class='sect'>Xác suất từng lớp</div>
      {bars}
      <div class='sect'>Thang độ chín</div>
      <div class='gauge'><div class='marker' style='left:{pos:.1f}%'></div></div>
      <div class='gauge-labels'><span>Xanh</span><span>Chín</span><span>Quá chín</span></div>
    </div>"""


def predict(img):
    if img is None:
        return EMPTY
    im = Image.fromarray(img).convert("RGB").resize(IMG_SIZE)
    x = np.asarray(im, dtype="float32")[None]          # thang 0-255
    p = model.predict(x, verbose=0)[0]
    return render(p)


def reset():
    return None, EMPTY


CSS = """
.gradio-container {max-width: 1080px !important; margin: auto;}
footer {display: none !important;}

#hero {
  background: linear-gradient(135deg, #ffd54a 0%, #ff9f43 55%, #ee6c4d 100%);
  border-radius: 20px; padding: 30px 28px; margin: 6px 0 18px; color: #3b2a05;
  box-shadow: 0 10px 30px rgba(238,108,77,.25);
}
#hero h1 {font-size: 2.3rem; margin: 0 0 6px; font-weight: 800; letter-spacing: -.5px;}
#hero p {margin: 0 0 14px; font-size: 1.05rem; opacity: .85;}
.chips {display: flex; flex-wrap: wrap; gap: 8px;}
.chip {background: rgba(255,255,255,.55); border-radius: 999px; padding: 5px 13px;
       font-size: .85rem; font-weight: 600; backdrop-filter: blur(4px);}

.panel {border: 1px solid var(--border-color-primary); border-radius: 18px; padding: 22px 24px;
        background: var(--background-fill-secondary);}
.empty {text-align: center; padding: 60px 20px;}
.empty-emoji {font-size: 3.6rem; margin-bottom: 8px; opacity: .8;}
.empty-title {font-size: 1.2rem; font-weight: 700;}
.empty-sub {opacity: .65; margin-top: 4px;}

.result {animation: pop .35s ease;}
@keyframes pop {from {opacity: 0; transform: translateY(8px);} to {opacity: 1; transform: none;}}
.badge {display: inline-block; color: #fff; font-weight: 700; font-size: .78rem; letter-spacing: .5px;
        text-transform: uppercase; padding: 3px 12px; border-radius: 999px; margin-bottom: 8px;}
.big {font-size: 1.9rem; font-weight: 800; line-height: 1.2;}
.conf {opacity: .75; margin: 2px 0 10px;}
.advice {margin: 0 0 6px; line-height: 1.55;}
.sect {font-size: .78rem; font-weight: 700; letter-spacing: .6px; text-transform: uppercase;
       opacity: .55; margin: 18px 0 8px;}

.row {display: flex; align-items: center; gap: 10px; margin: 7px 0;}
.row-label {width: 150px; font-size: .92rem; white-space: nowrap;}
.track {flex: 1; height: 12px; border-radius: 99px; background: var(--border-color-primary); overflow: hidden;}
.fill {height: 100%; border-radius: 99px; transition: width .6s ease;}
.row-val {width: 56px; text-align: right; font-variant-numeric: tabular-nums; font-weight: 600;}

.gauge {position: relative; height: 14px; border-radius: 99px;
        background: linear-gradient(90deg, #4caf50, #c6d94a, #f2b705, #c98a2b, #6d4322);}
.marker {position: absolute; top: -5px; width: 8px; height: 24px; margin-left: -4px; border-radius: 4px;
         background: #fff; border: 2px solid #333; box-shadow: 0 2px 6px rgba(0,0,0,.35);
         transition: left .6s ease;}
.gauge-labels {display: flex; justify-content: space-between; font-size: .8rem; opacity: .65; margin-top: 6px;}

.tips {opacity: .8; line-height: 1.6;}
#foot {text-align: center; opacity: .55; font-size: .85rem; margin-top: 14px;}
"""

theme = gr.themes.Soft(
    primary_hue="amber", secondary_hue="orange", neutral_hue="stone",
    font=[gr.themes.GoogleFont("Nunito"), "ui-sans-serif", "system-ui", "sans-serif"],
    radius_size="lg",
)

# Gradio 5 nhận theme/css ở Blocks(), Gradio 6 nhận ở launch(): tự nhận biết.
style_in_launch = "theme" in inspect.signature(gr.Blocks.launch).parameters
blocks_kw = {} if style_in_launch else {"theme": theme, "css": CSS}
launch_kw = {"theme": theme, "css": CSS} if style_in_launch else {}

chips = f"<span class='chip'>🧠 {MODEL_NAME}</span><span class='chip'>🖼️ Ảnh {IMG_SIZE[0]}×{IMG_SIZE[1]}</span><span class='chip'>🏷️ 3 lớp</span>"
if STATS:
    chips += f"<span class='chip'>🎯 Accuracy {STATS[0] * 100:.1f}%</span><span class='chip'>📊 Macro-F1 {STATS[1] * 100:.1f}%</span>"

with gr.Blocks(title="Banana Ripeness Classifier", **blocks_kw) as demo:
    gr.HTML(
        "<div id='hero'><h1>🍌 Banana Ripeness Classifier</h1>"
        "<p>Nhận diện độ chín của chuối bằng Deep Learning: xanh, chín hay quá chín.</p>"
        f"<div class='chips'>{chips}</div></div>"
    )
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            inp = gr.Image(type="numpy", sources=["upload", "webcam", "clipboard"],
                           label="Ảnh chuối", height=360)
            with gr.Row():
                btn = gr.Button("🔍 Phân loại", variant="primary", scale=3)
                clear = gr.Button("Xóa", scale=1)
            with gr.Accordion("💡 Mẹo để kết quả chính xác hơn", open=False):
                gr.HTML("<div class='tips'>• Chụp rõ một quả hoặc một nải, đủ sáng.<br>"
                        "• Nền đơn giản, tránh nhiều vật khác trong khung hình.<br>"
                        "• Tránh đèn màu hoặc filter làm lệch màu vỏ chuối.</div>")
        with gr.Column(scale=6):
            out = gr.HTML(EMPTY)
    btn.click(predict, inp, out)
    inp.change(predict, inp, out)
    clear.click(reset, None, [inp, out])
    gr.HTML("<div id='foot'>Nhóm BaNaNaNoNoN · 22PFIEV2 · Dữ liệu: BananaImageBD</div>")

if __name__ == "__main__":
    demo.launch(**launch_kw)
