# 왜지 v2 렌더러 — 프레임 단위 합성 (켄번즈 줌, 컷 플래시, 자막 박스 팝인, 스티커 바운스, 진행바, 비네팅)
import sys, os, json, math, subprocess, shutil
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
FB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc'
FBo = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
def font(p, s): return ImageFont.truetype(p, s, index=1)
YEL = (255, 214, 64); RED = (235, 55, 60); BLUE = (45, 125, 255); ORG = (255, 145, 40); GRN = (40, 175, 95)
TOOLS = os.environ.get('WAEJI_TOOLS', os.path.dirname(os.path.abspath(__file__)))
LOGO = os.path.join(TOOLS, 'logo.png'); BGM = os.path.join(TOOLS, 'bgm.wav')

def parse(line):
    out, hi, buf = [], False, ''
    for ch in line:
        if ch == '[': out.append((buf, hi)); buf = ''; hi = True
        elif ch == ']': out.append((buf, hi)); buf = ''; hi = False
        else: buf += ch
    out.append((buf, hi)); return [(s, h) for s, h in out if s]

# ---------- 그래픽 에셋 ----------
def make_caption(ko, en):
    fk, fe = font(FB, 64), font(FBo, 32)
    tmp = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    lines = ko.split('\n'); lh = 64 * 1.3
    wmax = max(sum(tmp.textlength(s, font=fk) for s, _ in parse(l)) for l in lines)
    ew = tmp.textlength(en.replace('[', '').replace(']', ''), font=fe)
    bw = int(max(wmax, ew) + 90); bh = int(len(lines) * lh + 32 + 60)
    L = Image.new('RGBA', (bw + 20, bh + 20), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    d.rounded_rectangle([10, 10, bw + 10, bh + 10], radius=34, fill=(12, 12, 18, 175))
    d.rounded_rectangle([10, 10, bw + 10, bh + 10], radius=34, outline=(255, 214, 64, 200), width=4)
    y = 10 + 22
    for l in lines:
        segs = parse(l); tw = sum(d.textlength(s, font=fk) for s, _ in segs); x = 10 + (bw - tw) / 2
        for s, h in segs:
            d.text((x, y), s, font=fk, fill=YEL if h else (255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0)); x += d.textlength(s, font=fk)
        y += lh
    segs = parse(en); tw = sum(d.textlength(s, font=fe) for s, _ in segs); x = 10 + (bw - tw) / 2
    for s, h in segs:
        d.text((x, y + 6), s, font=fe, fill=YEL if h else (205, 212, 225)); x += d.textlength(s, font=fe)
    return L

def make_sticker(text, bg, fg=None, size=None, rot=None, pad=None):
    # 3안: 어두운 반투명 알약 + 컬러 점·테두리 + 흰 글씨 (기울기·외곽선 없음)
    from PIL import ImageFilter
    col = fg if fg is not None else bg
    f = font(FBo, 80); tw = ImageDraw.Draw(Image.new('RGB', (1, 1))).textlength(text, font=f)
    w, h = int(tw + 140), 128; C = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(C)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=h // 2, fill=(15, 15, 20, 200), outline=col, width=4)
    d.ellipse([38, h // 2 - 14, 66, h // 2 + 14], fill=col)
    d.text((90, (h - 80) // 2 - 13), text, font=f, fill=(255, 255, 255))
    r, off = 18, 8; L = Image.new('RGBA', (w + 2 * r, h + 2 * r + off), (0, 0, 0, 0))
    sh = Image.new('RGBA', (w, h), (0, 0, 0, 0)); sh.putalpha(C.split()[3].point(lambda v: 80 if v > 0 else 0))
    L.alpha_composite(sh, (r, r + off)); L = L.filter(ImageFilter.GaussianBlur(r / 2)); L.alpha_composite(C, (r, r))
    return L

def make_banner(title):
    B = Image.new('RGBA', (W, 260), (0, 0, 0, 0)); a = np.zeros((260, W), np.uint8)
    for yy in range(260): a[yy, :] = int(230 * max(0, 1 - (yy / 260) ** 1.6))
    B.putalpha(Image.fromarray(a)); d = ImageDraw.Draw(B)
    logo = Image.open(LOGO).convert('RGBA'); logo.thumbnail((118, 118)); B.alpha_composite(logo, (34, 42))
    sz = 64
    while d.textlength(title, font=font(FB, sz)) > W - 230 and sz > 40: sz -= 2
    d.text((172, 101 - sz // 2 - 6), title, font=font(FB, sz), fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    tw = d.textlength(title, font=font(FB, sz))
    d.rounded_rectangle([172, 101 + sz // 2 + 18, 172 + min(tw, 260), 101 + sz // 2 + 26], radius=4, fill=YEL)
    return B

def rgba_np(img): return np.array(img.convert('RGBA'))

def blend(frame, ov, x, y, alpha=1.0):
    h, w = ov.shape[:2]; x, y = int(x), int(y)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1: return
    o = ov[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (o[:, :, 3:4].astype(np.float32) / 255.0) * alpha
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (roi * (1 - a) + o[:, :, 2::-1].astype(np.float32) * a).astype(np.uint8)  # RGBA→BGR

def scaled(ov, s):
    if abs(s - 1) < 1e-3: return ov
    h, w = ov.shape[:2]; return cv2.resize(ov, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_LINEAR)

def ease_out(p): return 1 - (1 - p) ** 3
def pop(t, dur=0.16):   # 자막 팝인: 0.88→1.0
    p = min(1, max(0, t / dur)); return 0.88 + 0.12 * ease_out(p), ease_out(p)
def bounce(t):          # 스티커: 0→1.18→1.0
    if t < 0.13: p = t / 0.13; return 1.18 * ease_out(p)
    if t < 0.28: p = (t - 0.13) / 0.15; return 1.18 - 0.18 * ease_out(p)
    return 1.0

# ---------- 메인 ----------
def build(cfg):
    D = cfg['dir']; os.chdir(D)
    dur = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', 'vo.mp3']))
    T = round(dur + 3.0, 2); NF = int(T * FPS)
    # 컷 리스트
    OFF = [0, 2.0, 3.8, 1.0]; S = []; k = 0
    for a, b, c in cfg['R']:
        b = T if b is None else b; n = max(1, round((b - a) / 2.0)); L = (b - a) / n
        for j in range(n):
            s = OFF[k % 4] if L <= 2.2 else 0; S.append((c, min(s, 5.9 - L), L)); k += 1
    starts = np.cumsum([0] + [d for _, _, d in S])
    # 에셋
    banner = rgba_np(make_banner(cfg['banner']))
    caps = [rgba_np(make_caption(ko, en)) for ko, en in cfg['C']]
    CT = cfg['CT'] + [T]
    sts = [(rgba_np(make_sticker(t, c, **kw)), s, e) for t, c, kw, s, e in cfg['ST']]
    logo = rgba_np(Image.open(LOGO).convert('RGBA').resize((560, 560)))
    yy, xx = np.mgrid[0:H, 0:W]; r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vign = np.clip(1 - 0.32 * np.clip(r - 0.55, 0, 1) ** 1.5, 0, 1).astype(np.float32)[..., None]
    enc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-crf', '19', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', 'v2_video.mp4'], stdin=subprocess.PIPE)
    fi = 0
    for si, (c, s0, L) in enumerate(S):
        cap = cv2.VideoCapture(f'clips/cut{c}.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, int(s0 * FPS))
        n = int(round(starts[si + 1] * FPS)) - int(round(starts[si] * FPS)); last = None
        zin = (si % 2 == 0); px = (0.5 if si % 3 == 0 else (0.42 if si % 3 == 1 else 0.58))
        for j in range(n):
            ok, fr = cap.read()
            if ok: last = fr
            fr = last.copy()
            t = fi / FPS; p = j / max(1, n - 1)
            z = (1.03 + 0.07 * p) if zin else (1.10 - 0.07 * p)
            if j < 5 and si > 0: z += 0.05 * (1 - j / 5)                     # 컷 펀치
            cw, ch = W / z, H / z; cx = (W - cw) * px; cy = (H - ch) * 0.5
            fr = cv2.resize(fr[int(cy):int(cy + ch), int(cx):int(cx + cw)], (W, H), interpolation=cv2.INTER_LINEAR)
            fr = cv2.convertScaleAbs(fr, alpha=1.06, beta=-4)                # 살짝 대비
            fr = (fr.astype(np.float32) * vign).astype(np.uint8)
            if j < 3 and si > 0:                                             # 화이트 플래시
                a = 0.35 * (1 - j / 3); fr = (fr * (1 - a) + 255 * a).astype(np.uint8)
            endc = t >= T - 2.0
            if endc:
                q = min(1, (t - (T - 2.0)) / 0.3); fr = (fr * (1 - 0.55 * q)).astype(np.uint8)
            blend(fr, banner, 0, 0)
            cv2.rectangle(fr, (0, 0), (int(W * t / T), 9), (64, 214, 255), -1)  # 진행바(BGR 노랑)
            for i in range(len(caps)):
                if CT[i] <= t < CT[i + 1]:
                    sc, al = pop(t - CT[i]); ov = scaled(caps[i], sc); h, w = ov.shape[:2]
                    blend(fr, ov, (W - w) / 2, 1330 - h / 2, al)
            for ov0, s, e in sts:
                if s <= t < e:
                    sc = bounce(t - s); fade = min(1, (e - t) / 0.15)
                    ov = scaled(ov0, max(0.05, sc)); h, w = ov.shape[:2]
                    bob = 6 * math.sin((t - s) * 5)
                    blend(fr, ov, (W - w) / 2, 0.40 * H - h / 2 + bob, fade)
            if endc:
                sc = bounce(t - (T - 2.0)); ov = scaled(logo, max(0.05, sc)); h, w = ov.shape[:2]
                blend(fr, ov, (W - w) / 2, 0.40 * H - h / 2)
            enc.stdin.write(fr.tobytes()); fi += 1
        cap.release()
    enc.stdin.close(); enc.wait()
    out = cfg['out_mp4']
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', 'v2_video.mp4', '-i', 'vo.mp3', '-i', BGM, '-filter_complex',
                    f'[1:a]adelay=200|200,loudnorm=I=-15:TP=-1.5,aresample=44100,apad[vo];[2:a]aloop=loop=-1:size=2000000000,volume=-17dB,aresample=44100,afade=t=out:st={T-2}:d=2[bg];[vo][bg]amix=inputs=2:duration=longest:normalize=0[a]',
                    '-map', '0:v', '-map', '[a]', '-t', str(T), '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k', out], check=True)
    print('done', out, T, len(S), fi)

def load(p):
    c = json.load(open(p))
    c['ST'] = [[t, tuple(col), {k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()}, s, e] for t, col, kw, s, e in c['ST']]
    return c

if __name__ == '__main__':
    build(load(sys.argv[1]))
