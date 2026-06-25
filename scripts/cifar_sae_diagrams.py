"""Static HTML/SVG explainer diagrams for the SAE tutorial (rendered inline in Colab
via `from IPython.display import HTML, display; display(HTML(...))` -- no server).

Each function returns a self-contained HTML string. Korean labels render fine in a
browser (Noto CJK); these are NOT matplotlib so there is no glyph issue.
"""
from __future__ import annotations

FONT = "'Noto Sans CJK KR','Noto Sans KR','Noto Sans CJK HK','NanumGothic',sans-serif"
BLUE, BLUE2, ORANGE = "#4a78d0", "#86a8e3", "#e8841f"
EMPTY, EMPTYS, ARROW = "#eef0f2", "#cdd2d8", "#9aa3ad"
TEXT, GRAY = "#1c2733", "#5b6876"


def _col(x, y0, n, w, h, gap, fills):
    """Vertical stack of n rounded cells; fills = list (len n) of fill colors (or None=empty)."""
    out = []
    for i in range(n):
        y = y0 + i * (h + gap); f = fills[i]
        if f is None:
            out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="{EMPTY}" stroke="{EMPTYS}"/>')
        else:
            out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="{f}"/>')
    return "".join(out)


def sae_encode_decode_svg():
    W, H = 920, 470
    # left: dense activation (all on)
    left = _col(60, 100, 10, 46, 14, 3, [BLUE] * 10)
    # middle: sparse code (mostly empty, 4 lit) + concept labels
    lit = {2: ("f12", "바퀴"), 6: ("f731", "오토바이"), 9: ("f88", "차체"), 13: ("f203", "배경·하늘")}
    fills = [ORANGE if i in lit else None for i in range(16)]
    mid = _col(240, 92, 16, 40, 12, 3, fills)
    labels = []
    for i, (fid, name) in lit.items():
        y = 92 + i * 15 + 10
        labels.append(f'<text x="286" y="{y}" font-size="13" fill="{TEXT}"><tspan font-weight="700" fill="{ORANGE}">{fid}</tspan> · {name}</text>')
    # right: reconstruction (faded blue)
    right = _col(640, 100, 10, 46, 14, 3, [BLUE2] * 10)

    ann = [
        ("① 과완비(overcomplete)", "2048 ≫ 256 — 개념 사전을 활성값보다 크게 둡니다."),
        ("② 희소(sparse)", "batch-topk: 한 번에 단 16개만 ON → 각 feature가 단의미(monosemantic)."),
        ("③ feature = 개념 방향", "켜진 feature 하나 = W_dec의 한 열 = 하나의 개념 방향."),
        ("④ 학습 목표", "‖x − x_rec‖²(재구성 오차)↓ + 희소 강제 → 라벨 없이 개념이 저절로 분해됨."),
    ]
    arows = []
    for k, (h, d) in enumerate(ann):
        y = 372 + k * 24
        arows.append(f'<text x="50" y="{y}" font-size="13.5" fill="{TEXT}"><tspan font-weight="700">{h}</tspan>  <tspan fill="{GRAY}">{d}</tspan></text>')

    return f'''<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:880px;font-family:{FONT}">
<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
<path d="M0,0 L7,3 L0,6 Z" fill="{ARROW}"/></marker></defs>
<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>
<text x="{W//2}" y="34" text-anchor="middle" font-size="18" font-weight="700" fill="{TEXT}">SAE 한 개가 하는 일 — layer4 예시</text>
<text x="{W//2}" y="56" text-anchor="middle" font-size="13" fill="{GRAY}">레이어 활성값(256-d)을 2048개 feature로 분해 → 16개만 ON → 다시 256-d로 재구성</text>

<text x="83" y="90" text-anchor="middle" font-size="13.5" font-weight="700" fill="{TEXT}">활성값 x</text>
{left}
<text x="83" y="278" text-anchor="middle" font-size="12" fill="{GRAY}">256-d</text>
<text x="83" y="294" text-anchor="middle" font-size="12" fill="{GRAY}">전부 켜짐(뒤섞임)</text>

<line x1="116" y1="170" x2="226" y2="170" stroke="{ARROW}" stroke-width="2" marker-end="url(#ah)"/>
<text x="171" y="150" text-anchor="middle" font-size="12.5" font-weight="700" fill="{TEXT}">인코더</text>
<text x="171" y="165" text-anchor="middle" font-size="11.5" fill="{GRAY}">ReLU(W_enc·x), topk</text>

<text x="260" y="78" text-anchor="middle" font-size="13.5" font-weight="700" fill="{TEXT}">희소 코드 z</text>
{mid}
{''.join(labels)}
<text x="260" y="320" text-anchor="middle" font-size="12" fill="{GRAY}">2048-d · 16개만 ON</text>
<text x="260" y="336" text-anchor="middle" font-size="12" fill="{GRAY}">(각 ON = 개념 1개)</text>

<line x1="520" y1="170" x2="628" y2="170" stroke="{ARROW}" stroke-width="2" marker-end="url(#ah)"/>
<text x="573" y="150" text-anchor="middle" font-size="12.5" font-weight="700" fill="{TEXT}">디코더</text>
<text x="573" y="165" text-anchor="middle" font-size="11.5" fill="{GRAY}">W_dec·z</text>

<text x="663" y="90" text-anchor="middle" font-size="13.5" font-weight="700" fill="{TEXT}">재구성</text>
{right}
<text x="663" y="278" text-anchor="middle" font-size="12" fill="{GRAY}">≈ 원래 x (256-d)</text>

<rect x="36" y="350" width="848" height="104" rx="8" fill="#f5f7fa" stroke="#dde3ea"/>
{''.join(arows)}
</svg>'''


def sae_encode_decode_html():
    return f'<div style="font-family:{FONT}">{sae_encode_decode_svg()}</div>'


def _grid_cells(x0, y0, g, cell, gap, color, opac):
    """g×g grid; opac(r,c) in [0,1] -> fill opacity (None => solid)."""
    out = []
    for r in range(g):
        for c in range(g):
            x = x0 + c * (cell + gap); y = y0 + r * (cell + gap)
            o = 1.0 if opac is None else opac(r, c)
            out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="2" '
                       f'fill="{color}" fill-opacity="{o:.2f}" stroke="#fff" stroke-width="0.8"/>')
    return "".join(out)


def sae_spatial_svg(img_b64, amap, feat_id, concept):
    """Show that the SAE runs at EVERY spatial position of the [C·H·W] activation, so a
    feature has a SPATIAL activation map (where it fires). amap = HxW list of floats."""
    g = len(amap); mx = max(max(r) for r in amap) or 1.0
    norm = [[v / mx for v in row] for row in amap]
    W, H = 820, 372
    CELL, GAP = 21, 3; gw = g * (CELL + GAP) - GAP
    # activation grid (spatial, all positions active) with a depth shadow = 256 channels
    actx, acty = 182, 92
    depth = "".join(f'<rect x="{actx+d*4}" y="{acty-d*4}" width="{gw}" height="{gw}" rx="3" '
                    f'fill="#dbe4f2" stroke="#c3d0e6"/>' for d in (2, 1))
    actg = _grid_cells(actx, acty, g, CELL, GAP, BLUE, lambda r, c: 0.55 + 0.45 * ((r + c) % 2))
    # feature spatial map (orange by amap)
    mapx, mapy = 372, 92
    mapg = _grid_cells(mapx, mapy, g, CELL, GAP, ORANGE, lambda r, c: 0.1 + 0.9 * norm[r][c])
    # overlay = image + translucent orange cells scaled to the 86px image
    ovx, ovy, IMG = 560, 95, 86
    oc = IMG / g
    ov = "".join(f'<rect x="{ovx+c*oc:.1f}" y="{ovy+r*oc:.1f}" width="{oc:.1f}" height="{oc:.1f}" '
                 f'fill="{ORANGE}" fill-opacity="{0.78*norm[r][c]:.2f}"/>' for r in range(g) for c in range(g))
    ann = [
        "① conv 활성값은 공간적: [채널 256 × 위치 4×4]. 같은 SAE를 4×4 모든 위치에 똑같이 적용(위치마다 독립 인코딩).",
        "② 그래서 feature 하나는 ‘어느 위치에서 켜졌나’ = 공간 활성맵(4×4)을 가진다.",
        "③ 이 맵을 이미지에 겹치면 ‘이 feature가 이미지의 어디에 반응했나’가 보임 — 트리 우측 패널의 act-map이 바로 이것.",
    ]
    arows = "".join(f'<text x="44" y="{280 + k*24}" font-size="13" fill="{TEXT}">{a}</text>' for k, a in enumerate(ann))
    return f'''<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:800px;font-family:{FONT}">
<defs><marker id="ah2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="{ARROW}"/></marker></defs>
<rect width="{W}" height="{H}" fill="#fff"/>
<text x="{W//2}" y="30" text-anchor="middle" font-size="17.5" font-weight="700" fill="{TEXT}">CNN에서 SAE는 각 '위치'에 적용된다 — feature는 '공간 활성맵'을 가진다</text>
<text x="{W//2}" y="52" text-anchor="middle" font-size="12.5" fill="{GRAY}">conv 활성값 = 공간(4×4) × 채널(256). SAE를 각 위치의 256-d 벡터에 적용 → feature가 어느 위치에서 켜지는지가 활성맵.</text>

<image x="44" y="95" width="86" height="86" href="{img_b64}" style="image-rendering:pixelated"/>
<text x="87" y="196" text-anchor="middle" font-size="12" fill="{GRAY}">입력 이미지</text>
<line x1="136" y1="138" x2="172" y2="138" stroke="{ARROW}" stroke-width="2" marker-end="url(#ah2)"/>
<text x="154" y="128" text-anchor="middle" font-size="11" fill="{GRAY}">CNN</text>

{depth}{actg}
<text x="{actx+gw//2}" y="205" text-anchor="middle" font-size="12" fill="{GRAY}">layer4 활성값</text>
<text x="{actx+gw//2}" y="221" text-anchor="middle" font-size="11.5" fill="{GRAY}">4×4 위치 × 256채널</text>
<line x1="300" y1="138" x2="362" y2="138" stroke="{ARROW}" stroke-width="2" marker-end="url(#ah2)"/>
<text x="331" y="126" text-anchor="middle" font-size="11" fill="{TEXT}">각 위치에</text>
<text x="331" y="116" text-anchor="middle" font-size="11" fill="{TEXT}">SAE 적용</text>

{mapg}
<text x="{mapx+gw//2}" y="205" text-anchor="middle" font-size="12" fill="{ORANGE}" font-weight="700">f{feat_id} 활성맵</text>
<text x="{mapx+gw//2}" y="221" text-anchor="middle" font-size="11.5" fill="{GRAY}">({concept} feature)</text>
<line x1="478" y1="138" x2="540" y2="138" stroke="{ARROW}" stroke-width="2" marker-end="url(#ah2)"/>
<text x="509" y="128" text-anchor="middle" font-size="11" fill="{GRAY}">이미지에 겹침</text>

<image x="{ovx}" y="{ovy}" width="{IMG}" height="{IMG}" href="{img_b64}" style="image-rendering:pixelated"/>{ov}
<rect x="{ovx}" y="{ovy}" width="{IMG}" height="{IMG}" fill="none" stroke="#e8841f" stroke-width="1.5"/>
<text x="{ovx+IMG//2}" y="205" text-anchor="middle" font-size="12" fill="{GRAY}">어디서 ON 인지 보임</text>

<rect x="30" y="248" width="760" height="104" rx="8" fill="#f5f7fa" stroke="#dde3ea"/>
{arows}
</svg>'''


def sae_spatial_html(img_b64, amap, feat_id, concept):
    return f'<div style="font-family:{FONT}">{sae_spatial_svg(img_b64, amap, feat_id, concept)}</div>'


def _marker(mid):
    return (f'<defs><marker id="{mid}" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
            f'<path d="M0,0 L7,3 L0,6 Z" fill="{ARROW}"/></marker></defs>')


def roadmap_svg():
    """One-glance pipeline: CNN -> SAE per layer -> feature -> index -> tree -> debug."""
    stages = [("CNN 학습", "이미지→클래스(블랙박스)"), ("레이어마다 SAE", "활성값 해석기"),
              ("feature", "개념 단위로 분해"), ("index", "top 이미지 = 이름"),
              ("mechanistic tree", "개념→클래스 합성"), ("디버깅·shortcut", "무엇이 잘못됐나")]
    W, bw, bh, y = 980, 138, 64, 58
    gap = (W - 40 - len(stages) * bw) / (len(stages) - 1)
    out = []
    for i, (t, s) in enumerate(stages):
        x = 20 + i * (bw + gap)
        col = "#eaf1fb" if i % 2 == 0 else "#f2f4f7"
        out.append(f'<rect x="{x:.1f}" y="{y}" width="{bw}" height="{bh}" rx="9" fill="{col}" stroke="#cdd8ea"/>')
        out.append(f'<text x="{x+bw/2:.1f}" y="{y+26}" text-anchor="middle" font-size="13" font-weight="700" fill="{TEXT}">{t}</text>')
        out.append(f'<text x="{x+bw/2:.1f}" y="{y+45}" text-anchor="middle" font-size="10.5" fill="{GRAY}">{s}</text>')
        if i < len(stages) - 1:
            out.append(f'<line x1="{x+bw+2:.1f}" y1="{y+bh/2}" x2="{x+bw+gap-3:.1f}" y2="{y+bh/2}" stroke="{ARROW}" stroke-width="2" marker-end="url(#ahR)"/>')
    return f'''<svg viewBox="0 0 {W} 150" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:940px;font-family:{FONT}">
{_marker("ahR")}<rect width="{W}" height="150" fill="#fff"/>
<text x="{W//2}" y="32" text-anchor="middle" font-size="17" font-weight="700" fill="{TEXT}">이 튜토리얼의 전체 흐름</text>
{''.join(out)}</svg>'''


def cnn_hook_svg():
    """Vertical CNN forward path with an SAE hung off every conv layer (sidecar)."""
    rows = [("입력 이미지", "3 · 32 · 32", None), ("conv1", "32 · 32 · 32", "32→256"),
            ("layer1", "32 · 32 · 32", "32→256"), ("layer2", "64 · 16 · 16", "64→512"),
            ("layer3", "128 · 8 · 8", "128→1024"), ("layer4", "256 · 4 · 4", "256→2048"),
            ("GAP (평균 풀링)", "256", None), ("fc → 예측", "100 클래스", None)]
    W, bx, bw, bh, y0, step = 760, 70, 210, 38, 56, 47
    out = []
    for i, (name, shape, exp) in enumerate(rows):
        y = y0 + i * step
        main = name in ("fc → 예측",)
        fill = "#fde9d6" if main else ("#eef0f2" if name in ("입력 이미지", "GAP (평균 풀링)") else "#e7eefb")
        out.append(f'<rect x="{bx}" y="{y}" width="{bw}" height="{bh}" rx="7" fill="{fill}" stroke="#cdd8ea"/>')
        out.append(f'<text x="{bx+12}" y="{y+24}" font-size="13" font-weight="700" fill="{TEXT}">{name}</text>')
        out.append(f'<text x="{bx+bw-12}" y="{y+24}" text-anchor="end" font-size="11.5" fill="{GRAY}">[{shape}]</text>')
        if i < len(rows) - 1:
            out.append(f'<line x1="{bx+bw/2}" y1="{y+bh}" x2="{bx+bw/2}" y2="{y+step}" stroke="{ARROW}" stroke-width="2" marker-end="url(#ahH)"/>')
        if exp:
            sx = bx + bw + 54
            out.append(f'<line x1="{bx+bw}" y1="{y+bh/2}" x2="{sx-2}" y2="{y+bh/2}" stroke="{ORANGE}" stroke-width="1.8" marker-end="url(#ahO)" stroke-dasharray="4,3"/>')
            out.append(f'<rect x="{sx}" y="{y+3}" width="150" height="{bh-6}" rx="7" fill="#fff4e8" stroke="#f0c79a"/>')
            out.append(f'<text x="{sx+10}" y="{y+20}" font-size="12" font-weight="700" fill="{ORANGE}">SAE</text>')
            out.append(f'<text x="{sx+45}" y="{y+20}" font-size="11" fill="{GRAY}">{exp} (×8)</text>')
            out.append(f'<text x="{sx+10}" y="{y+33}" font-size="9.5" fill="{GRAY}">feature (k=16 ON)</text>')
    note_y = y0 + len(rows) * step + 6
    return f'''<svg viewBox="0 0 {W} {note_y+70}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:740px;font-family:{FONT}">
{_marker("ahH")}<defs><marker id="ahO" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="{ORANGE}"/></marker></defs>
<rect width="{W}" height="{note_y+70}" fill="#fff"/>
<text x="{W//2}" y="32" text-anchor="middle" font-size="17" font-weight="700" fill="{TEXT}">SAE는 CNN 어디에 붙나 — 각 레이어 활성값에 sidecar로</text>
{''.join(out)}
<rect x="40" y="{note_y}" width="680" height="58" rx="8" fill="#f5f7fa" stroke="#dde3ea"/>
<text x="54" y="{note_y+24}" font-size="12.5" fill="{TEXT}">· <tspan font-weight="700">주황 점선 = SAE</tspan>: 각 레이어 활성값을 읽어 feature로 분해(옆에 붙는 sidecar). <tspan font-weight="700">forward(파랑) 경로는 안 바뀜.</tspan></text>
<text x="54" y="{note_y+44}" font-size="12.5" fill="{TEXT}">· <tspan font-weight="700">layer4 → GAP → fc 는 선형</tspan> → feature의 클래스 기여도를 정확히 계산할 수 있음(트리 가중치의 근거).</text>
</svg>'''


def feature_concept_svg(feat_id, concept, thumbs):
    """feature -> its top-activating images (index) -> a human name."""
    W, H = 900, 250
    tw = 78; tx0 = 318
    imgs = "".join(f'<image x="{tx0+i*(tw+8)}" y="78" width="{tw}" height="{tw}" href="{t}" style="image-rendering:pixelated"/>'
                   f'<rect x="{tx0+i*(tw+8)}" y="78" width="{tw}" height="{tw}" fill="none" stroke="#cdd8ea"/>' for i, t in enumerate(thumbs))
    nx = tx0 + len(thumbs) * (tw + 8) + 18
    return f'''<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:800px;font-family:{FONT}">
{_marker("ahC")}<rect width="{W}" height="{H}" fill="#fff"/>
<text x="{W//2}" y="32" text-anchor="middle" font-size="17" font-weight="700" fill="{TEXT}">feature를 어떻게 '읽나' — top 이미지로 이름 붙이기 (index)</text>
<text x="{W//2}" y="54" text-anchor="middle" font-size="12.5" fill="{GRAY}">feature는 그냥 번호. 가장 세게 켜는 이미지들을 모으면(index) 무슨 개념인지 보인다.</text>
<rect x="40" y="92" width="86" height="50" rx="8" fill="#e7eefb" stroke="#cdd8ea"/>
<text x="83" y="113" text-anchor="middle" font-size="13" font-weight="700" fill="{TEXT}">f{feat_id}</text>
<text x="83" y="131" text-anchor="middle" font-size="10.5" fill="{GRAY}">(번호일 뿐)</text>
<line x1="130" y1="117" x2="312" y2="117" stroke="{ARROW}" stroke-width="2" marker-end="url(#ahC)"/>
<text x="221" y="108" text-anchor="middle" font-size="11.5" fill="{GRAY}">가장 세게 켜는 top 이미지</text>
{imgs}
<line x1="{nx-14}" y1="117" x2="{nx-2}" y2="117" stroke="{ARROW}" stroke-width="2" marker-end="url(#ahC)"/>
<text x="{nx+8}" y="113" font-size="15" font-weight="700" fill="{ORANGE}">= {concept}</text>
<text x="{nx+8}" y="133" font-size="10.5" fill="{GRAY}">(사람이 붙인 이름)</text>
<rect x="40" y="178" width="740" height="50" rx="8" fill="#f5f7fa" stroke="#dde3ea"/>
<text x="54" y="200" font-size="12.5" fill="{TEXT}">트리 우측 패널은 이 top 이미지 + <tspan font-weight="700">활성맵</tspan>(이미지 어디서 켜지나) + <tspan font-weight="700">ERF</tspan>(어떤 입력 픽셀이 만드나)까지 같이 보여줍니다.</text>
<text x="54" y="218" font-size="12.5" fill="{GRAY}">→ 노드가 무슨 개념인지 한눈에 판단.</text>
</svg>'''


if __name__ == "__main__":
    import pathlib
    html = f'<!doctype html><meta charset=utf-8><body style="margin:0;background:#fff">{sae_encode_decode_html()}</body>'
    p = pathlib.Path("/tmp/diag3.html"); p.write_text(html)
    print(p)
