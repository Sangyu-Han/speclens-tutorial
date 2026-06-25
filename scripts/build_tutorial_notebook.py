"""Build the Colab tutorial notebook (cifar_speclens_tutorial.ipynb), 5 sections:
CNN+SAE -> mechanistic tree -> debugging -> why-confused -> spurious bug fix.
Heavy work loads from the precomputed artifact bundle; live cells are light.
Markdown + printed output are in Korean (the students are Korean).
Run: PYTHONPATH=. python scripts/build_tutorial_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

C = []
def md(s): C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})
def code(s): C.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": s.strip("\n").splitlines(keepends=True)})

md("""
# SpecLens 미니 튜토리얼 — 작은 CNN을 SAE로 해석하기

CIFAR-100을 학습한 작은 CNN(~71%)과, 각 레이어에 붙인 **희소 오토인코더(sparse autoencoder, SAE)**로:
(1) 인터랙티브 **mechanistic tree**(기계론적 트리)를 만들고, (2) 오분류를 **디버깅**하고, (3) **왜 두
클래스가 헷갈리는지** 보고, (4) **가짜 단서(spurious shortcut)**를 데이터로 잡아내고 고칩니다.

무거운 학습은 미리 해뒀습니다. 이 노트북은 작은 산출물만 받아 가벼운 셀만 돌립니다(무료 T4 GPU에서
약 10분). 정직한 주제: 해석가능성은 **진단**과 **진짜 버그 수정**에는 강력하지만, 이미 깨끗한
데이터에서 정확도를 공짜로 올려주는 마법 버튼은 아닙니다.
""")

code("""
# ---- 셋업 ----
# 공개 repo 하나만 clone하면 끝 — 코드 + 산출물 + CIFAR가 전부 repo 안에 있어 외부 다운로드(느린 토론토 서버) 없음.
import os, sys
if not os.path.isdir("SpecLens"):
    !git clone -q --depth 1 https://github.com/Sangyu-Han/speclens-tutorial.git SpecLens
%cd SpecLens
sys.path.insert(0, ".")
!pip -q install timm pyyaml pyarrow 2>/dev/null
ART = "cifar_tutorial_artifacts"
if not os.path.isdir(ART):
    !tar xzf cifar_tutorial_artifacts.tar.gz
# CIFAR-100: repo에 동봉된 split 사본을 합쳐 ./data 로 풀기 (torchvision 기본 서버가 매우 느려서 우회)
if os.path.isdir("cifar_data") and not os.path.isdir("data/cifar-100-python"):
    os.makedirs("data", exist_ok=True)
    !cat cifar_data/cifar-100-python.tar.gz.part-* > data/_c.tgz && tar xzf data/_c.tgz -C data && rm data/_c.tgz
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| 산출물:", sorted(os.listdir(ART)), "| CIFAR:", os.path.isdir("data/cifar-100-python"))
""")

md("""
## 1. 모델과 SAE — 먼저 개념부터

작은 CNN(~71%)을 불러와 정확도를 확인하고, **SAE가 무엇이고 CNN에 어떻게 붙는지**를 그림으로 이해합니다.
이 절만 제대로 봐두면 뒤의 트리·디버깅·shortcut이 전부 같은 부품(**feature**)으로 보입니다.
""")

code("""
import numpy as np, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.packs.cifar_cnn.models.model_loaders import load_cifar_cnn_model
from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD
from scripts.cifar_fri_feature import load_sae

EVAL_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
test = datasets.CIFAR100("./data", train=False, download=True, transform=EVAL_TF)
train_raw = datasets.CIFAR100("./data", train=True, download=True)
classes = test.classes
model = load_cifar_cnn_model({"ckpt": f"{ART}/cnn.pt"}, device=DEVICE).eval()

correct = 0
with torch.no_grad():
    for x, y in DataLoader(test, 256):
        correct += (model(x.to(DEVICE)).argmax(1).cpu() == y).sum().item()
print(f"CNN 테스트 정확도: {correct/len(test):.4f}")
sae4 = load_sae("model.layer4.0", f"{ART}/sae", DEVICE)
print("layer4 SAE: feature 2048개")
""")

md("""
### 전체 흐름
CNN은 잘 맞히지만 **블랙박스**입니다. 각 레이어의 활성값을 **SAE**로 개념 단위 **feature**로 분해하면 →
feature에 **이름**을 붙이고(index) → 클래스로 어떻게 합쳐지는지(**tree**) 보고 → 무엇이 잘못됐는지
(**디버깅·shortcut**) 추적할 수 있습니다. 아래 그림들이 이 순서입니다.
""")
code("""
from IPython.display import HTML, display
import scripts.cifar_sae_diagrams as D
display(HTML(D.roadmap_svg()))
""")

md("""
### SAE는 CNN 어디에 붙나
이미지는 conv1 → layer1 → … → layer4 → GAP → fc 를 거쳐 클래스가 됩니다. 각 레이어의 활성값(텐서)에
**SAE를 옆에(sidecar) 붙여 읽기만** 합니다 — forward 경로는 그대로. 특히 **layer4 → GAP → fc 는 선형**이라,
feature가 클래스에 주는 기여를 **정확히 계산**할 수 있습니다(뒤의 트리 가중치의 근거).
""")
code("display(HTML(D.cnn_hook_svg()))")

md("""
### SAE 한 개가 하는 일
SAE는 한 위치의 활성값(layer4면 256차원)을 받아 **과완비(2048개) 사전 중 16개만 켜는 희소 코드**로
인코딩하고, 다시 그것으로 원래 활성값을 **재구성**합니다. 켜진 feature 하나하나가 **단의미(monosemantic)
개념**이 되도록 (재구성 오차↓ + 희소) 학습됩니다.
""")
code("display(HTML(D.sae_encode_decode_html()))")

md("""
### 공간: feature는 '어디서' 켜지나
conv 활성값은 **공간(H×W) × 채널(C)** 구조라 같은 SAE가 **모든 위치에** 적용됩니다. 그래서 feature 하나는
'어느 위치에서 켜졌나' = **공간 활성맵**을 가집니다. 아래는 **실제 `f731`(오토바이) feature**의 layer4 4×4
활성맵 — 위(하늘)는 0, 아래(오토바이)에서 강하게 켜지고, 이미지에 겹치면 오토바이 위에 표시됩니다.
""")
code('''
import io, base64
from PIL import Image as _PIL
from scripts.cifar_fri_feature import CnnFri
_fri = CnnFri(f"{ART}/cnn.pt", f"{ART}/sae", DEVICE)
_cand = [i for i in range(len(train_raw.targets)) if train_raw.targets[i] == classes.index("motorcycle")][:60]
_best, _bv, _bmap = _cand[0], -1.0, None
for _i in _cand:
    _a = _fri.feat_map_gated(EVAL_TF(train_raw.data[_i]), "model.layer4.0", 731)
    if float(_a.max()) > _bv: _bv, _best, _bmap = float(_a.max()), _i, _a
def _b64(arr, s=96):
    _im = _PIL.fromarray(arr).resize((s, s), _PIL.NEAREST); _bf = io.BytesIO(); _im.save(_bf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(_bf.getvalue()).decode()
display(HTML(D.sae_spatial_html(_b64(train_raw.data[_best]), _bmap.tolist(), 731, "오토바이")))
''')

md("""
### feature 이름 = top 이미지 (index)
feature는 그냥 번호입니다. **가장 세게 켜는 이미지들을 모으면(index)** 무슨 개념인지 보입니다. 트리 우측
패널은 여기에 활성맵·ERF까지 더해 보여줘서, 노드가 무슨 개념인지 한눈에 판단하게 합니다.
""")
code('''
from scripts.cifar_mech_tree import load_index
from scripts.cifar_mech_tree_html import top_samples
_idx = load_index(f"{ART}/index", "model.layer4.0")
_sids = [s for s, _, _ in top_samples(_idx, 731, 5)]
display(HTML(D.feature_concept_svg(731, "오토바이", [_b64(train_raw.data[s]) for s in _sids])))
''')

md("""
> **한 줄 요약**
> - **SAE** = 폴리세만틱한 레이어 활성값을 **과완비·희소한 단의미 feature**로 푸는 사전학습기 (재구성 + 희소).
> - CNN에선 **각 공간 위치에** 적용 → feature는 **공간 활성맵**(어디서 켜지나)을 가짐.
> - feature 이름은 **top 이미지(index)**로 붙임. **layer4 → fc 가 선형** → feature→클래스 기여를 계산 가능.
>
> 아래에서 직접 `FEATURE` 번호를 바꿔 다른 feature의 top 이미지를 봐도 됩니다.
""")

md("""
**(선택) SAE를 직접 학습해보기.** 위의 미리 만든 SAE는 명령 하나로 만들어집니다 — **공유 activation
버퍼**가 한 번의 forward로 모든 레이어를 잡아내므로, 5개 레이어의 SAE가 동시에 학습됩니다. 주석을
풀면 재학습(T4에서 ~5분); 아니면 건너뛰세요(이미 로드됨).
""")
code("# !PYTHONPATH=. python scripts/train_sae_config.py --config configs/cifar_cnn_sae_colab.yaml   # ~5분, 5개 레이어 동시 학습")

code("""
import matplotlib.pyplot as plt
FEATURE = 731                                  # 바꿔보세요
norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
cap = {}; h = model.layer4.register_forward_hook(lambda m,i,o: cap.__setitem__("v", o))
acts, idxs = [], list(range(0, 50000, 5))
with torch.no_grad():
    for k in range(0, len(idxs), 512):
        xb = torch.stack([norm(train_raw.data[i]) for i in idxs[k:k+512]]).to(DEVICE)
        model(xb); v = cap["v"]
        a = sae4.encode(v.permute(0,2,3,1).reshape(-1,256)).reshape(v.shape[0],-1,2048)[:,:,FEATURE].amax(1)
        acts.append(a.cpu())
h.remove()
acts = torch.cat(acts); top = [idxs[i] for i in acts.argsort(descending=True)[:8]]
fig, ax = plt.subplots(1, 8, figsize=(12, 1.7))
for a, i in zip(ax, top):
    a.imshow(train_raw.data[i]); a.axis("off"); a.set_title(classes[train_raw.targets[i]][:9], fontsize=7)
fig.suptitle(f"layer4 feature {FEATURE}: top-activating images"); plt.show()
""")

md("""
## 2. Mechanistic tree (기계론적 트리)
클래스에서 출발해 위에서 아래로(top-down) 만든 트리: 각 레이어의 어떤 feature들이 그 클래스의 핵심
feature를 구성하는지 보여줍니다(엣지 = FRI attribution, feature 공간의 insertion/deletion으로 검증됨).
**완전한 인터랙티브** 버전(`tree/motorcycle/tree.html`)은 **맨 아래 셀에서 Colab 안에 바로** 띄웁니다
(노드 클릭 → 샘플 5개 + 활성화맵 + ERF, "구성됐는지"로 드릴다운). 먼저 핵심 노드 몇 개를 인라인으로:
""")

code("""
from IPython.display import Image, display
tree = f"{ART}/tree/motorcycle/details"
for f in ["L4_f731", "L3_f557", "L3_f690"]:
    p = f"{tree}/{f}.png"
    if os.path.exists(p):
        print(f); display(Image(p, width=560))
# f731 = 'motorcycle' feature; 기여 feature로 f557(빨간 차체), f690(둥근/곡선 -> 바퀴) 등이 있음
""")

md("""
**전체 인터랙티브 트리 — Colab 안에서 바로.** 아래 셀이 작은 웹서버로 트리를 띄웁니다. 노드를 클릭하면
우측 패널에 그 feature의 5개 샘플 + 활성화맵 + ERF가 뜨고, "composed of"로 더 깊이 내려갑니다. (학생도 동일하게 봅니다.)
""")

code('''
# 인터랙티브 트리 HTML 을 Colab 셀 안에 띄우기 (노드 클릭 -> 드릴다운).
# 상대경로(nodes/, details/)는 작은 웹서버가 해결해 줍니다.
import http.server, socketserver, threading, functools
from google.colab import output
TREE_DIR = f"{ART}/tree/motorcycle"
_handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=TREE_DIR)
_httpd = socketserver.TCPServer(("", 0), _handler)          # 빈 포트 자동 할당
_port = _httpd.server_address[1]
threading.Thread(target=_httpd.serve_forever, daemon=True).start()
output.serve_kernel_port_as_iframe(_port, path="/tree.html", height=780)
''')

md("""
## 3. 오분류 디버깅
`layer4 -> GAP -> fc`는 **선형**이라, 각 feature가 어떤 클래스로 미는 정도가 정확히
`평균활성_f x (fc.weight[클래스] · W_dec[feature])`이고, feature를 억제하면 추가 forward 없이 logit이
바로 이동합니다. 이를 이용해 오류 뒤의 **범인 feature**를 지목합니다.
""")

code("""
fcw = model.fc.weight.detach().cpu(); Wd = sae4.W_dec.detach().cpu()
A = Wd @ fcw.t()                                   # [2048,100] feature -> 클래스
cap = {}; h = model.layer4.register_forward_hook(lambda m,i,o: cap.__setitem__("v", o))
feats, logits, labels = [], [], []
with torch.no_grad():
    for x, y in DataLoader(test, 256):
        lg = model(x.to(DEVICE)).cpu(); v = cap["v"]
        feats.append(sae4.encode(v.permute(0,2,3,1).reshape(-1,256)).reshape(v.shape[0],-1,2048).mean(1).cpu())
        logits.append(lg); labels.append(y)
h.remove()
feats, logits, labels = torch.cat(feats), torch.cat(logits), torch.cat(labels)
preds = logits.argmax(1)

mis = (preds != labels).nonzero().squeeze(1); fixed = []
for i in mis.tolist():
    p, t = int(preds[i]), int(labels[i])
    f = int((feats[i] * A[:, p]).argmax())                       # 틀린 클래스를 가장 민 feature
    if int((logits[i] - feats[i, f] * A[f]).argmax()) == t:      # 그걸 억제하면 -> 정답?
        fixed.append((i, t, p, f))
print(f"오류 {len(mis)}개 중 {len(fixed)}개 ({100*len(fixed)/len(mis):.0f}%)가 feature 1개 억제로 교정됨")
for i, t, p, f in fixed[:6]:
    print(f"  img{i}: {classes[t]} -> {classes[p]}로 오분류  <- 범인 f{f} (미는 곳: {classes[int(A[f].argmax())]})")
""")

md("""
**전체 그림 — 샘플별 "왜 틀렸나" 트리.** 범인 하나를 지목하는 데서 그치지 않고, 틀린 이미지 하나를 그
이미지 *자신의* feature 활성을 따라 추적합니다: 어떤 feature가 틀린 클래스로 밀었고, 정답 클래스의
feature는 왜 잠잠했는지. 루트 = 틀린 클래스. §2처럼 **인터랙티브**(노드 클릭 → 우측 패널에 그 feature의
정체 + 입력 이미지, "composed of"로 드릴다운)로 Colab 안에 띄웁니다. (~30초)
""")

code('''
# 이 이미지가 왜 틀렸는지 "인터랙티브 트리" 생성 (~30초) → Colab 안에서 노드 클릭/드릴다운
!PYTHONPATH=. python scripts/cifar_misclass_tree_html.py --true bicycle --pred motorcycle --ckpt {ART}/cnn.pt --sae-root {ART}/sae --index-dir {ART}/index --data-root ./data --out-dir {ART}/misclass --device {DEVICE}
import glob, http.server, socketserver, threading, functools
from google.colab import output
D = sorted(glob.glob(f"{ART}/misclass/*_to_*/"))[-1]          # 방금 생성된 트리 폴더
_h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=D)
_s = socketserver.TCPServer(("", 0), _h); _p = _s.server_address[1]
threading.Thread(target=_s.serve_forever, daemon=True).start()
output.serve_kernel_port_as_iframe(_p, path="/tree.html", height=820)
# 다른 오류도: --true cloud --pred sea  |  --true baby --pred boy  |  --sample-id <0..9999>
''')

md("""
## 4. 두 클래스는 왜 헷갈릴까?
헷갈리는 쌍에서는 **공유(shared)** feature(둘 다에 반응, 차이 거의 없음)가 혼동을 일으키고, 어떤
**판별(discriminative)** feature가 그 *차이*를 담습니다. 각 feature의 분리력(클래스 A vs B 이미지에서의
활성에 대한 Cohen's d)을 모든 레이어에서 측정합니다.
""")

code("""
M = np.zeros((100,100), int)
for t, p in zip(labels.numpy(), preds.numpy()):
    if t != p: M[t, p] += 1
sym = M + M.T
a, b = np.unravel_index(sym.argmax(), sym.shape)
print(f"가장 헷갈리는 쌍: {classes[a]} <-> {classes[b]}  (n={sym[a,b]})")

from scripts.cifar_fri_feature import CnnFri
fri = CnnFri(f"{ART}/cnn.pt", f"{ART}/sae", DEVICE)
LAYERS = ["model.conv1","model.layer1.0","model.layer2.0","model.layer3.0","model.layer4.0"]
for L in LAYERS: fri.sae(L).configure_visualization_gating(mode="hard")
by = {c: [i for i in range(50000) if train_raw.targets[i]==c] for c in (a,b)}
def feats_of(cls, L):
    out=[]; cap={}; mod=fri.model
    for p in L.replace("model.","").split("."): mod = mod[int(p)] if p.isdigit() else getattr(mod,p)
    hh = mod.register_forward_hook(lambda m,i,o: cap.__setitem__("v",o))
    with torch.no_grad():
        for k in range(0,len(by[cls]),256):
            xb=torch.stack([norm(train_raw.data[i]) for i in by[cls][k:k+256]]).to(DEVICE); fri.model(xb)
            v=cap["v"]; out.append(fri.sae(L).encode(v.permute(0,2,3,1).reshape(-1,v.shape[1])).reshape(v.shape[0],-1,fri.sae(L).W_dec.shape[0]).mean(1).cpu())
    hh.remove(); return torch.cat(out)
print("레이어             최대 |Cohen d| (쌍을 분리하는 정도)")
for L in LAYERS:
    fa, fb = feats_of(a,L), feats_of(b,L)
    d = ((fa.mean(0)-fb.mean(0)) / (((fa.var(0)+fb.var(0))/2).clamp(min=1e-8).sqrt())).abs()
    print(f"  {L:16s} d={d.max():.2f}  (feature f{int(d.argmax())})")
print("=> 둘을 구분할 정보는 분명히 존재함(layer4에서 가장 강함). 모델이 그저 덜 쓸 뿐.")
""")

md("""
## 5. 가짜 단서(spurious shortcut) — SAE로 찾아 데이터로 고치기
모든 **apple(사과)** 학습 이미지에 마젠타색 코너 패치를 넣은 데이터로 '지름길(shortcut)' 모델을
학습시켰습니다. 모델은 `패치 => 사과`를 배워버렸죠. SAE가 그 **패치 feature**를 찾아냅니다. 데이터에서
패치를 빼고 재학습하면 고쳐집니다. (두 모델 모두 미리 만들어둠.)
""")

code("""
import json
meta = json.load(open(f"{ART}/spurious_meta.json")); C_cls = meta["C"]; PS = meta["patch_size"]
short = load_cifar_cnn_model({"ckpt": f"{ART}/shortcut_cnn.pt"}, device=DEVICE).eval()
clean = load_cifar_cnn_model({"ckpt": f"{ART}/clean_cnn.pt"}, device=DEVICE).eval()
PATCH = ((torch.tensor([1.,0.,1.]) - torch.tensor(CIFAR100_MEAN)) / torch.tensor(CIFAR100_STD))
def stamp(x): x=x.clone(); x[:, :PS, :PS] = PATCH[:,None,None]; return x

@torch.no_grad()
def attack_rate(m, lim=1500):
    n=c=0
    for i in range(len(test)):
        if test.targets[i]==C_cls: continue
        if int(m(stamp(test[i][0]).unsqueeze(0).to(DEVICE)).argmax())==C_cls: c+=1
        n+=1
        if n>=lim: break
    return c/n
print(f"대상 클래스 = {classes[C_cls]}")
print(f"shortcut 모델: 깨끗한 {classes[C_cls]} recall {meta['shortcut']['C_recall']:.2f} | 패치공격 성공률 {attack_rate(short):.2f}")
print(f"  -> SAE 패치 feature = f{meta['patch_feature']} (패치가 있을 때만 켜짐)")
print(f"깨끗한 데이터로 재학습: 깨끗한 {classes[C_cls]} recall {meta['fixed']['C_recall']:.2f} | 패치공격 성공률 {attack_rate(clean):.2f}")
print("나쁜 feature의 데이터 단서(패치)를 제거하니 모델이 진짜 개념을 학습함.")
""")

md("""
**shortcut을 트리로 (위 트리들과 같은 포맷).** 표준 파이프라인: shortcut CNN(apple에만 패치 학습) → 패치-인식
SAE(전레이어) → 인덱싱 → 트리. 공격: 패치 붙은 **bicycle**을 모델이 **apple**이라 합니다. 노드 클릭 → 우측 패널에
**개념(top 샘플) + 이 입력에서의 반응 + composed of**(드릴다운). 패치는 **apple 이미지에만** 찍어서(=실제 학습과 동일)
개념이 정직합니다 — 비-apple feature 개념엔 패치 없음. **패치 feature(<span style="color:red">빨강 ⚠</span>)의 개념 =
패치된 apple**이지만, actmap이 사과 몸통이 아니라 **코너에 발화**하고 "이 입력"은 apple도 아닌 패치된 bicycle인데 그 코너에서
발화해 apple로 갑니다 → 정체는 '코너'(=shortcut). seed는 delta(patched−clean). 고치는 법: 학습데이터에서 패치 제거 후
재학습(apple recall 0→0.85). (~1-2분)
""")

code('''
# 패치 공격 다층 트리: 패치된 bicycle -> apple, apple feature를 conv1까지 분해 (패치-학습 전레이어 SAE)
!PYTHONPATH=. python scripts/cifar_shortcut_tree_html.py --shortcut-ckpt {ART}/shortcut_cnn.pt --sae-root shortcut_sae_slim --meta {ART}/spurious_meta.json --data-root ./data --true bicycle --out-dir {ART}/shortcut_tree --device {DEVICE}
import http.server, socketserver, threading, functools
from google.colab import output
_h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=f"{ART}/shortcut_tree")
_s = socketserver.TCPServer(("", 0), _h); _p = _s.server_address[1]
threading.Thread(target=_s.serve_forever, daemon=True).start()
output.serve_kernel_port_as_iframe(_p, path="/tree.html", height=720)
''')

md("""
## 정리
- SAE feature로 CNN을 기계론적으로 읽을 수 있습니다: 트리로 구성하고, 오류 뒤의 범인을 지목하고,
  클래스가 *왜* 헷갈리는지(공유 vs 판별 feature, 그리고 어느 레이어에서) 봅니다.
- **공짜 이득**은 진짜 버그를 고칠 때 나옵니다(가짜 패치: 사과 recall 0 → 0.85). 이미 깨끗한
  데이터에서 라벨 정제나 contrastive 기법은 대개 전체 정확도를 올리기보다 **trade-off**입니다 —
  그리고 분석은 어떤 혼동이 *진짜* 유사성(girl/woman)인지, 아니면 고칠 수 있는 지름길인지 정직하게
  알려줍니다.
""")

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                               "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
Path("cifar_speclens_tutorial.ipynb").write_text(json.dumps(nb, indent=1))
print(f"wrote cifar_speclens_tutorial.ipynb ({len(C)} cells)")
