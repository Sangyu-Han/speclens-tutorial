"""Build every SAE explainer diagram to /tmp/*.html (real images + real f731 map from
the index), for chrome-screenshot previews / as a template for the notebook cells."""
import base64
import io

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from scripts.cifar_fri_feature import CnnFri
from scripts.cifar_mech_tree import load_index
from scripts.cifar_mech_tree_html import top_samples
from scripts import cifar_sae_diagrams as D
from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD


def b64(arr, size=96):
    im = Image.fromarray(arr).resize((size, size), Image.NEAREST)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def page(html, name):
    open(f"/tmp/{name}.html", "w").write(
        '<!doctype html><meta charset=utf-8><body style="margin:0;background:#fff">' + html + "</body>")


dev = "cuda:0" if torch.cuda.is_available() else "cpu"
fri = CnnFri("outputs/cifar_speclens/cnn.pt", "outputs/cifar_speclens/sae", dev)
norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
tr = datasets.CIFAR100("/home/sangyu/Desktop/Master/CBM_test/data", train=True, download=False)

FEAT, CONC = 731, "오토바이"
idx = load_index("outputs/cifar_speclens/index", "model.layer4.0")
sids = [s for s, _, _ in top_samples(idx, FEAT, 6)]
bmap = np.asarray(fri.feat_map_gated(norm(tr.data[sids[0]]), "model.layer4.0", FEAT))
top5 = [b64(tr.data[s]) for s in sids[:5]]

page(D.roadmap_svg(), "diag_roadmap")
page(D.cnn_hook_svg(), "diag_cnn")
page(D.sae_encode_decode_html(), "diag3")
page(D.sae_spatial_html(b64(tr.data[sids[0]]), bmap.tolist(), FEAT, CONC), "diag3b")
page(D.feature_concept_svg(FEAT, CONC, top5), "diag_concept")
print(f"built; f{FEAT} top sids {sids[:5]}, map max {bmap.max():.1f}")
