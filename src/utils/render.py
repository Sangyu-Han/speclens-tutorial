
import torch
import numpy as np
import matplotlib.cm
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import skimage.io
import skimage.feature
import skimage.filters
from skimage import img_as_ubyte
import cv2
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def vec2im(V, shape = () ):
    '''
    Transform an array V into a specified shape - or if no shape is given assume a square output format.

    Parameters
    ----------

    V : numpy.ndarray
        an array either representing a matrix or vector to be reshaped into an two-dimensional image

    shape : tuple or list
        optional. containing the shape information for the output array if not given, the output is assumed to be square

    Returns
    -------

    W : numpy.ndarray
        with W.shape = shape or W.shape = [np.sqrt(V.size)]*2

    '''

    if len(shape) < 2:
        shape = [np.sqrt(V.size)]*2
        shape = map(int, shape)
    return np.reshape(V, shape)


def enlarge_image(img, scaling = 3):
    '''
    Enlarges a given input matrix by replicating each pixel value scaling times in horizontal and vertical direction.

    Parameters
    ----------

    img : numpy.ndarray
        array of shape [H x W] OR [H x W x D]

    scaling : int
        positive integer value > 0

    Returns
    -------

    out : numpy.ndarray
        two-dimensional array of shape [scaling*H x scaling*W]
        OR
        three-dimensional array of shape [scaling*H x scaling*W x D]
        depending on the dimensionality of the input
    '''

    if scaling < 1 or not isinstance(scaling,int):
        print ('scaling factor needs to be an int >= 1')

    if len(img.shape) == 2:
        H,W = img.shape

        out = np.zeros((scaling*H, scaling*W))
        for h in range(H):
            fh = scaling*h
            for w in range(W):
                fw = scaling*w
                out[fh:fh+scaling, fw:fw+scaling] = img[h,w]

    elif len(img.shape) == 3:
        H,W,D = img.shape

        out = np.zeros((scaling*H, scaling*W,D))
        for h in range(H):
            fh = scaling*h
            for w in range(W):
                fw = scaling*w
                out[fh:fh+scaling, fw:fw+scaling,:] = img[h,w,:]

    return out


def repaint_corner_pixels(rgbimg, scaling = 3):
    '''
    DEPRECATED/OBSOLETE.

    Recolors the top left and bottom right pixel (groups) with the average rgb value of its three neighboring pixel (groups).
    The recoloring visually masks the opposing pixel values which are a product of stabilizing the scaling.
    Assumes those image ares will pretty much never show evidence.

    Parameters
    ----------

    rgbimg : numpy.ndarray
        array of shape [H x W x 3]

    scaling : int
        positive integer value > 0

    Returns
    -------

    rgbimg : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3]
    '''


    #top left corner.
    rgbimg[0:scaling,0:scaling,:] = (rgbimg[0,scaling,:] + rgbimg[scaling,0,:] + rgbimg[scaling, scaling,:])/3.0
    #bottom right corner
    rgbimg[-scaling:,-scaling:,:] = (rgbimg[-1,-1-scaling, :] + rgbimg[-1-scaling, -1, :] + rgbimg[-1-scaling,-1-scaling,:])/3.0
    return rgbimg


def digit_to_rgb(X, scaling=3, shape = (), cmap = 'binary'):
    '''
    Takes as input an intensity array and produces a rgb image due to some color map

    Parameters
    ----------

    X : numpy.ndarray
        intensity matrix as array of shape [M x N]

    scaling : int
        optional. positive integer value > 0

    shape: tuple or list of its , length = 2
        optional. if not given, X is reshaped to be square.

    cmap : str
        name of color map of choice. default is 'binary'

    Returns
    -------

    image : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3] , where H*W == M*N
    '''

    #create color map object from name string
    cmap = eval('matplotlib.cm.{}'.format(cmap))

    image = enlarge_image(vec2im(X,shape), scaling) #enlarge
    image = cmap(image.flatten())[...,0:3].reshape([image.shape[0],image.shape[1],3]) #colorize, reshape

    return image

def make_ERF_figure(ERF:torch.Tensor,target_img:torch.Tensor,alpha=0.2,cmap='viridis',padding_portion=0.5):

    erf = ERF.squeeze().cpu().detach().numpy()

    H_size,W_size = erf.shape

    W_above_thres = (erf.sum(axis=0)>erf.sum(axis=0).mean()).nonzero()[0]
    H_above_thres = (erf.sum(axis=1)>erf.sum(axis=1).mean()).nonzero()[0]
    W_min,W_max = W_above_thres[0], W_above_thres[-1] +1
    H_min,H_max = H_above_thres[0], H_above_thres[-1] +1
    W_range = W_max - W_min
    H_range = H_max - H_min
    W_padding = int(W_range * padding_portion)
    H_padding = int(H_range * padding_portion)

    W_start = np.clip(W_min - W_padding,a_min=0,a_max=W_size)
    W_end = np.clip(W_max + W_padding,a_min=0,a_max=W_size)
    H_start = np.clip(H_min - H_padding,a_min=0,a_max=H_size)
    H_end = np.clip(H_max + H_padding,a_min=0,a_max=H_size)

    my_cm = plt.cm.get_cmap('viridis')
    normed_ERF = (erf - erf.min()) / (erf.max() - erf.min())
    ERF_img = my_cm(normed_ERF)[:,:,:3]

    img = target_img.copy()
    img[H_min:H_max,W_min:W_max,:] = alpha * target_img[H_min:H_max,W_min:W_max,:] + (1-alpha) * ERF_img[H_min:H_max,W_min:W_max,:]
    result_img = img[H_start:H_end,W_start:W_end]
    return result_img

def hm_to_rgb(R, X = None, scaling = 1, shape = (), sigma = 2, cmap = 'hot', normalize = True):
    '''
    Takes as input an intensity array and produces a rgb image for the represented heatmap.
    optionally draws the outline of another input on top of it.

    Parameters
    ----------

    R : numpy.ndarray
        the heatmap to be visualized, shaped [M x N]

    X : numpy.ndarray
        optional. some input, usually the data point for which the heatmap R is for, which shall serve
        as a template for a black outline to be drawn on top of the image
        shaped [M x N]

    scaling: int
        factor, on how to enlarge the heatmap (to control resolution and as a inverse way to control outline thickness)
        after reshaping it using shape.

    shape: tuple or list, length = 2
        optional. if not given, X is reshaped to be square.

    sigma : double
        optional. sigma-parameter for the canny algorithm used for edge detection. the found edges are drawn as outlines.

    cmap : str
        optional. color map of choice

    normalize : bool
        optional. whether to normalize the heatmap to [-1 1] prior to colorization or not.

    Returns
    -------

    rgbimg : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3] , where H*W == M*N
    '''
    
    if isinstance(R, torch.Tensor):
        R = R.squeeze().cpu().detach().numpy()

    #create color map object from name string
    cmap = eval('matplotlib.cm.{}'.format(cmap))

    if normalize:
        R = R / np.max(np.abs(R)) # normalize to [-1,1] wrt to max relevance magnitude
        #R = np.interp(R, (R.min(),R.max()), (-1,1))
        R = (R + 1.)/2. # shift/normalize to [0,1] for color mapping

    R = R
    R = enlarge_image(R, scaling)
    rgb = cmap(R.flatten())[...,0:3].reshape([R.shape[0],R.shape[1],3])
    # rgb = repaint_corner_pixels(rgb, scaling) #obsolete due to directly calling the color map with [0,1]-normalized inputs

    if not X is None: #compute the outline of the input
        #X = enlarge_image(vec2im(X,shape), scaling)
        xdims = X.shape
        Rdims = R.shape

        # if not np.all(xdims == Rdims):
        #     print 'transformed heatmap and data dimension mismatch. data dimensions differ?'
        #     print 'R.shape = ',Rdims, 'X.shape = ', xdims
        #     print 'skipping drawing of outline\n'
        # else:
        #     #edges = skimage.filters.canny(X, sigma=sigma)
        #     edges = skimage.feature.canny(X, sigma=sigma)
        #     edges = np.invert(np.dstack([edges]*3))*1.0
        #     rgb *= edges # set outline pixels to black color

    return rgb

def torch_to_image(tensor, mean=MEAN, std=STD):
    """
    Helper function to convert torch tensor containing input data into image.
    """
    if len(tensor.shape) == 4:
        img = tensor.permute(0, 2, 3, 1)
    elif len(tensor.shape) == 3 and tensor.shape[0] == 3:
        img = tensor.permute(1, 2, 0)
    elif len(tensor.shape) == 3 and tensor.shape[2] == 3:
        img = tensor

    img = img.contiguous().squeeze().detach().cpu().numpy()

    img = img * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
    return np.clip(img, 0, 1)
    
def save_image(rgb_images, path, gap = 2):
    '''
    Takes as input a list of rgb images, places them next to each other with a gap and writes out the result.

    Parameters
    ----------

    rgb_images : list , tuple, collection. such stuff
        each item in the collection is expected to be an rgb image of dimensions [H x _ x 3]
        where the width is variable

    path : str
        the output path of the assembled image

    gap : int
        optional. sets the width of a black area of pixels realized as an image shaped [H x gap x 3] in between the input images

    Returns
    -------

    image : numpy.ndarray
        the assembled image as written out to path
    '''

    sz = []
    image = []
    for i in range(len(rgb_images)):
        if not sz:
            sz = rgb_images[i].shape
            image = rgb_images[i]
            gap = np.zeros((sz[0],gap,sz[2]))
            continue
        if not sz[0] == rgb_images[i].shape[0] and sz[1] == rgb_images[i].shape[2]:
            print ('image',i, 'differs in size. unable to perform horizontal alignment')
            print ('expected: Hx_xD = {0}x_x{1}'.format(sz[0],sz[1]))
            print ('got     : Hx_xD = {0}x_x{1}'.format(rgb_images[i].shape[0],rgb_images[i].shape[1]))
            print ('skipping image\n')
        else:
            image = np.hstack((image,gap,rgb_images[i]))

    image *= 255
    image = image.astype(np.uint8)

    print ('saving image to ', path)
    skimage.io.imsave(path,image)
    return image

# def make_ERF_figure(
#     ERF: torch.Tensor,
#     target_img: np.ndarray | torch.Tensor,
#     token_idx: int | torch.Tensor | None = None,   # None이면 토큰 강조 X
#     *,
#     alpha: float = 0.3,
#     cmap: str = "viridis",
#     padding_portion: float = 0.5,
#     cls_color: tuple = (0.5, 0.5, 0.5),
#     bg_color: tuple = (0.9, 0.9, 0.9),
#     highlight_color: tuple = (1, 0, 0),
#     extra_cutoff_erf: bool = True,
#     min_threshold: float = 0.2,
#     # ── NEW ─────────────────────────────────────────
#     segmentation_only: bool = False,
#     highlight_on_seg: bool = False,
# ):
#     """
#     1) ERF(2D)와 원본 이미지를 받아 heat-map 시각화 생성
#     2) segmentation_only 옵션:
#        True  → segmentation map만 반환
#        False → (미니맵 | 원본+heat-map | segmentation map) 형태로 반환
#     3) highlight_on_seg 옵션:
#        segmentation_only=True일 때, 원래 토큰 위치에 빨간 테두리를 그림
#     """

#     # ---------------------- 사전 처리 ----------------------
#     if isinstance(token_idx, torch.Tensor):
#         token_idx = int(token_idx.clone().cpu().detach())
#     if isinstance(target_img, torch.Tensor):
#         target_img = np.array(target_img.clone().cpu().detach())

#     erf = ERF.squeeze().cpu().detach().numpy()
#     H_size, W_size = erf.shape

#     erf[erf < 0] = 0
#     W_max = erf.max(axis=0)
#     H_max = erf.max(axis=1)
#     W_min, W_max = (W_max > -1).nonzero()[0][[0, -1]] + [0, 1]
#     H_min, H_max = (H_max > -1).nonzero()[0][[0, -1]] + [0, 1]

#     # padding
#     W_padding = int((W_max - W_min) * padding_portion)
#     H_padding = int((H_max - H_min) * padding_portion)
#     W_start = np.clip(W_min - W_padding, 0, W_size)
#     W_end   = np.clip(W_max + W_padding, 0, W_size)
#     H_start = np.clip(H_min - H_padding, 0, H_size)
#     H_end   = np.clip(H_max + H_padding, 0, H_size)

#     # ------------------- 컬러맵 및 cutoff ------------------
#     cm = plt.get_cmap(cmap)
#     norm_erf = (erf - erf.min()) / (erf.max() - erf.min() + 1e-8)
#     ERF_img = cm(norm_erf)[:, :, :3]

#     # percentile 기반 cutoff
#     p95 = np.percentile(norm_erf, 95)
#     ratio = norm_erf[norm_erf >= p95].sum() / (norm_erf.sum() + 1e-8)
#     cutoff = max(p95, min_threshold * ratio)

#     mask = np.clip(norm_erf, 0, cutoff) / (cutoff + 1e-8)
#     mask[norm_erf > cutoff] = 1.0
#     seg_full = (mask[:, :, None] * target_img).astype(np.float32)  # segmentation map

#     # ───────────────────── segmentation-only 모드 ─────────────────────
#     if segmentation_only:
#         seg_img = (seg_full * 255).astype(np.uint8)

#         # 빨간 테두리 강조
#         if highlight_on_seg and token_idx not in (None, 0):
#             H_full, W_full = seg_img.shape[:2]
#             ph, pw = H_full // 16, W_full // 16
#             r, c = divmod(token_idx - 1, 16)
#             top, left = r * ph, c * pw
#             bottom, right = top + ph, left + pw
#             cv2.rectangle(seg_img,
#                           (left, top), (right, bottom),
#                           (0, 0, 255), thickness=2)

#         return seg_img
#     # ────────────────────────────────────────────────────────────────

#     # ---------------- heat-map을 원본에 블렌딩 ----------------
#     if target_img.dtype not in (np.float32, np.float64):
#         base_img = (target_img / 255.0).astype(np.float32)
#     else:
#         base_img = target_img.copy()

#     # ERF가 검출된 영역에만 블렌딩
#     base_img[H_min:H_max, W_min:W_max] = (
#         alpha * base_img[H_min:H_max, W_min:W_max] +
#         (1 - alpha) * ERF_img[H_min:H_max, W_min:W_max]
#     )

#     # ---------------- box_img (확장 영역 + 빨간 박스) -------------
#     box_img = base_img[H_start:H_end, W_start:W_end].copy()
#     H_box, W_box = box_img.shape[:2]
#     # 빨간 박스
#     box_img[H_min-H_start : H_min-H_start+2, W_min-W_start : W_max-W_start] = [1, 0, 0]
#     box_img[H_max-H_start-2 : H_max-H_start, W_min-W_start : W_max-W_start] = [1, 0, 0]
#     box_img[H_min-H_start : H_max-H_start, W_min-W_start : W_min-W_start+2] = [1, 0, 0]
#     box_img[H_min-H_start : H_max-H_start, W_max-W_start-2 : W_max-W_start] = [1, 0, 0]

#     # ---------------- 토큰 검은 테두리 ----------------------
#     if token_idx not in (None, 0):
#         H_full, W_full = base_img.shape[:2]
#         ph, pw = H_full // 16, W_full // 16
#         r, c = divmod(token_idx - 1, 16)
#         top, left = r * ph, c * pw
#         bottom, right = top + ph, left + pw
#         cv2.rectangle(base_img, (left, top), (right, bottom), (0, 0, 0), 2)

#         # box_img에도 동일 좌표 변환 후 표시
#         tb, lb = top - H_start, left - W_start
#         bb, rb = bottom - H_start, right - W_start
#         if 0 <= tb < H_box and 0 <= bb <= H_box and 0 <= lb < W_box and 0 <= rb <= W_box:
#             cv2.rectangle(box_img, (lb, tb), (rb, bb), (0, 0, 0), 2)

#     # ---------------- 결합 -------------------------------
#     crop_erf = base_img[H_min:H_max, W_min:W_max]
#     crop_seg = seg_full[H_min:H_max, W_min:W_max]

#     # 높이 맞추기
#     if crop_erf.shape[0] < box_img.shape[0]:
#         pad = np.ones((box_img.shape[0] - crop_erf.shape[0], crop_erf.shape[1], 3), np.float32)
#         crop_erf = np.vstack((crop_erf, pad))
#         crop_seg = np.vstack((crop_seg, pad))
#     elif crop_erf.shape[0] > box_img.shape[0]:
#         pad = np.ones((crop_erf.shape[0] - box_img.shape[0], box_img.shape[1], 3), np.float32)
#         box_img = np.vstack((box_img, pad))

#     left_img  = (box_img * 255).astype(np.uint8)
#     mid_img   = (crop_erf * 255).astype(np.uint8)
#     right_img = (crop_seg * 255).astype(np.uint8)
#     combined  = np.hstack((left_img, mid_img, right_img)) if extra_cutoff_erf else np.hstack((left_img, mid_img))

#     # ---------------- 미니맵 ------------------------------
#     if token_idx is not None:
#         cell = max(1, (H_end - H_start) // (16 * 3))
#         mini = _make_token_mini_map(
#             num_tokens=257,
#             highlight_idx=token_idx,
#             cell_size=cell,
#             cls_color=cls_color,
#             bg_color=bg_color,
#             highlight_color=highlight_color,
#         )
#         # 세로 맞추기
#         if mini.shape[0] < combined.shape[0]:
#             pad = np.ones((combined.shape[0] - mini.shape[0], mini.shape[1], 3), np.uint8) * 255
#             mini = np.vstack((mini, pad))
#         elif mini.shape[0] > combined.shape[0]:
#             pad = np.ones((mini.shape[0] - combined.shape[0], combined.shape[1], 3), np.uint8) * 255
#             combined = np.vstack((combined, pad))
#         final_img = np.hstack((mini, combined))
#     else:
#         final_img = combined

#     return final_img


def make_ERF_figure(
    ERF: torch.Tensor | np.ndarray,
    target_img: np.ndarray | torch.Tensor,
    token_idx: int | None = None,
    *,
    alpha: float = 0.4,
    cmap: str = "plasma",
    min_threshold: float = 0.5,
    value_threshold: float = 0.15,
    cls_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    bg_color: tuple[float, float, float] = (0.9, 0.9, 0.9),
    highlight_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
):
    """Overlay a 2‑D *ERF* heat‑map onto *target_img*.

    A pixel is blended with opacity *alpha* **only** when the normalised
    heat‑map value is ≥ *value_threshold*; otherwise the original pixel is
    left untouched, making low‑activation areas fully transparent.

    Parameters
    ----------
    ERF, target_img :  see below.
    token_idx : int | None, default = ``None``
        If given (``1‥256``), appends a ViT patch mini‑map and draws a black
        rectangle around the selected patch on the overlay.
    alpha : float, default = ``0.9``
        Opacity applied to *high‑value* heat‑map pixels.
    cmap : str, default = ``"plasma"``
        Matplotlib colour‑map name used to colourise *ERF*.
    value_threshold : float in [0,1], default = ``0.1``
        Heat‑map normalised value below which pixels become *fully
        transparent* (no blending).
    """

    # ------------------------------------------------------------------
    # 1.  Convert inputs to NumPy & shape checks
    # ------------------------------------------------------------------
    erf = ERF.detach().cpu().squeeze().numpy() if isinstance(ERF, torch.Tensor) else np.asarray(ERF)
    if erf.ndim != 2:
        raise ValueError("ERF must be 2‑D (H×W)")

    img = target_img.detach().cpu().numpy() if isinstance(target_img, torch.Tensor) else np.asarray(target_img)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("target_img must be H×W×3")
    if target_img.dtype not in (np.float32, np.float64):
        img_f = (target_img / 255.0).astype(np.float32).copy()
    else:
        img_f = target_img.copy()
    # ------------------------------------------------------------------
    # 2.  Resize heat‑map to match image size if necessary
    # ------------------------------------------------------------------
    H, W = img.shape[:2]
    if erf.shape != (H, W):
        erf = cv2.resize(erf, (W, H), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    # 3.  Normalise and colourise heat‑map
    # ------------------------------------------------------------------
    denom = erf.max() - erf.min() + 1e-8
    normed_ERF = (erf - erf.min()) / denom
    

    norm = normed_ERF
    cmap_fn = plt.get_cmap(cmap)
    heat_rgb = (cmap_fn(norm)[..., :3] * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # 4.  Alpha‑blend *only* where norm ≥ threshold
    # ------------------------------------------------------------------

    heat_f = heat_rgb.astype(np.float32) / 255.0

    overlay_f = img_f.copy()
    mask = norm >= value_threshold  # True where we apply blending
    overlay_f[mask] = (1.0 - alpha) * img_f[mask] + alpha * heat_f[mask]

    overlay = (overlay_f * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # 5.  Draw patch rectangle when token_idx given
    # ------------------------------------------------------------------
    if token_idx is not None and token_idx != 0:
        patch_h, patch_w = H // 16, W // 16
        row, col = divmod(token_idx - 1, 16)
        top, left = row * patch_h, col * patch_w
        bottom, right = top + patch_h, left + patch_w
        cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 0), 2)

    # ------------------------------------------------------------------
    # 6.  Return overlay alone when no mini‑map requested
    # ------------------------------------------------------------------
    if token_idx is None:
        return overlay

    # ------------------------------------------------------------------
    # 7.  Build mini‑map and concatenate horizontally
    # ------------------------------------------------------------------
    cell_size = max(1, H // (16 * 3))
    mini = _make_token_mini_map(
        num_tokens=257,
        highlight_idx=token_idx,
        cell_size=cell_size,
        cls_color=cls_color,
        bg_color=bg_color,
        highlight_color=highlight_color,
    )

    # Pad shorter image so that heights match
    if mini.shape[0] < overlay.shape[0]:
        pad_height = overlay.shape[0] - mini.shape[0]
        pad = np.full((pad_height, mini.shape[1], 3), 255, dtype=np.uint8)
        mini = np.vstack([mini, pad])
    elif overlay.shape[0] < mini.shape[0]:
        pad_height = mini.shape[0] - overlay.shape[0]
        pad = np.full((pad_height, overlay.shape[1], 3), 255, dtype=np.uint8)
        overlay = np.vstack([overlay, pad])

    return np.hstack([mini, overlay])

# def make_ERF_figure(
#     ERF: torch.Tensor | np.ndarray,
#     target_img: np.ndarray | torch.Tensor,
#     token_idx: int | None = None,
#     *,
#     alpha: float = 0.4,
#     cmap: str = "plasma",
#     min_threshold: float = 0.5,
#     value_threshold: float = 0.15,
#     cls_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
#     bg_color: tuple[float, float, float] = (0.9, 0.9, 0.9),
#     highlight_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
# ):
#     """Overlay a 2‑D *ERF* heat‑map onto *target_img*.

#     A pixel is blended with opacity *alpha* **only** when the normalised
#     heat‑map value is ≥ *value_threshold*; otherwise the original pixel is
#     left untouched, making low‑activation areas fully transparent.

#     Parameters
#     ----------
#     ERF, target_img :  see below.
#     token_idx : int | None, default = ``None``
#         If given (``1‥256``), appends a ViT patch mini‑map and draws a black
#         rectangle around the selected patch on the overlay.
#     alpha : float, default = ``0.9``
#         Opacity applied to *high‑value* heat‑map pixels.
#     cmap : str, default = ``"plasma"``
#         Matplotlib colour‑map name used to colourise *ERF*.
#     value_threshold : float in [0,1], default = ``0.1``
#         Heat‑map normalised value below which pixels become *fully
#         transparent* (no blending).
#     """

#     # ------------------------------------------------------------------
#     # 1.  Convert inputs to NumPy & shape checks
#     # ------------------------------------------------------------------
#     erf = ERF.detach().cpu().squeeze().numpy() if isinstance(ERF, torch.Tensor) else np.asarray(ERF)
#     if erf.ndim != 2:
#         raise ValueError("ERF must be 2‑D (H×W)")

#     img = target_img.detach().cpu().numpy() if isinstance(target_img, torch.Tensor) else np.asarray(target_img)
#     if img.ndim != 3 or img.shape[2] != 3:
#         raise ValueError("target_img must be H×W×3")
#     if target_img.dtype not in (np.float32, np.float64):
#         img_f = (target_img / 255.0).astype(np.float32).copy()
#     else:
#         img_f = target_img.copy()
#     # ------------------------------------------------------------------
#     # 2.  Resize heat‑map to match image size if necessary
#     # ------------------------------------------------------------------
#     H, W = img.shape[:2]
#     if erf.shape != (H, W):
#         erf = cv2.resize(erf, (W, H), interpolation=cv2.INTER_LINEAR)

#     # ------------------------------------------------------------------
#     # 3.  Normalise and colourise heat‑map
#     # ------------------------------------------------------------------
#     denom = erf.max() - erf.min() + 1e-8
#     normed_ERF = (erf - erf.min()) / denom
    
#     cutoff_ERF = normed_ERF.copy()

#     percentile = 95
#     cutoff = np.percentile(cutoff_ERF,percentile)
#     high_mask = cutoff_ERF >= cutoff

#     # 각각의 합 계산
#     high_sum = cutoff_ERF[high_mask].sum()
#     low_sum = cutoff_ERF[~high_mask].sum()

#     # 비율 계산 (high / low)
#     ratio = high_sum / (cutoff_ERF.sum() + 1e-8) 
    
#     cutoff = max(cutoff,min_threshold * ratio)
#     # print(f'threshold: {cutoff}')
#     # cutoff = normed_ERF[normed_ERF > 1e-8].mean()
#     cutoff_ERF = np.clip(cutoff_ERF,0,cutoff) / (cutoff+1e-8) # clip cutoff to 1 and linearlize
#     # cutoff = cutoff
#     # cutoff_ERF[cutoff_ERF < cutoff] = 0
#     cutoff_ERF[normed_ERF > cutoff] = 1
    
#     norm = cutoff_ERF
#     cmap_fn = plt.get_cmap(cmap)
#     heat_rgb = (cmap_fn(norm)[..., :3] * 255).astype(np.uint8)

#     # ------------------------------------------------------------------
#     # 4.  Alpha‑blend *only* where norm ≥ threshold
#     # ------------------------------------------------------------------

#     heat_f = heat_rgb.astype(np.float32) / 255.0

#     overlay_f = img_f.copy()
#     mask = norm >= value_threshold  # True where we apply blending
#     overlay_f[mask] = (1.0 - alpha) * img_f[mask] + alpha * heat_f[mask]

#     overlay = (overlay_f * 255).astype(np.uint8)

#     # ------------------------------------------------------------------
#     # 5.  Draw patch rectangle when token_idx given
#     # ------------------------------------------------------------------
#     if token_idx is not None and token_idx != 0:
#         patch_h, patch_w = H // 16, W // 16
#         row, col = divmod(token_idx - 1, 16)
#         top, left = row * patch_h, col * patch_w
#         bottom, right = top + patch_h, left + patch_w
#         cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 0), 2)

#     # ------------------------------------------------------------------
#     # 6.  Return overlay alone when no mini‑map requested
#     # ------------------------------------------------------------------
#     if token_idx is None:
#         return overlay

#     # ------------------------------------------------------------------
#     # 7.  Build mini‑map and concatenate horizontally
#     # ------------------------------------------------------------------
#     cell_size = max(1, H // (16 * 3))
#     mini = _make_token_mini_map(
#         num_tokens=257,
#         highlight_idx=token_idx,
#         cell_size=cell_size,
#         cls_color=cls_color,
#         bg_color=bg_color,
#         highlight_color=highlight_color,
#     )

#     # Pad shorter image so that heights match
#     if mini.shape[0] < overlay.shape[0]:
#         pad_height = overlay.shape[0] - mini.shape[0]
#         pad = np.full((pad_height, mini.shape[1], 3), 255, dtype=np.uint8)
#         mini = np.vstack([mini, pad])
#     elif overlay.shape[0] < mini.shape[0]:
#         pad_height = mini.shape[0] - overlay.shape[0]
#         pad = np.full((pad_height, overlay.shape[1], 3), 255, dtype=np.uint8)
#         overlay = np.vstack([overlay, pad])

#     return np.hstack([mini, overlay])





def _make_token_mini_map(
    num_tokens: int,
    highlight_idx: int,
    cell_size: int,
    cls_color: tuple,
    bg_color: tuple,
    highlight_color: tuple
):
    """
    미니맵: [CLS](0) + 16x16=256 → (17 x 16) 격자.
    highlight_idx 토큰을 highlight_color로 표시.
    """
    H, W = 16, 17  # 행=16, 열=17
    mini_map = np.ones((H, W, 3), dtype=np.float32)
    mini_map *= bg_color  # 전체 배경
    mini_map[:,0] = (1,1,1)
    

    # [CLS] 위치 (0,0)
    mini_map[0, 0] = cls_color

    if highlight_idx == 0:
        # [CLS]
        mini_map[0, 0] = highlight_color
    elif 1 <= highlight_idx <= 256:
        # 1~256
        patch_i = highlight_idx
        r = (patch_i - 1) // 16
        c = 1 + (patch_i - 1) % 16
        mini_map[r, c] = highlight_color

    # 픽셀 확대
    mini_map = cv2.resize(
        mini_map,
        (W * cell_size, H * cell_size),
        interpolation=cv2.INTER_NEAREST
    )

    mini_map = (mini_map * 255).astype(np.uint8)
    return mini_map


# def make_ERF_figure(ERF: torch.Tensor, target_img: torch.Tensor, alpha=0.3, cmap='viridis', padding_portion=0.5): # 원래는 alpha = 0, cmap = hot
#     erf = ERF.squeeze().cpu().detach().numpy()

#     H_size, W_size = erf.shape

#     W_above_thres = (erf.sum(axis=0) > erf.sum(axis=0).mean()).nonzero()[0]
#     H_above_thres = (erf.sum(axis=1) > erf.sum(axis=1).mean()).nonzero()[0]
#     W_min, W_max = W_above_thres[0], W_above_thres[-1] + 1
#     H_min, H_max = H_above_thres[0], H_above_thres[-1] + 1
#     W_range = W_max - W_min
#     H_range = H_max - H_min
#     W_padding = int(W_range * padding_portion)
#     H_padding = int(H_range * padding_portion)

#     W_start = np.clip(W_min - W_padding, a_min=0, a_max=W_size)
#     W_end = np.clip(W_max + W_padding, a_min=0, a_max=W_size)
#     H_start = np.clip(H_min - H_padding, a_min=0, a_max=H_size)
#     H_end = np.clip(H_max + H_padding, a_min=0, a_max=H_size)

#     my_cm = plt.get_cmap(cmap)
#     normed_ERF = (erf - erf.min()) / (erf.max() - erf.min())
#     ERF_img = my_cm(normed_ERF)[:, :, :3]

#     img = target_img.copy()
#     img[H_min:H_max, W_min:W_max, :] = alpha * target_img[H_min:H_max, W_min:W_max, :] + (1-alpha) * ERF_img[H_min:H_max, W_min:W_max, :]

#     cropped_img_ERF = img[H_min:H_max, W_min:W_max]
#     cropped_img_original = target_img[H_min:H_max, W_min:W_max]

#     # ERF 패딩 부분으로 box_img 생성
#     box_img = target_img[H_start:H_end, W_start:W_end].copy()
#     H_box, W_box, _ = box_img.shape
#     box_img[H_min-H_start:H_min-H_start+2, W_min-W_start:W_max-W_start, :] = [1, 0, 0]  # 위쪽 선
#     box_img[H_max-H_start-2:H_max-H_start, W_min-W_start:W_max-W_start, :] = [1, 0, 0]  # 아래쪽 선
#     box_img[H_min-H_start:H_max-H_start, W_min-W_start:W_min-W_start+2, :] = [1, 0, 0]  # 왼쪽 선
#     box_img[H_min-H_start:H_max-H_start, W_max-W_start-2:W_max-W_start, :] = [1, 0, 0]  # 오른쪽 선

#     # 높이를 맞추기 위해 하얀색 패딩 추가
#     if cropped_img_ERF.shape[0] < box_img.shape[0]:
#         padding_height = box_img.shape[0] - cropped_img_ERF.shape[0]
#         padding = np.ones((padding_height, cropped_img_ERF.shape[1], 3))  # 하얀색 패딩
#         cropped_img_ERF = np.vstack((cropped_img_ERF, padding))
#     elif cropped_img_ERF.shape[0] > box_img.shape[0]:
#         padding_height = cropped_img_ERF.shape[0] - box_img.shape[0]
#         padding = np.ones((padding_height, box_img.shape[1], 3))  # 하얀색 패딩
#         box_img = np.vstack((box_img, padding))

#     # 두 이미지를 하나로 결합
#     combined_img = np.hstack((box_img, cropped_img_ERF))
#     combined_img = (combined_img * 255).astype(np.uint8)
#     return combined_img.astype(np.uint8)