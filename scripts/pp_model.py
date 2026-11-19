#!/usr/bin/env python3
"""
纯 PyTorch PointPillars（不含自定义 CUDA 算子，Windows 直接跑）。

结构：
  PillarFeatureNet  逐柱 PointNet（9 维 -> 64 维，柱内 max pool）
  Backbone          250x250 -> 125x125 -> 63x63，再上采样拼回 125x125
  DetectionHead     每个 anchor 输出 1 个"自己类别"的分数 + 7 维框回归

设计取舍：每个 anchor 绑定一个类别（尺寸先验），所以分类头只需 1 通道
而不是 NUM_CLASS 通道——250k anchor x 8 类会白白多算 8 倍。
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pp_config as C  # noqa: E402
import pp_data as D  # noqa: E402

EPS = 1e-6


def conv_bn(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class PillarFeatureNet(nn.Module):
    def __init__(self, in_ch=C.IN_FEATURES, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, out_ch), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Linear(out_ch, out_ch), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        )
        self.out_ch = out_ch

    def forward(self, feat, coords, npts):
        """feat (B,P,T,C)  coords (B,P,2)=[iy,ix]  npts (B,P) -> (B,C,H,W)

        注意两处必须做掩码，否则补零会污染特征：
          1) 柱内 max-pool：补出来的零向量过 MLP 后是 MLP(0)（非零，因为 BN/偏置），
             可能盖过真实点，所以先把补零位置的输出压成 -1e4。
          2) scatter：补零柱的 coords 全是 (0,0)，会把格子 (0,0) 的真实特征覆盖成 0，
             所以先把无效柱的特征乘 0，再用 scatter_reduce(amax) 而不是覆盖式 scatter。
        """
        B, P, T, _ = feat.shape
        x = feat.reshape(B * P * T, -1)
        x = self.net(x).reshape(B, P, T, self.out_ch)

        mask = torch.arange(T, device=feat.device)[None, None, :] < npts[..., None]
        x = x.masked_fill(~mask[..., None], -1e4)
        x = x.max(dim=2).values                      # (B,P,C)

        valid = (npts > 0).to(x.dtype).unsqueeze(-1)  # (B,P,1)
        x = x * valid

        canvas = x.new_zeros(B, self.out_ch, C.GRID_Y, C.GRID_X)
        idx = coords[..., 0] * C.GRID_X + coords[..., 1]      # (B,P)
        canvas.view(B, self.out_ch, -1).scatter_reduce_(
            2, idx.unsqueeze(1).expand(-1, self.out_ch, -1),
            x.permute(0, 2, 1), reduce='amax', include_self=True)
        return canvas


class DeepBackbone(nn.Module):
    """标准 PointPillars 式颈：两个上采样分支，输出全分辨率（0.32 m）特征。

    默认的 Backbone 只下采样两次再上采样一次，检测头停在 125x125（0.64 m），
    定位量化精度直接砍半。这里把 b1 做深、加一条到 250x250 的上采样支路，
    检测头拿到 0.32 m 分辨率。
    """

    def __init__(self, cin=64, c1=64, c2=128, c3=256):
        super().__init__()
        self.b1 = nn.Sequential(conv_bn(cin, c1, 3, 1, 1),
                                conv_bn(c1, c1, 3, 1, 1))
        self.b2 = nn.Sequential(conv_bn(c1, c2, 3, 2, 1),
                                *([conv_bn(c2, c2, 3, 1, 1)] if C.WIDE else []))
        self.b3 = nn.Sequential(conv_bn(c2, c3, 3, 2, 1),
                                *([conv_bn(c3, c3, 3, 1, 1)] if C.WIDE else []))
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 2, stride=2),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.mid = conv_bn(c2 + c2, c2, 3, 1, 1)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, 2, stride=2),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True))
        self.out_ch = c1 + c1

    def forward(self, x):
        x1 = self.b1(x)                       # (B,64,250,250)
        x2 = self.b2(x1)                      # (B,128,125,125)
        x3 = self.b3(x2)                      # (B,256,63,63)
        u = self.up3(x3)
        u = u[:, :, :x2.shape[2], :x2.shape[3]]
        m = self.mid(torch.cat([x2, u], dim=1))   # (B,128,125,125)
        v = self.up2(m)
        v = v[:, :, :x1.shape[2], :x1.shape[3]]
        return torch.cat([x1, v], dim=1)      # (B,128,250,250)


class Backbone(nn.Module):
    def __init__(self, cin=64, c1=64, c2=128, c3=256):
        super().__init__()
        self.b1 = nn.Sequential(conv_bn(cin, c1, 3, 1, 1))
        self.b2 = nn.Sequential(conv_bn(c1, c2, 3, 2, 1))
        self.b3 = nn.Sequential(conv_bn(c2, c3, 3, 2, 1))
        self.up = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 2, stride=2),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.out_ch = c2 + c2

    def forward(self, x):
        x1 = self.b1(x)                 # (B,c1,250,250)
        x2 = self.b2(x1)                # (B,c2,125,125)
        x3 = self.b3(x2)                # (B,c3,63,63)
        u = self.up(x3)                 # (B,c2,126,126)
        u = u[:, :, :x2.shape[2], :x2.shape[3]]
        return torch.cat([x2, u], dim=1)


class PointPillars(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn = PillarFeatureNet()
        self.backbone = DeepBackbone(self.pfn.out_ch) if C.DEEP_NECK \
            else Backbone(self.pfn.out_ch)
        c = self.backbone.out_ch
        self.head = conv_bn(c, 128, 3, 1, 1)
        self.cls = nn.Conv2d(128, C.NA, 1)
        self.reg = nn.Conv2d(128, C.NA * 7, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 分类头偏置设负值，避免训练初期背景被大量误检
        nn.init.constant_(self.cls.bias, -4.0)

    def forward(self, feat, coords, npts):
        x = self.pfn(feat, coords, npts)
        x = self.backbone(x)
        x = self.head(x)
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]
        cls = self.cls(x).view(B, C.NA, H, W)
        reg = self.reg(x).view(B, C.NA, 7, H, W)
        return cls, reg


# ------------------------------------------------------------------ loss
def focal_loss(logits, y, valid, alpha=0.25, gamma=2.0):
    """logits (B,N)，y (B,N) in {0,1}，valid (B,N) bool（False = 忽略）"""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    pt = p * y + (1 - p) * (1 - y)
    w = alpha * (1 - pt).pow(gamma)
    loss = (w * ce * valid).sum()
    npos = (y * valid).sum().clamp(min=1.0)
    return loss / npos


def reg_loss(pred, target, weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2), beta=1 / 9,
             anchor_w=None):
    """pred/target (M,7)，M 可含 0 长度；anchor_w (M,) 可选逐样本权重"""
    if pred.numel() == 0:
        return pred.sum() * 0.0
    d = F.smooth_l1_loss(pred, target, beta=beta, reduction='none')
    w = pred.new_tensor(weights)
    per = (d * w).sum(dim=1)
    if anchor_w is not None:
        per = per * anchor_w
    return per.sum() / max(len(pred), 1)


def _class_weights(pos_cls, n_cls=C.NUM_CLASS):
    """按类别重加权：让每个类的正样本对损失的总贡献相等。

    背景：Car(4.5x2.0) 在 0.64m 网格上每个目标能覆盖 ~42 个 anchor，
    Person(0.78x0.74) 只有 ~3 个。再加上 Car 每帧 33 个、Person 18 个，
    结果 Person 只占全部正样本的 4%，focal loss 归一化后几乎没有梯度，
    Person AP 一直是 0。这里给稀有类的正样本放大权重。
    """
    if len(pos_cls) == 0:
        return None
    cnt = torch.bincount(pos_cls, minlength=n_cls).float().clamp(min=1.0)
    present = (cnt > 0).sum().clamp(min=1.0)
    w = len(pos_cls) / (present * cnt[pos_cls])
    return w


def compute_loss(cls_logits, reg_pred, pos_idx_list, reg_t_list, pos_cls_list,
                 cls_balance=True, heat_list=None):
    """cls_logits (B,NA,H,W) reg_pred (B,NA,7,H,W)

    heat_list 不为 None 时，分类目标用 [0,1] 软标签（高斯热图）代替 0/1 硬标签，
    其余焦点损失形式不变 —— 尺度与旧实现一致，可直接热启动。软标签使峰值唯一
    （只有离框中心最近的 anchor 是 1.0），推理时不再出现"平顶"导致中心随机偏移。
    """
    B = cls_logits.shape[0]
    cls_flat = cls_logits.permute(0, 2, 3, 1).reshape(B, -1)      # (B, H*W*NA)
    reg_flat = reg_pred.permute(0, 3, 4, 1, 2).reshape(B, -1, 7)  # (B, H*W*NA, 7)
    dev = cls_logits.device

    tot_cls = 0.0
    tot_reg = 0.0
    for b in range(B):
        pos = pos_idx_list[b].to(dev)
        if heat_list is not None:
            y = heat_list[b].to(dev).clamp(0.0, 1.0)
        else:
            y = torch.zeros(cls_flat.shape[1], device=dev)
            y[pos] = 1.0
        p = torch.sigmoid(cls_flat[b])
        ce = F.binary_cross_entropy_with_logits(cls_flat[b], y, reduction='none')
        pt = p * y + (1 - p) * (1 - y)
        per = 0.25 * (1 - pt).pow(2.0) * ce                # (N,)
        npos = max(len(pos), 1)
        if cls_balance and len(pos):
            pc = pos_cls_list[b].to(dev)
            w = _class_weights(pc)
            wfull = torch.ones_like(per)
            wfull[pos] = w
            tot_cls = tot_cls + (per * wfull).sum() / npos
        else:
            tot_cls = tot_cls + per.sum() / npos
        if len(pos):
            w = _class_weights(pos_cls_list[b].to(dev)) if cls_balance else None
            tot_reg = tot_reg + reg_loss(reg_flat[b][pos], reg_t_list[b].to(dev),
                                         anchor_w=w)
    return tot_cls / B, tot_reg / B


# ------------------------------------------------------------------ decode
def decode(reg, anchors):
    """reg (M,7) 相对量，anchors (M,7) -> (M,7) 绝对框 [x,y,z,L,W,H,hd]"""
    a = anchors
    diag = torch.sqrt(a[:, 3] ** 2 + a[:, 4] ** 2)
    out = torch.empty_like(a)
    out[:, 0] = a[:, 0] + reg[:, 0] * diag
    out[:, 1] = a[:, 1] + reg[:, 1] * diag
    out[:, 2] = a[:, 2] + reg[:, 2] * a[:, 5]
    out[:, 3] = a[:, 3] * torch.exp(reg[:, 3].clamp(-2, 2))
    out[:, 4] = a[:, 4] * torch.exp(reg[:, 4].clamp(-2, 2))
    out[:, 5] = a[:, 5] * torch.exp(reg[:, 5].clamp(-2, 2))
    th = a[:, 6] + reg[:, 6]
    out[:, 6] = th - torch.floor(th / np.pi + 0.5) * np.pi
    return out


PEAK = int(os.environ.get('PP_PEAK', '3'))


@torch.no_grad()
def predict_frame(model, feat, coords, npts, score_thr=0.10, topk=200,
                  snap_z=True, per_class=True):
    """单帧推理 -> (boxes (M,7) numpy, scores (M,), class_ids (M,))

    topk 在 per_class=True 时是**每类**的上限，否则是全局上限。
    全局模式下 Car 的 anchor 数量最多、分数也最高，会把 Person / Bicycle /
    Unknown 的 anchor 全部挤出 top-k（实测 Person 几乎进不来）。但全局模式
    预测框更少（~50/帧），对安静的测试时段（45 框/帧）精度更匹配。
    两种模式都在本地验证集上量过，见 reports/technical_report.md §6.2。
    """
    cls, reg = model(feat, coords, npts)
    smap = torch.sigmoid(cls[0])                              # (NA, H, W)
    reg = reg[0].permute(2, 3, 0, 1).reshape(-1, 7)           # (H*W*NA, 7)
    if PEAK > 0:
        # 分数场"取局部极大"：一个目标的响应在 anchor 图上是 14x6 格（Car 4.4x1.9
        # @0.32m）的一整片，per-class top-k 因此被两个目标就占满 —— 实测 topk 300
        # 只覆盖 ~10 辆车、AP 0.307，topk 3000 才 0.449，而阈值怎么调都无效
        # （0.03~0.30 AP 恒定）。先做 3x3 极大值抑制，一个目标只留 1~2 个候选，
        # top-k 的覆盖面就按"目标数"而不是"目标面积"计。
        mx = F.max_pool2d(smap[:, None], PEAK, stride=1,
                          padding=PEAK // 2).squeeze(1)
        smap = torch.where(smap >= (mx - 1e-6), smap, torch.zeros_like(smap))
    score = smap.permute(1, 2, 0).reshape(-1)                 # (H*W*NA,)
    idx = torch.nonzero(score > score_thr).squeeze(1)
    if len(idx) == 0:
        return np.zeros((0, 7), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    dev = idx.device

    if per_class:
        acls_all = torch.from_numpy(D.ANCHOR_CLS_FLAT).to(dev)[idx]
        s_all = score[idx]
        picks = []
        for c in range(C.NUM_CLASS):
            m = acls_all == c
            if not bool(m.any()):
                continue
            sc, ii = s_all[m], idx[m]
            if len(sc) > topk:
                sc, sel = sc.topk(topk)
                ii = ii[sel]
            picks.append(ii)
        if picks:
            idx = torch.cat(picks)
            idx = idx[torch.argsort(score[idx], descending=True)]
    elif len(idx) > topk:
        _, sel = score[idx].topk(topk)
        idx = idx[sel]

    anchors = torch.from_numpy(D.ANCHOR_FLAT).to(dev)[idx]
    acls = torch.from_numpy(D.ANCHOR_CLS_FLAT).to(dev)[idx]
    boxes = decode(reg[idx], anchors)
    if snap_z:
        # 底面全部落在同一个地面高度上，直接吸附
        boxes[:, 2] = C.GROUND_Z + boxes[:, 5] / 2.0
    return boxes.cpu().numpy(), score[idx].cpu().numpy(), acls.cpu().numpy()


# ------------------------------------------------------------------ NMS
def _corners(b):
    """b (N,7) -> (N,4,2) 逆时针"""
    x, y, _, dx, dy, _, h = [b[:, i] for i in range(7)]
    c, s = np.cos(h), np.sin(h)
    lx = np.array([0.5, -0.5, -0.5, 0.5])
    ly = np.array([0.5, 0.5, -0.5, -0.5])
    px = lx[None, :] * dx[:, None]
    py = ly[None, :] * dy[:, None]
    X = x[:, None] + c[:, None] * px - s[:, None] * py
    Y = y[:, None] + s[:, None] * px + c[:, None] * py
    return np.stack([X, Y], axis=-1)


def _polys(b):
    import shapely
    return shapely.polygons(_corners(b))


def bev_iou_matrix(a, b):
    import shapely
    pa, pb = _polys(a), _polys(b)
    inter = shapely.area(shapely.intersection(pa[:, None], pb[None, :]))
    aa = shapely.area(pa)[:, None]
    ab = shapely.area(pb)[None, :]
    union = aa + ab - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def nms(boxes, scores, iou_thr=0.25):
    """旋转框 BEV NMS，返回保留索引。"""
    if len(boxes) == 0:
        return np.zeros(0, np.int64)
    order = np.argsort(-scores)
    keep = []
    alive = np.ones(len(boxes), bool)
    for i in order:
        if not alive[i]:
            continue
        keep.append(i)
        alive[i] = False
        rest = np.flatnonzero(alive)
        if len(rest) == 0:
            break
        iou = bev_iou_matrix(boxes[i:i + 1], boxes[rest])[0]
        alive[rest[iou > iou_thr]] = False
    return np.array(keep, np.int64)


@torch.no_grad()
def detect(model, pts, device, score_thr=0.10, iou_thr=0.25, topk=600,
           moto_from_bike=0.0, snap_z=True, per_class=True):
    """整条推理链：pillar 化 -> 前向 -> 分类别 NMS。

    moto_from_bike > 0 时，把每个 Bicycle 预测复制一份作为 Motorcycle，
    分数乘以该系数。Motorcycle 真实只有 0.04 帧/帧、且尺寸与 Bicycle 几乎相同
    （1.97x0.84x1.83 vs 2.00x0.86x1.90），本质不可分；但 AP 是按类平均的，
    多一个类归零就少 1/7。复制一份低分 Motorcycle 相当于买保险。

    返回 (boxes (M,7) float32, scores (M,), class_ids (M,))
    """
    feat, coords, npts = D.pillarize(pts)
    if len(feat) == 0:
        return np.zeros((0, 7), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    ft = torch.from_numpy(feat)[None].to(device, non_blocking=True)
    ct = torch.from_numpy(coords).long()[None].to(device, non_blocking=True)
    nt = torch.from_numpy(npts).long()[None].to(device, non_blocking=True)
    boxes, scores, cls = predict_frame(model, ft, ct, nt, score_thr, topk,
                                       snap_z, per_class)
    ob, os_, oc = [], [], []
    for c in range(C.NUM_CLASS):
        m = cls == c
        if not m.any():
            continue
        idx = nms(boxes[m], scores[m], iou_thr)
        ob.append(boxes[m][idx])
        os_.append(scores[m][idx])
        oc.append(np.full(len(idx), c, np.int64))
    if not ob:
        return np.zeros((0, 7), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    out_b, out_s, out_c = np.concatenate(ob), np.concatenate(os_), np.concatenate(oc)
    if moto_from_bike > 0:
        mb = out_c == 2
        if mb.any():
            out_b = np.concatenate([out_b, out_b[mb]])
            out_s = np.concatenate([out_s, out_s[mb] * moto_from_bike])
            out_c = np.concatenate([out_c, np.full(int(mb.sum()), 3, np.int64)])
    return (out_b.astype(np.float32), out_s.astype(np.float32), out_c)


@torch.no_grad()
def detect_tta(model, pts, device, score_thr=0.10, iou_thr=0.25, topk=600,
               flips=(1,), dup=None, snap_z=True):
    """镜像 TTA + 跨类框复用。

    flips: 要镜像的轴。镜像 y 后 heading 取反即可还原，几何严格等价，
           所以两份预测可以按同类 NMS 合并，等价于"多一个视角"。
    dup:   {目标类: (来源类, 分数系数)}。Van 与 Car 在 BEV 里几乎同形，
           模型把 Van 全判成 Car，于是把 Car 框复制成 Van 低分框 —— 几何对得上
           就算真阳性，Van 那一类的 AP 从 0 直接有了。
           这是后处理，不引入任何标注信息。
    """
    outs = [detect(model, pts, device, score_thr, iou_thr, topk, snap_z=snap_z)]
    for ax in flips:
        q = np.ascontiguousarray(pts.copy())
        q[:, ax] = -q[:, ax]
        b, s, c = detect(model, q, device, score_thr, iou_thr, topk, snap_z=snap_z)
        if len(b):
            b = b.copy()
            b[:, ax] = -b[:, ax]
            b[:, 6] = -b[:, 6]
        outs.append((b, s, c))

    B = np.concatenate([o[0] for o in outs])
    S = np.concatenate([o[1] for o in outs])
    K = np.concatenate([o[2] for o in outs])
    if len(B) == 0:
        return B.astype(np.float32), S.astype(np.float32), K

    # 同类合并：保留分数最高的，抑制重叠副本
    keep = []
    for c in np.unique(K):
        m = K == c
        idx = nms(B[m], S[m], iou_thr)
        keep.append(np.flatnonzero(m)[idx])
    keep = np.concatenate(keep) if keep else np.zeros(0, np.int64)
    B, S, K = B[keep], S[keep], K[keep]

    if dup:
        eb, es, ek = [], [], []
        for dst, (src, f) in dup.items():
            m = K == C.CLASS_NAMES.index(src)
            if not m.any():
                continue
            d = C.CLASS_NAMES.index(dst)
            t = np.array(C.ANCHOR_SIZE[dst], np.float32)
            nb = B[m].copy()
            # 换成目标类的典型尺寸，否则长宽高对不上，IoU 过不了 0.5
            nb[:, 3:6] = t
            nb[:, 2] = C.GROUND_Z + t[2] / 2.0
            eb.append(nb)
            es.append(S[m] * f)
            ek.append(np.full(int(m.sum()), d, np.int64))
        if eb:
            B = np.concatenate([B] + eb)
            S = np.concatenate([S] + es)
            K = np.concatenate([K] + ek)
    return B.astype(np.float32), S.astype(np.float32), K


if __name__ == '__main__':
    m = PointPillars()
    n = sum(p.numel() for p in m.parameters())
    print(f'参数量 {n/1e6:.2f} M')
    feat = torch.zeros(2, 32768, C.MAX_PTS_PER_PILLAR, C.IN_FEATURES)
    coords = torch.zeros(2, 32768, 2, dtype=torch.long)
    npts = torch.full((2, 32768), 2, dtype=torch.long)
    cls, reg = m(feat, coords, npts)
    print('cls', tuple(cls.shape), 'reg', tuple(reg.shape))
    if torch.cuda.is_available():
        m = m.cuda()
        feat, coords, npts = feat.cuda(), coords.cuda(), npts.cuda()
        import time
        for _ in range(3):
            cls, reg = m(feat, coords, npts)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            cls, reg = m(feat, coords, npts)
        torch.cuda.synchronize()
        print(f'batch2 前向 {1000*(time.time()-t0)/10:.0f} ms')
        print(f'显存 {torch.cuda.max_memory_allocated()/2**20:.0f} MB')
