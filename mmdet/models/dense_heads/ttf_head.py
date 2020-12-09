import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init
import numpy as np
from mmcv.ops import ModulatedDeformConv2dPack, batched_nms
from ..utils import gen_ellipse_gaussian_target
from mmdet.core import multi_apply, force_fp32
from mmdet.core.anchor import calc_region
from mmcv.cnn import ConvModule, bias_init_with_prob, build_norm_layer
# from .anchor_head import AnchorHead
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead


class ShortcutConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 paddings,
                 activation_last=False):
        super(ShortcutConv2d, self).__init__()
        assert len(kernel_sizes) == len(paddings)

        layers = []
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes, paddings)):
            inc = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(inc, out_channels, kernel_size, padding=padding))
            if i < len(kernel_sizes) - 1 or activation_last:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y

def bbox_areas(bboxes, keep_axis=False):
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (y_max - y_min + 1) * (x_max - x_min + 1)
    if keep_axis:
        return areas[:, None]
    return areas

@HEADS.register_module
class TTFHead(BaseDenseHead):

    def __init__(self,
                 inplanes=(64, 128, 256, 512),
                 planes=(256, 128, 64),
                 use_dla=False,
                 base_down_ratio=32,
                 head_conv=256,
                 wh_conv=64,
                 hm_head_conv_num=2,
                 wh_head_conv_num=2,
                 num_classes=80,
                 shortcut_kernel=3,
                 norm_cfg=dict(type='BN'),
                 shortcut_cfg=(1, 2, 3),
                 wh_offset_base=16.,
                 wh_area_process='log',
                 loss_heatmap=dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     loss_weight=1),
                 loss_bbox=dict(
                     type='GIoULoss',
                     loss_weight=5),
                 alpha=0.54,
                 train_cfg=None,
                 test_cfg=None,):
        super(TTFHead, self).__init__()
        assert len(planes) in [2, 3, 4]
        self.shortcut_num = min(len(inplanes) - 1, len(planes))
        assert self.shortcut_num == len(shortcut_cfg)
        assert wh_area_process in [None, 'norm', 'log', 'sqrt']
        self.inplanes = inplanes
        self.planes = planes
        self.use_dla = use_dla
        self.head_conv = head_conv
        self.num_classes = num_classes
        self.wh_offset_base = wh_offset_base
        self.wh_area_process = wh_area_process
        self.loss_heatmap = build_loss(
            loss_heatmap) if loss_heatmap is not None else None
        self.loss_bbox = build_loss(
            loss_bbox) if loss_bbox is not None else None
        self.alpha = alpha
        self.fp16_enabled = False
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.down_ratio = base_down_ratio // 2 ** len(planes)
        self.wh_planes = 4
        self.base_loc = None
        self.hm_head_conv_num = hm_head_conv_num
        self.wh_head_conv_num = wh_head_conv_num
        self.wh_conv = wh_conv
        self.norm_cfg = norm_cfg
        self.shortcut_kernel = shortcut_kernel
        self.shortcut_cfg = shortcut_cfg
        self._init_layers()

    def _make_shortcut(self,
                       inplanes,
                       planes,
                       shortcut_cfg,
                       kernel_size=3,
                       padding=1):
        assert len(inplanes) == len(planes) == len(shortcut_cfg)

        shortcut_layers = nn.ModuleList()
        for (inp, outp, layer_num) in zip(
                inplanes, planes, shortcut_cfg):
            assert layer_num > 0
            layer = ShortcutConv2d(
                inp, outp, [kernel_size] * layer_num, [padding] * layer_num)
            shortcut_layers.append(layer)
        return shortcut_layers

    def _make_upsample(self, inplanes, planes, norm_cfg=None):
        mdcn = ModulatedDeformConv2dPack(inplanes, planes, 3, stride=1,
                                       padding=1, dilation=1, deformable_groups=1)
        up = nn.UpsamplingBilinear2d(scale_factor=2)

        layers = []
        layers.append(mdcn)
        if norm_cfg:
            layers.append(build_norm_layer(norm_cfg, planes)[1])
        layers.append(nn.ReLU(inplace=True))
        layers.append(up)

        return nn.Sequential(*layers)

    def _make_head(self, out_channel, conv_num=1, head_conv_plane=None):
        head_convs = []
        head_conv_plane = self.head_conv if not head_conv_plane else head_conv_plane
        for i in range(conv_num):
            inp = self.planes[-1] if i == 0 else head_conv_plane
            head_convs.append(ConvModule(inp, head_conv_plane, 3, padding=1))

        inp = self.planes[-1] if conv_num <= 0 else head_conv_plane
        head_convs.append(nn.Conv2d(inp, out_channel, 1))
        return nn.Sequential(*head_convs)

    def _init_ttf_kpt_layers(self):
        self.deconv_layers = nn.ModuleList()
        self.shortcut_layers = nn.ModuleList()

        self.deconv_layers.append(
            self._make_upsample(self.inplanes[-1], self.planes[0], norm_cfg=self.norm_cfg))
        for i in range(1, len(self.planes)):
            self.deconv_layers.append(
                self._make_upsample(self.planes[i - 1], self.planes[i], norm_cfg=self.norm_cfg))

        padding = (self.shortcut_kernel - 1) // 2
        self.shortcut_layers = self._make_shortcut(self.inplanes[:-1][::-1][:self.shortcut_num],
                                self.planes[:self.shortcut_num],
                                self.shortcut_cfg,
            kernel_size=self.shortcut_kernel, padding=padding)

        # heads
        self.wh = self._make_head(self.wh_planes, self.wh_head_conv_num, self.wh_conv)
        self.hm = self._make_head(self.num_classes, self.hm_head_conv_num)

    def init_weights(self):
        for _, m in self.shortcut_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for _, m in self.hm.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.hm[-1], std=0.01, bias=bias_cls)

        for _, m in self.wh.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def _init_layers(self):
        self._init_ttf_kpt_layers()

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        x = feats[-1]
        if not self.use_dla:
            for i, upsample_layer in enumerate(self.deconv_layers):
                x = upsample_layer(x)
                if i < len(self.shortcut_layers):
                    shortcut = self.shortcut_layers[i](feats[-i - 2])
                    x = x + shortcut

        hm = self.hm(x)
        wh = F.relu(self.wh(x)) * self.wh_offset_base

        return hm, wh

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh'))
    def get_bboxes(self,
                   pred_heatmaps,
                   pred_whs,
                   img_metas,
                   rescale=False,
                   with_nms=False):

        assert len(pred_heatmaps) == len(img_metas)
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_single(
                    pred_heatmaps[img_id:img_id + 1, :],
                    pred_whs[img_id:img_id + 1, :],
                    img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))

        return result_list


    def _get_bboxes_single(self,
                           pred_heatmap,
                           pred_wh,
                           img_meta,
                           rescale=False,
                           with_nms=False):
        if isinstance(img_meta, (list, tuple)):
            img_meta = img_meta[0]
        batch_bboxes, batch_scores, batch_clses = self.decode_heatmap(
            pred_heatmap=pred_heatmap.sigmoid(),
            pred_wh=pred_wh,
            img_meta=img_meta,
            k=self.test_cfg.ttf_topk,
            kernel=self.test_cfg.local_maximum_kernel)


        if rescale:
            batch_bboxes /= batch_bboxes.new_tensor(img_meta['scale_factor'])
        bboxes = batch_bboxes.view([-1, 4])
        scores = batch_scores.view([-1, 1])
        clses = batch_clses.view([-1, 1])

        idx = scores.argsort(dim=0, descending=True)
        bboxes = bboxes[idx].view([-1, 4])
        scores = scores[idx].view(-1)
        clses = clses[idx].view(-1)

        detections = torch.cat([bboxes, scores.unsqueeze(-1)], -1)
        keepinds = (detections[:, -1] > -0.1)
        detections = detections[keepinds]
        labels = clses[keepinds]
        if with_nms:
            detections, labels = self._bboxes_nms(detections, labels,
                                                  self.test_cfg)

        return detections, labels

    def get_targets(self,
                    gt_boxes,
                    gt_labels,
                    feat_shape):
        """
        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            reg_weight: tensor, same as box_target.
        """

        heatmap, box_target, reg_weight = multi_apply(
            self.target_single_image,
            gt_boxes,
            gt_labels,
            feat_shape=feat_shape
        )

        heatmap, box_target = [torch.stack(t, dim=0).detach() for t in [heatmap, box_target]]
        reg_weight = torch.stack(reg_weight, dim=0).detach()

        return heatmap, box_target, reg_weight
    def target_single_image(self, gt_boxes, gt_labels, feat_shape):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w) or (80 * 4, h, w).
            reg_weight: tensor, same as box_target
        """
        _, output_h, output_w = feat_shape
        heatmap_channel = self.num_classes

        heatmap = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        fake_heatmap = gt_boxes.new_zeros((output_h, output_w))
        box_target = gt_boxes.new_ones((self.wh_planes, output_h, output_w)) * -1
        reg_weight = gt_boxes.new_zeros((self.wh_planes // 4, output_h, output_w))

        if self.wh_area_process == 'log':
            boxes_areas_log = bbox_areas(gt_boxes).log()
        elif self.wh_area_process == 'sqrt':
            boxes_areas_log = bbox_areas(gt_boxes).sqrt()
        else:
            boxes_areas_log = bbox_areas(gt_boxes)
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))

        if self.wh_area_process == 'norm':
            boxes_area_topk_log[:] = 1.

        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        feat_gt_boxes = gt_boxes / self.down_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0,
                                               max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,
                                               max=output_h - 1)
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)

        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()

        # larger boxes have lower priority than small boxes.
        for k in range(boxes_ind.shape[0]):
            cls_id = gt_labels[k]
            fake_heatmap = fake_heatmap.zero_()
            fake_heatmap = gen_ellipse_gaussian_target(fake_heatmap, ct_ints[k],
                                                       h_radiuses_alpha[k],
                                                       w_radiuses_alpha[k])
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)
            box_target_inds = fake_heatmap > 0
            box_target[:, box_target_inds] = gt_boxes[k][:, None]
            local_heatmap = fake_heatmap[box_target_inds]
            ct_div = local_heatmap.sum()
            local_heatmap *= boxes_area_topk_log[k]
            reg_weight[0, box_target_inds] = local_heatmap / ct_div
            # tmpp = reg_weight.squeeze(0).cpu()
            # tmp = tmpp.numpy()
            # import matplotlib.pyplot as plt
            # plt.imshow(tmp*255)
            # plt.axis('off')
            # plt.show()
        return heatmap, box_target, reg_weight

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh'))
    def loss(self,
             pred_heatmap,
             pred_wh,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        targets = self.get_targets(
            gt_bboxes,
            gt_labels,
            pred_heatmap[-1].shape)
        hm_loss, wh_loss = self.loss_calc(pred_heatmap, pred_wh, *targets)
        loss_dict = dict(heatmap_loss=hm_loss, size_loss=wh_loss)
        return loss_dict


    def loss_calc(self,
                  pred_hm,
                  pred_wh,
                  heatmap,
                  box_target,
                  reg_weight):
        """

        Args:
            pred_hm: tensor, (batch, 80, h, w).
            pred_wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            heatmap: tensor, same as pred_hm.
            box_target: tensor, same as pred_wh.
            wh_weight: tensor, same as pred_wh.

        Returns:
            hm_loss
            wh_loss
        """
        H, W = pred_hm.shape[2:]

        hm_loss = self.loss_heatmap(
            pred_hm.sigmoid(),
            heatmap,
            avg_factor=max(1, heatmap.eq(1).sum()))

        if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
            base_step = self.down_ratio
            shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        # (batch, h, w, 4)
        pred_boxes = torch.cat((self.base_loc - pred_wh[:, [0, 1]],
                                self.base_loc + pred_wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)
        # (batch, h, w, 4)
        boxes = box_target.permute(0, 2, 3, 1)

        mask = reg_weight.view(-1, H, W)
        avg_factor = mask.sum() + 1e-4
        pos_mask = mask > 0
        weight = mask[pos_mask].float()
        bboxes1 = pred_boxes[pos_mask].view(-1, 4)
        bboxes2 = boxes[pos_mask].view(-1, 4)
        wh_loss = self.loss_bbox(bboxes1, bboxes2, weight, avg_factor=avg_factor)

        return hm_loss, wh_loss

    def _bboxes_nms(self, bboxes, labels, cfg):
        out_bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1], labels,
                                       cfg.nms_cfg)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature according to index.

        Args:
            feat (Tensor): Target feature map.
            ind (Tensor): Target coord index.
            mask (Tensor | None): Mask of featuremap. Default: None.

        Returns:
            feat (Tensor): Gathered feature.
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).repeat(1, 1, dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _local_maximum(self, heat, kernel=3):
        """Extract local maximum pixel with given kernal.

        Args:
            heat (Tensor): Target heatmap.
            kernel (int): Kernel size of max pooling. Default: 3.

        Returns:
            heat (Tensor): A heatmap where local maximum pixels maintain its
                own value and other positions are 0.
        """
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _transpose_and_gather_feat(self, feat, ind):
        """Transpose and gather feature according to index.

        Args:
            feat (Tensor): Target feature map.
            ind (Tensor): Target coord index.

        Returns:
            feat (Tensor): Transposed and gathered feature.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _topk(self, scores, k=20):
        """Get top k positions from heatmap.

        Args:
            scores (Tensor): Target heatmap with shape
                [batch, num_classes, height, width].
            k (int): Target number. Default: 20.

        Returns:
            tuple[torch.Tensor]: Scores, indexes, categories and coords of
                topk keypoint. Containing following Tensors:

            - topk_scores (Tensor): Max scores of each topk keypoint.
            - topk_inds (Tensor): Indexes of each topk keypoint.
            - topk_clses (Tensor): Categories of each topk keypoint.
            - topk_ys (Tensor): Y-coord of each topk keypoint.
            - topk_xs (Tensor): X-coord of each topk keypoint.
        """
        batch, _, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
        topk_clses = topk_inds // (height * width)
        topk_inds = topk_inds % (height * width)
        topk_ys = topk_inds // width
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    def decode_heatmap(self,
                       pred_heatmap,
                       pred_wh,
                       img_meta,
                       k=100,
                       kernel=3):

        batch, _, height, width = pred_heatmap.size()
        inp_h, inp_w, _ = img_meta['pad_shape']

        # perform nms on heatmaps
        heat = self._local_maximum(pred_heatmap, kernel)

        # (batch, topk)
        scores, inds, clses, ys, xs = self._topk(heat, k=k)
        xs = xs.view(batch, k, 1) * self.down_ratio
        ys = ys.view(batch, k, 1) * self.down_ratio

        wh = self._transpose_and_gather_feat(pred_wh, inds)
        wh = wh.view(batch, k, 4)

        clses = clses.view(batch, k, 1).float()
        scores = scores.view(batch, k, 1)

        bboxes = torch.cat([xs - wh[..., [0]], ys - wh[..., [1]],
                            xs + wh[..., [2]], ys + wh[..., [3]]], dim=2)
        return bboxes, scores, clses
