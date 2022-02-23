# -*- coding:utf-8 -*-

# @Filename: My_test
# @Project : Unsupervised_Domian_Adaptation
# @date    : 2021-11-04 19:04
# @Author  : Linshan
from data.nj import TestLoader
import logging
import cv2
from utils.tools import *
from utils.my_tools import *
from ever.util.param_util import count_model_parameters
from module.viz import VisualizeSegmm
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def evaluate_nj(model, cfg, is_training=False, ckpt_path=None):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    if cfg.SNAPSHOT_DIR is not None:
        vis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'vis-{}'.format(os.path.basename(ckpt_path)))
        palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
        viz_op = VisualizeSegmm(vis_dir+'_TEST', palette)
    if not is_training:
        model_state_dict = torch.load(ckpt_path)
        model.load_state_dict(model_state_dict, strict=True)
    model.eval()
    print(cfg.TEST_DATA_CONFIG)
    test_dataloader = TestLoader(cfg.TEST_DATA_CONFIG)

    save_path = './log/MSE/2urban/result'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        for ret in tqdm(test_dataloader):
            rgb = ret['rgb'].to(torch.device('cuda'))

            # old predict
            # cls = model(rgb)
            # slide predict
            # cls = tta_predict(model, rgb)
            cls = pre_slide(model, rgb, num_classes=7, tile_size=(512, 512), tta=True)

            cls = cls.argmax(dim=1).cpu().numpy()
            cv2.imwrite(save_path + '/' + ret['fname'][0], cls.reshape(1024, 1024).astype(np.uint8))

            for fname, pred in zip(ret['fname'], cls):
                viz_op(pred, fname.replace('tif', 'png'))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    seed_torch(2333)
    ckpt_path = './log/URBAN_0.4635.pth'
    from module.Encoder import Deeplabv2

    cfg = import_config('st.my.2urban')

    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type='resnet50',
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=True,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=7,
            use_aux=False,
            fc_dim=2048,
        ),
        inchannels=2048,
        num_classes=7
    )).cuda()
    evaluate_nj(model, cfg, False, ckpt_path)

