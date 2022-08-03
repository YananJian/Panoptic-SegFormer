import torch
from mmcv.runner import checkpoint
from mmdet.apis.inference import init_detector,LoadImage, inference_detector
import easymd
import re
from mmcv.runner.checkpoint import _load_checkpoint,load_state_dict
from collections import OrderedDict

config = '/home/azure4yanan/work/Panoptic-SegFormer/configs/panformer/panformer_swinl_24e_coco_panoptic.py'
#config = '/home/azure4yanan/work/Panoptic-SegFormer/configs/panformer/panformer_pvtb5_24e_coco_panoptic.py'

checkpoint = "/home/azure4yanan/work/Panoptic-SegFormer/converted_panoptic_segformer_swinl_2x.pth"
#checkpoint = '/home/azure4yanan/work/Panoptic-SegFormer/converted_panformer_pvtb5_24e_coco_panoptic.pth'
img  = 'test_images/view_1.jpg'


def convert_ckpt(filename, revise_keys=[(r'^module\.', '')]):
    defautl_revise_keys = [
        ('\\.mask_head\\.','.things_mask_head.'),
        ('\\.mask_head2\\.','.stuff_mask_head.'),
        ('\\.cls_branches2\\.', '.cls_thing_branches.'),

    ]
    revise_keys.extend(defautl_revise_keys)
    checkpoint = _load_checkpoint(filename, None, None)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
 
    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata
    torch.save(state_dict, '/home/azure4yanan/work/Panoptic-SegFormer/converted_panoptic_segformer_swinl_2x.pth')

#convert_ckpt(checkpoint)

results = {
    'img': './'+img
}
model = init_detector(config,checkpoint=checkpoint)

import time
from torch.profiler import profile, record_function, ProfilerActivity
imgs = ['./test_images/apartment_room/view_1.jpg', 
        './test_images/apartment_room/view_2.jpg',
        './test_images/apartment_room/view_3.jpg',
        './test_images/apartment_room/view_4.jpg']

'''
for i in range(10):
    inference_detector(model, imgs)
with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        inference_detector(model, [imgs[0]])
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
print ('--------------------')
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
time_sum = 0

for i in range(3):
    start = time.time()
    inference_detector(model, imgs[::-1])
    end = time.time()
    print(end-start)
    time_sum += (end-start)

time_sum /= 12
print('avg total time:', time_sum)
'''
inference_detector(model, [imgs[0]])
