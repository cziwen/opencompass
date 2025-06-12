from mmengine.dataset import BaseDataset
from opencompass.registry import LOAD_DATASET

@LOAD_DATASET.register_module()
class MyFinDataset(BaseDataset):
    def __init__(self, path: str, **kwargs):
        self.path = path

        # ✅ 移除 BaseDataset 不接受的字段
        kwargs.pop('abbr', None)
        kwargs.pop('reader_cfg', None)
        kwargs.pop('infer_cfg', None)
        kwargs.pop('evaluator', None)

        super().__init__(**kwargs)

    def load_data_list(self):
        import json
        data_list = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                data_list.append({
                    'instruction': sample['instruction'],
                    'input': sample.get('input', ''),
                    'output': sample['output'],
                })
        return data_list