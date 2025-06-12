from configs.my_eval.my_fin_eval import datasets
from opencompass.utils.build import build_dataset_from_cfg

for cfg in datasets:
    print(f"\n🔍 Testing dataset: {cfg['abbr']}")

    # 取出传给你类的真正部分（避免 abbr、reader_cfg 等干扰）
    dataset_cfg = dict(
        type=cfg['type'],  # 传入类或类名
        path=cfg['path']   # 你只定义了支持 path
    )

    dataset = build_dataset_from_cfg(dataset_cfg)

    print("✅ Built:", type(dataset))
    print("🔹 First sample:", dataset[0])