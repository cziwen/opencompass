# configs/my_eval/qwen2p5_1p5b_fin_eval.py
from mmengine.config import read_base

with read_base():  # 让 OpenCompass 自动补 import
    # ↓↓↓ 只要在 with read_base 里 import，就能让后面 dict 里写字符串类型
    from opencompass.datasets.myfindataset import MyFinDataset          # 你的数据集
    from opencompass.openicl.icl_retriever import ZeroRetriever         # 简单 0-shot 检索器
    from opencompass.openicl.icl_inferencer import GenInferencer        # 通用推理器
    from opencompass.openicl.icl_prompt_template import PromptTemplate  # Prompt 模板
    from opencompass.openicl.icl_evaluator import RougeEvaluator        # ROUGE 评测

# ---------------------------------------------------------------------
# 1. 评测公共配置
# ---------------------------------------------------------------------
default_eval_cfg = dict(
    evaluator=dict(type=RougeEvaluator),   # 使用 ROUGE 评价
    pred_role='BOT',
    pred_postprocessor=dict(type='strip')
)

# ---------------------------------------------------------------------
# 2. 数据集
# ---------------------------------------------------------------------
datasets = [
    dict(
        type=MyFinDataset,
        abbr='finqa1k_nihar',
        path='MyData/NiharS_financial_qa_1K_alpaca.jsonl',
        reader_cfg=dict(
            input_columns=['instruction', 'input'],
            output_column='output',
        ),
        infer_cfg=dict(                         # ★ 推理配置必须含 retriever 与 inferencer
            prompt_template=dict(
                type=PromptTemplate,
                template='''### Instruction:
{instruction}

### Input:
{input}

### Response:'''),
            retriever=dict(type=ZeroRetriever),   # 零样本直接拿每条样本用 prompt
            inferencer=dict(type=GenInferencer)   # 默认 greedy / temperature 采样
        ),
        eval_cfg=default_eval_cfg
    ),
]

# ---------------------------------------------------------------------
# 3. 模型
# ---------------------------------------------------------------------
models = [
    dict(
        type='HuggingFaceCausalLM',          # 直接写字符串即可
        abbr='qwen2.5-1.5b',
        path='Qwen/Qwen2.5-1.5B',            # 🤗 模型仓库
        tokenizer_path='Qwen/Qwen2.5-1.5B',
        max_out_len=512,
        max_seq_len=8192,                    # Qwen2.5-1.5B 官方上下文 8K
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        generation_cfg=dict(                 # 可选：写一些采样参数
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        ),
    )
]

# ---------------------------------------------------------------------
# 4. 推理与评测 Runner
# ---------------------------------------------------------------------
infer = dict(   # 单机单 GPU 演示，按需改 num_worker
    partitioner=dict(type='NumWorkerPartitioner', num_worker=1),
    runner=dict(type='LocalRunner', task=dict(type='OpenICLInferTask')),
)

eval = dict(    # ROUGE 纯 CPU 就行
    partitioner=dict(type='NaivePartitioner', n=1),
    runner=dict(type='LocalRunner', task=dict(type='OpenICLEvalTask')),
)

# ---------------------------------------------------------------------
# 5. 结果输出目录
# ---------------------------------------------------------------------
work_dir = 'outputs/qwen2p5_1p5b_fin_eval'