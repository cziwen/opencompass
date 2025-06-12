# configs/my_eval/qwen2p5_1p5b_base_eval.py
# ================================================================
from opencompass.datasets import JsonlDataset

# ---------------- 1. 公共评测配置 ----------------
default_eval_cfg = dict(
    evaluator=dict(type='RougeEvaluator'),
    pred_role='BOT',
    pred_postprocessor=dict(type='strip'),
)

# ---------------- 2. 数据集 ----------------
datasets = [
    dict(
        type=JsonlDataset,                     # ← 内置通用 JSONL 数据集
        abbr='finqa1k_nihar',
        path='MyData/NiharS_financial_qa_1K_alpaca.jsonl',
        reader_cfg=dict(                         # 指定列名即可
            input_columns=['instruction', 'input'],
            output_column='output',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template=(
                    '### Instruction:\n{instruction}\n\n'
                    '### Input:\n{input}\n\n'
                    '### Response:'
                ),
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(
                type='GenInferencer',
                max_out_len=512,
                temperature=0.7,
                top_p=0.95,
                batch_size=1,
            ),
        ),
        eval_cfg=default_eval_cfg,
    ),
]

# ---------------- 3. 模型 ----------------
models = [
    dict(
        type='HuggingFaceCausalLM',
        abbr='qwen2.5-1.5b',
        path='Qwen/Qwen2.5-1.5B',
        tokenizer_path='Qwen/Qwen2.5-1.5B',
        max_out_len=512,
        max_seq_len=8192,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        # 生成参数已放到 inferencer，不需要 gen_config
    ),
]

# ---------------- 4. 推理 / 评测 Runner ----------------
infer = dict(
    partitioner=dict(type='NumWorkerPartitioner', num_worker=1),
    runner=dict(type='LocalRunner', task=dict(type='OpenICLInferTask')),
)

eval = dict(
    partitioner=dict(type='NaivePartitioner', n=4),   # 4 进程并行评测
    runner=dict(type='LocalRunner', task=dict(type='OpenICLEvalTask')),
)

# ---------------- 5. 输出目录 ----------------
work_dir = 'outputs/qwen2p5_1p5b_base_eval'