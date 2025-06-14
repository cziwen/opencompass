# configs/my_eval/qwen2p5_1p5b_base_eval.py
# ================================================================
from opencompass.datasets import JsonlDataset

# ---------------- 1. 公共评测配置 ----------------
default_eval_cfg = dict(
    evaluator=dict(type='RougeEvaluator', use_percent=True),
    pred_role='BOT',
)

# ---------------- 2. 数据集 ----------------
datasets = [
    # -------- 数据集 1：1K ----------
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
                batch_size=4,
            ),
        ),
        eval_cfg=default_eval_cfg,
    ),
    # -------- 数据集 2：10K-modified ----------
    # dict (
    #     type=JsonlDataset,
    #     abbr='finqa10k_modified',
    #     path='MyData/itzme091_financial-qa-10K-modified_alpaca.jsonl',
    #     reader_cfg=dict (
    #         input_columns=['instruction', 'input'],
    #         output_column='output',
    #     ),
    #     infer_cfg=dict (  # 与上面同一套配置即可
    #         prompt_template=dict (
    #             type='PromptTemplate',
    #             template=(
    #                 '### Instruction:\n{instruction}\n\n'
    #                 '### Input:\n{input}\n\n'
    #                 '### Response:'
    #             ),
    #         ),
    #
    #         retriever=dict (type='ZeroRetriever'),
    #         inferencer=dict (
    #             type='GenInferencer',
    #             max_out_len=512,
    #             temperature=0.7,
    #             top_p=0.95,
    #             batch_size=16,
    #         ),
    #     ),
    #     eval_cfg=default_eval_cfg,
    # ),
    # -------- 数据集 3：10K-virattt ----------
    # dict (
    #     type=JsonlDataset,
    #     abbr='finqa10k_virattt',
    #     path='MyData/virattt_financial_qa_10K_alpaca.jsonl',
    #     reader_cfg=dict (
    #         input_columns=['instruction', 'input'],
    #         output_column='output',
    #     ),
    #     infer_cfg=dict (
    #         prompt_template=dict (
    #             type='PromptTemplate',
    #             template=(
    #                 '### Instruction:\n{instruction}\n\n'
    #                 '### Input:\n{input}\n\n'
    #                 '### Response:'
    #             ),
    #         ),
    #         retriever=dict (type='ZeroRetriever'),
    #         inferencer=dict (
    #             type='GenInferencer',
    #             max_out_len=512,
    #             temperature=0.7,
    #             top_p=0.95,
    #             batch_size=4,
    #         ),
    #     ),
    #     eval_cfg=default_eval_cfg,
    # ),
]

# ---------------- 3. 模型 ----------------
models = [
    dict(
        type='HuggingFaceCausalLM',
        abbr='qwen2.5-1.5b',
        path='Qwen/Qwen2.5-1.5B',
        tokenizer_path='Qwen/Qwen2.5-1.5B',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        batch_padding=False,
        max_out_len=512,
        max_seq_len=8192,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
    ),
]
# models = [
#     dict(
#         type='HuggingFaceCausalLM',
#         abbr='qwen2.5-1.5b-fin',
#         path='Ziwen001/Qwen2.5-1.5B-Fin',
#         tokenizer_path='Ziwen001/Qwen2.5-1.5B-Fin',
#         tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
#         batch_padding=False,
#         max_out_len=512,
#         max_seq_len=8192,
#         batch_size=4,
#         run_cfg=dict(num_gpus=1),
#     ),
# ]

# ---------------- 4. 推理 / 评测 Runner ----------------
infer = dict(
    partitioner=dict(type='NumWorkerPartitioner', num_worker=1),
    runner=dict(type='LocalRunner', task=dict(type='OpenICLInferTask')),
)

eval = dict(
    partitioner=dict(type='NaivePartitioner', n=8),
    runner=dict(type='LocalRunner', task=dict(type='OpenICLEvalTask')),
)

# ---------------- 5. 输出目录 ----------------
work_dir = 'outputs/qwen2p5_1p5b_base_eval'