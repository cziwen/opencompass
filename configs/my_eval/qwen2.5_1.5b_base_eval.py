# configs/my_eval/qwen2.5_1.5b_base_eval.py




# ---------------------------------------------------------------------
# 1. 公共评测配置
# ---------------------------------------------------------------------
default_eval_cfg = dict(
    evaluator=dict(type='RougeEvaluator'),   # 用字符串，不手动 import
    pred_role='BOT',
    pred_postprocessor=dict(type='strip')
)

# ---------------------------------------------------------------------
# 2. 数据集
# ---------------------------------------------------------------------
datasets = [
    dict(
        # 这里写注册名即可，或用完整路径
        # 如果你在 myfindataset.py 里  @LOAD_DATASET.register_module(name='MyFinDataset')
        # 就写 'MyFinDataset'
        type='MyFinDataset',
        abbr='finqa1k_nihar',
        path='MyData/NiharS_financial_qa_1K_alpaca.jsonl',
        reader_cfg=dict(
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
                )
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(type='GenInferencer')
        ),
        eval_cfg=default_eval_cfg,
    ),
]

# ---------------------------------------------------------------------
# 3. 模型
# ---------------------------------------------------------------------
models = [
    dict(
        type='HuggingFaceCausalLM',
        abbr='qwen2.5-1.5b',
        path='Qwen/Qwen2.5-1.5B',
        tokenizer_path='Qwen/Qwen2.5-1.5B',
        max_out_len=512,
        max_seq_len=8192,        # 官方 8K
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
]

# ---------------------------------------------------------------------
# 4. 推理 / 评测 Runner
# ---------------------------------------------------------------------
infer = dict(
    partitioner=dict(type='NumWorkerPartitioner', num_worker=1),
    runner=dict(type='LocalRunner', task=dict(type='OpenICLInferTask')),
)

eval = dict(
    partitioner=dict(type='NaivePartitioner', n=4),
    runner=dict(type='LocalRunner', task=dict(type='OpenICLEvalTask')),
)

# ---------------------------------------------------------------------
# 5. 输出目录
# ---------------------------------------------------------------------
work_dir = 'outputs/qwen2p5_1p5b_base_eval'