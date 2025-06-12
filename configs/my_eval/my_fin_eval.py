from opencompass.datasets.myfindataset import MyFinDataset
from opencompass.models import HuggingFaceCausalLM
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner

# -------------------------------
# ✅ 评估配置（共用）
# -------------------------------
default_eval_cfg = dict(
    evaluator=dict(type='ROUGEEvaluator'),      # 使用语言模型输出评估方式
    pred_role='BOT',                         # 预测角色设置为 BOT（默认）
    pred_postprocessor=dict(type='strip')    # 对模型输出进行简单处理
)

# -------------------------------
# ✅ 数据集
# -------------------------------
datasets = [
    dict(
        type=MyFinDataset,
        abbr='finqa1k_nihar',
        path='MyData/NiharS_financial_qa_1K_alpaca.jsonl',
        reader_cfg=dict(
            input_columns=['instruction', 'input'],
            output_column='output',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template='### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
            ),
        ),
        eval_cfg=default_eval_cfg
    ),
    # dict(
    #     type=MyFinDataset,
    #     abbr='finqa10k_modified',
    #     path='MyData/itzme091_financial-qa-10K-modified_alpaca.jsonl',
    #     reader_cfg=dict(
    #         input_columns=['instruction', 'input'],
    #         output_column='output',
    #     ),
    #     infer_cfg=dict(
    #         prompt_template=dict(
    #             type='PromptTemplate',
    #             template='### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
    #         ),
    #     ),
    #     eval_cfg=default_eval_cfg
    # ),
    # dict(
    #     type=MyFinDataset,
    #     abbr='finqa10k_virattt',
    #     path='MyData/virattt_financial_qa_10K_alpaca.jsonl',
    #     reader_cfg=dict(
    #         input_columns=['instruction', 'input'],
    #         output_column='output',
    #     ),
    #     infer_cfg=dict(
    #         prompt_template=dict(
    #             type='PromptTemplate',
    #             template='### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
    #         ),
    #     ),
    #     eval_cfg=default_eval_cfg
    # )
]

# -------------------------------
# ✅ 模型
# -------------------------------
models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen2.5-1.5b',
        path='Qwen/Qwen2.5-1.5B',
        tokenizer_path='Qwen/Qwen2.5-1.5B',
        max_out_len=512,
        max_seq_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]

# -------------------------------
# ✅ 推理配置
# -------------------------------
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner),
    task=dict(type=OpenICLInferTask)
)

# -------------------------------
# ✅ 评估配置
# -------------------------------
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner),
    task=dict(type=OpenICLEvalTask)
)