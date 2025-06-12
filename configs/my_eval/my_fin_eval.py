from opencompass.datasets.myfindataset import MyFinDataset
from opencompass.models import HuggingFaceCausalLM

datasets = [
    dict(
        type=MyFinDataset,
        abbr='finqa10k_modified',
        path='MyData/itzme091_financial-qa-10K-modified_alpaca.jsonl',
        reader_cfg=dict(
            input_columns=['instruction', 'input'],
            output_column='output',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',  # 改成基础支持类型
                template='### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
            ),
            retriever=dict(type='zero-shot'),
        ),
        evaluator=dict(type='LMEvaluator'),
    ),
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
            retriever=dict(type='zero-shot'),
        ),
        evaluator=dict(type='LMEvaluator'),
    ),
    dict(
        type=MyFinDataset,
        abbr='finqa10k_virattt',
        path='MyData/virattt_financial_qa_10K_alpaca.jsonl',
        reader_cfg=dict(
            input_columns=['instruction', 'input'],
            output_column='output',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template='### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
            ),
            retriever=dict(type='zero-shot'),
        ),
        evaluator=dict(type='LMEvaluator'),
    )
]

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