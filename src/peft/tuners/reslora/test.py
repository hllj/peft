from config import ResLoraConfig
from model import ResLoraModel

from transformers import AutoConfig, AutoModelForSequenceClassification

if __name__ == '__main__':
    # training_args
    cls_dropout = False

    # data_args
    task_name = 'cola'

    # model_args
    config_name = None
    num_labels = 1
    cache_dir = "./cache"
    model_revision = "main"
    use_auth_token = False
    lora_alpha = 16
    lora_r = 4
    adapter_type = 'houlsby'
    adapter_size = 64
    reg_loss_wgt = 0.0
    masking_prob = 0.0
    method = "reslora"

    model_name_or_path = "FacebookAI/roberta-large"

    target_modules = "q.v"
    lora_dropout = 0.1
    res_flag = 3
    merge_flag = 3
    pre_num = 4

    config = AutoConfig.from_pretrained(
        config_name if config_name else model_name_or_path,
        num_labels=num_labels,
        finetuning_task=task_name,
        cache_dir=cache_dir,
        revision=model_revision,
        use_auth_token=True if use_auth_token else None,
        cls_dropout=cls_dropout,
        # apply_lora=apply_lora,
        lora_alpha=lora_alpha,
        lora_r=lora_r,
        # apply_adapter=apply_adapter,
        adapter_type=adapter_type,
        adapter_size=adapter_size,
        reg_loss_wgt=reg_loss_wgt,
        masking_prob=masking_prob,
        method=method
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=cache_dir,
        revision=model_revision,
        use_auth_token=True if use_auth_token else None,
    )

    if merge_flag:
        assert res_flag, "Error: merge flag must be used with res flag 1."
    if res_flag == 2 or res_flag == 3:
        assert pre_num != 0, "Error: pre num must be used when res flag 2."
    resconfig = ResLoraConfig(
        rank=lora_r, lora_alpha=lora_alpha,
        target_modules="q.v", lora_dropout=0.1, res_flag=res_flag,
        merge_flag=merge_flag, pre_num=pre_num
    )

    resmodel = ResLoraModel(model, resconfig, epochs=1, model_name='roberta')

    print(resconfig.to_json_string())

    print(resmodel)