from config import MainConfiguration
from eval import get_output_dir
import pandas as pd
import glob

def load_results_dataframe(config_path: str) -> pd.DataFrame:
    
    configuration = MainConfiguration.parse_yaml(config_path)

    res = {
        "pvalue": [],
        "completion": [],
        "prompt": [],
        "modif_type": [],
        "full_modification": [],
        "ppl": [],
        "eval_type": []
    }

    merge_configs = configuration.merge_configs
    prune_configs = configuration.pruning_configs
    quantization_configs = configuration.quantization_configs
    finetuning_configs = configuration.finetuning_configs

    types = ["merge", "prune", "quantize", "finetuning"]
    finetuning_checkpoints = [500,1000,1500,2000,2500]

    configs = [merge_configs, prune_configs, quantization_configs, finetuning_configs] + [finetuning_configs for _ in range(len(finetuning_checkpoints))]


    types = types + [f"finetuning-{ckpt}" for ckpt in finetuning_checkpoints]

    for i, config in enumerate(configs):
        for c in config:
            output_dir = get_output_dir(configuration, c, types[i], "results")
            print(output_dir)
            try:
                files = glob.glob(f"{output_dir}/results_*.jsonl")
                for file in files:
                    eval_type = file.split("/")[-1].split("_")[1:]
                    eval_type = "_".join(eval_type)
                    eval_type = eval_type.split(".")[0]
                    df = pd.read_json(file, lines=True)
                    for _, row in df.iterrows():
                        res["pvalue"].append(row["pvalues"])
                        res["completion"].append(row["completions"])
                        res["prompt"].append(row["prompts"])
                        res["modif_type"].append(types[i])
                        res["full_modification"].append(c.short_str())
                        res["ppl"].append(row.get("ppls", None))
                        res["eval_type"].append(eval_type)
            except ValueError:
                print(f"Error with {output_dir}")
                
    #Get original results
    output_dir = get_output_dir(configuration, None, "original", "results")
    try:
        files = glob.glob(f"{output_dir}/results_*.jsonl")
        for file in files:
            df = pd.read_json(file, lines=True)
            eval_type = file.split("/")[-1].split("_")[1:]
            eval_type = "_".join(eval_type)
            eval_type = eval_type.split(".")[0]
            for _, row in df.iterrows():
                res["pvalue"].append(row["pvalues"])
                res["completion"].append(row["completions"])
                res["prompt"].append(row["prompts"])
                res["ppl"].append(row.get("ppls", None))
                res["modif_type"].append("original")
                res["full_modification"].append("original")
                res["eval_type"].append(eval_type)
    except ValueError:
        print(f"Error with {output_dir}")
                
                
    df = pd.DataFrame(res)
    
    return df