import argparse
import json

from lm_eval import evaluator, tasks
from tasks import EvalHarnessAdaptor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("--output-file", type=str, default="input.jsonl")
    parser.add_argument("--task-name", type=str, default="hellaswag")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--num-data", type=int, default=None)
    args = parser.parse_args()

    seq = 1024
    total_batch = 1
    pe = "fixed"

    with open(args.output_file, "w") as f:
        pass
    
    # load prompt from eval, 
    class DryRunner:
        def eval(self, batch):
            with open(args.output_file, "a") as f:
                for text in batch["question"]:
                    item = {
                        "best_of": 1,
                        "echo": True,
                        "logprobs": 1,
                        "max_tokens": 0,
                        "model": "x",
                        "n": 1,
                        "prompt": text,
                        "request_type": "language-model-inference",
                        "stop": None,
                        "temperature": 0,
                        "top_p": 1,
                    }

                    f.write(json.dumps(item) + "\n")

            out = {
                "mask_loss": [1.0] * len(batch),
                "each_correct": [True] * len(batch),
            }
            return out

    t = DryRunner()

    adaptor = EvalHarnessAdaptor(t, seq, total_batch, shrink=pe != "fixed")
    #(jhkim/issue) task.build_requests() did not find any docs
    results = evaluator.evaluate(
        # LM : 
        lm=adaptor,
        # task_dict : 
        task_dict=tasks.get_task_dict(
            [
                args.task_name,
                # "lambada_openai",
                # "piqa",
                # "hellaswag",
                # "winogrande",
                # "mathqa",
                # "pubmedqa",
                # "boolq",
                # "cb",
                # "copa",
                # "multirc",
                # "record",
                # "wic",
                # "wsc",
            ]
        ),
        # provide_description : 
        # provide_description=False,
        # num_fewshot : =>deleted in 0.4.0!
        # args.num_fewshot,
        # limit :
        limit=args.num_data,
    )
    # dumped = json.dumps(results, indent=2)
    # print(dumped)
