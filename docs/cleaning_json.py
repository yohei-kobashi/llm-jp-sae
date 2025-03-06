import json
import os

ckpts = [988240, 100000, 10000, 1000, 100, 10]

for ckpt in ckpts:
    print(f"Processing {ckpt}")
    dir_path = f"data/{ckpt}"
    for idx in range(100):
        if os.path.exists(f"{dir_path}/{idx}.json"):
            with open(f"{dir_path}/{idx}.json", "r") as f:
                data = json.load(f)

            data = {
                "token_act": data["token_act"],
                "language": data["language"],
                "granularity": data["granularity"],
            }

            with open(f"{dir_path}/{idx}.json", "w") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
