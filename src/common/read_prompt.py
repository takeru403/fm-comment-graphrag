from pathlib import Path

import yaml


def read_prompt(prompt_path: Path) -> str:
    with open(prompt_path, "r") as f:
        prompt = yaml.safe_load(f)
    return prompt


if __name__ == "__main__":
    prompt_path = Path(__file__).parent / "prompt.yml"
    print(read_prompt(prompt_path))
