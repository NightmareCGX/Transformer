from pathlib import Path

import yaml


def main():
    cfg_path = Path("configs/base.yaml")
    cfg = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    print("[train] config:", cfg)
    print("[train] need to be implemented.")


if __name__ == "__main__":
    main()
