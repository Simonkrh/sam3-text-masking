# SAM 3 text masking

Mask objects in local images using text prompts.

Requires Linux and [uv](https://docs.astral.sh/uv/). Get access to
[facebook/sam3](https://huggingface.co/facebook/sam3), then:

```bash
uv sync --locked
uv run hf auth login
uv run python mask.py --image images/cola_8pack.jpg --prompt "pack of Coca-Cola bottles"
```
