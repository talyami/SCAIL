import os

from .arguments import get_args, update_args_with_file
from .tokenization import get_tokenizer
from .model import AutoModel

if os.getenv("SCAIL_SKIP_DEEPSPEED") == "1":
    training_main = None
else:
    try:
        from .training.deepspeed_training import training_main
    except Exception as e:
        message = str(e).lower()
        if "deepspeed" in message or "nvcc" in message:
            from sat.helpers import print_rank0

            print_rank0(
                "DeepSpeed unavailable; training_main disabled.",
                level="WARNING",
            )
            training_main = None
        else:
            raise e
__version__ = '0.4.11'
