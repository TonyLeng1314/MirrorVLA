import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform, DummyDataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ======== 直接在这里改参数即可 ========
# 与你当前 LIBERO spatial 训练脚本保持一致
CONFIG_FILE_PATH = "pretrained_models/configs"
DATA_ROOT_DIR = Path("data/libero")
DATASET_NAME = "libero_spatial_no_noops"

BATCH_SIZE = 16
NUM_IMAGES_IN_INPUT = 1
USE_PROPRIO = True
USE_MINIVLM = True
MASK_CAM = "None"
HISTORY_FRAMES = 0
IMAGE_AUG = True
SHUFFLE_BUFFER_SIZE = 100_000
ENABLE_FOUR_QUADRANT_INPUT = True
# 如果只是想快速调试 pipeline，可以把这个设成 False，用 DummyDataset 生成伪数据，几乎不读磁盘
USE_REAL_RLDS_DATA = True
# =====================================


def debug_dataloader() -> None:
    """
    仅用于测试 / 打印 RLDS 数据读取结果，不做任何模型 / 训练相关操作。
    """
    config_file_path = CONFIG_FILE_PATH.rstrip("/")
    print(f"[DEBUG] use_real_rlds_data = {USE_REAL_RLDS_DATA}")
    print(f"[DEBUG] data_root_dir      = {DATA_ROOT_DIR}")
    print(f"[DEBUG] dataset_name       = {DATASET_NAME}")
    print(f"[DEBUG] config_path        = {config_file_path}")

    # 处理 HF Hub / 本地路径，与 finetune.py 逻辑保持一致（但不加载模型）
    if model_is_on_hf_hub(config_file_path):
        from huggingface_hub import snapshot_download

        vla_download_path = snapshot_download(repo_id=config_file_path)
        config_file_path = vla_download_path
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, None)

    update_auto_map(config_file_path)
    check_model_logic_mismatch(config_file_path)

    # 只加载 Processor（Tokenizer + ImageProcessor），不加载大模型
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(config_file_path, trust_remote_code=True)

    # 创建 action tokenizer 和 batch transform
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    use_wrist_image = NUM_IMAGES_IN_INPUT > 1
    if USE_REAL_RLDS_DATA:
        # 获取图像大小，用于 RLDSDataset 的 resize_resolution
        try:
            config = AutoConfig.from_pretrained(config_file_path, trust_remote_code=True)
            resize_resolution = tuple(config.image_sizes)
        except Exception:
            # 兜底：从 image_processor 里拿 size
            size = getattr(processor.image_processor, "size", None)
            if isinstance(size, dict) and "height" in size and "width" in size:
                resize_resolution = (size["height"], size["width"])
            elif isinstance(size, int):
                resize_resolution = (size, size)
            else:
                resize_resolution = (224, 224)

        print(f"[DEBUG] resize_resolution = {resize_resolution}")

        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            use_wrist_image=use_wrist_image,
            use_proprio=USE_PROPRIO,
            use_minivlm=USE_MINIVLM,
            mask_cam=MASK_CAM,
            history_frames=0,
            enable_four_quadrant_input=ENABLE_FOUR_QUADRANT_INPUT,
        )

        train_dataset = RLDSDataset(
            DATA_ROOT_DIR,
            DATASET_NAME,
            batch_transform,
            resize_resolution=resize_resolution,
            shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
            image_aug=IMAGE_AUG,
            history_frames=0,
        )

        print(f"[DEBUG] RLDS Dataset loaded. len(train_dataset) = {len(train_dataset)} (如果是 RLDS 流式，可能是估计值)")
    else:
        # 使用 DummyDataset：完全在内存里造数据，初始化极快，只用于检查 pipeline / 模型接口是否正常
        train_dataset = DummyDataset(
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
        )
        print(f"[DEBUG] Using DummyDataset. len(train_dataset) = {len(train_dataset)}")

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    print(f"[DEBUG] Dataloader constructed. len(dataloader) = {len(dataloader)}")

    # 取前几个 batch 打印结构
    max_batches_to_show = 2
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n========== Batch {batch_idx} ==========")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(
                    f"[{key}] tensor | shape={tuple(value.shape)}, "
                    f"dtype={value.dtype}, device={value.device}"
                )
            else:
                print(f"[{key}] type={type(value)}")

        if batch_idx + 1 >= max_batches_to_show:
            break

    print("\n[DEBUG] 数据读取与 collate 测试完成。")


if __name__ == "__main__":
    debug_dataloader()

