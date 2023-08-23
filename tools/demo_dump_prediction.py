# modify from demo.py
import argparse
import copy
from pathlib import Path
import pickle
from tqdm import tqdm
import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="cfgs/kitti_models/second.yaml",
        help="specify the config for demo",
    )
    # parser.add_argument(
    #     "--data_path",
    #     type=str,
    #     default="demo_data",
    #     help="specify the point cloud data file or directory",
    # )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="specify the pretrained model"
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".bin",
        help="specify the extension of your point cloud data file",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="specify the output tag",
    )
    parser.add_argument("--set_cfgs", type=str, nargs="+")

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info("-----------------Quick Demo of OpenPCDet-------------------------")

    output_base_dir = (
        Path(__file__).absolute().parents[1]
        / "output"
        / args.tag
    )
    output_base_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"output dir: {output_base_dir}")
    data_config = cfg.DATA_CONFIG
    logger.info(f"use data from {data_config.DATA_PATH}")
    disable_preprocessor_config = copy.copy(data_config)
    disable_preprocessor_config["DATA_PROCESSOR"] = []
    fullpc_set, _, _ = build_dataloader(
        dataset_cfg=disable_preprocessor_config,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=4,
        logger=logger,
        training=False,
    )

    demo_set, demo_loader, sampler = build_dataloader(
        dataset_cfg=data_config,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=4,
        logger=logger,
        training=False,
    )
    logger.info(f"Total number of samples: \t{len(demo_set)}")

    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_set
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(tqdm(demo_set)):
            fullpc_data_dict = fullpc_set[idx]
            fullpc_data_dict = fullpc_set.collate_batch([fullpc_data_dict])

            # logger.info(f"Visualized sample index: \t{idx + 1}/{len(demo_set)}")
            data_dict = demo_set.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            for in_batch_id, pred_dict in enumerate(pred_dicts):
                # print(pred_dicts[0].keys())
                frame_id = data_dict["frame_id"][in_batch_id]
                draw_infos = {
                    "frame_id": frame_id,
                    # "points": data_dict["points"].cpu().numpy(),  # downsampled
                    "points": fullpc_data_dict["points"],  # full
                    "gt_boxes": data_dict["gt_boxes"][in_batch_id].cpu().numpy(),
                    "boxes": pred_dict["pred_boxes"].cpu().numpy(),
                    "scores": pred_dict["pred_scores"].cpu().numpy(),
                    "labels": pred_dict["pred_labels"].cpu().numpy(),
                }
                if pred_dict.get("pred_als", None) is not None:
                    draw_infos.update(
                        {"als": pred_dict["pred_als"].cpu().numpy()}
                    )
                if pred_dict.get("pred_eps", None) is not None:
                    draw_infos.update(
                        {"eps": pred_dict["pred_eps"].cpu().numpy()}
                    )
                with open(output_base_dir / f"{str(frame_id)}.pkl", "wb") as f:
                    pickle.dump(draw_infos, f)

    logger.info("Demo done.")


if __name__ == "__main__":
    main()
