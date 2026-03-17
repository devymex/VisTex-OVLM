"""
VisTex-OVLM Inference Script

Given:
  - A REF patch image containing the target object
  - A text description of that object
  - A query image (IMG) potentially containing similar objects

Output:
  - Bounding boxes of all objects in IMG similar to the REF object
"""
import os
import sys
import copy
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer


# ---------------------------------------------------------------------------
# Image preprocessing (matching the model's training pipeline)
# ---------------------------------------------------------------------------

def load_and_preprocess_image(image_path, min_size=800, max_size=800):
    """Load image and preprocess for model input.

    Pipeline: load RGB -> resize -> to tensor [0,1] -> BGR reorder -> scale to 255 -> normalize
    This matches TO_BGR255=True with PIXEL_MEAN/STD in BGR 0-255 space.

    Returns:
        tensor: [3, H, W] preprocessed image tensor
        orig_size: (W, H) of the original image
        new_size: (H, W) after resize
    """
    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    # Resize: keep aspect ratio, short side = min_size, long side <= max_size
    scale = min_size / min(orig_w, orig_h)
    if max(orig_w, orig_h) * scale > max_size:
        scale = max_size / max(orig_w, orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)

    # To tensor [0,1]
    img_tensor = TF.to_tensor(pil_img)  # [3, H, W] RGB in [0, 1]

    # BGR reorder + scale to 255
    img_tensor = img_tensor[[2, 1, 0]] * 255.0

    # Normalize with pixel mean/std (BGR, 0-255 space)
    pixel_mean = torch.tensor([103.530, 116.280, 123.675]).view(3, 1, 1)
    pixel_std = torch.tensor([57.375, 57.120, 58.395]).view(3, 1, 1)
    img_tensor = (img_tensor - pixel_mean) / pixel_std

    return img_tensor, (orig_w, orig_h), (new_h, new_w)


def preprocess_ref_patch(image_path, target_h=800, target_w=800):
    """Load the REF patch and preprocess it like the model's img_preprocess does.

    The model's internal pipeline:
    1. Takes the already-preprocessed (BGR-norm) image tensor
    2. Creates a binary mask for the box region
    3. Applies blur + bg_fac attenuation outside the mask
    4. Feeds the result tensor to the backbone

    For a standalone REF patch, the entire image IS the object, so the mask
    covers the full image. We resize to target_h x target_w to match what the
    backbone expects.

    Returns:
        tensor: [1, 3, target_h, target_w] ready for backbone
    """
    pil_img = Image.open(image_path).convert("RGB")
    pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)

    img_tensor = TF.to_tensor(pil_img)  # [3, H, W] RGB [0, 1]
    img_tensor = img_tensor[[2, 1, 0]] * 255.0  # BGR [0, 255]

    pixel_mean = torch.tensor([103.530, 116.280, 123.675]).view(3, 1, 1)
    pixel_std = torch.tensor([57.375, 57.120, 58.395]).view(3, 1, 1)
    img_tensor = (img_tensor - pixel_mean) / pixel_std

    # Since the REF patch is entirely the object, the mask is all 1s
    # (no background blurring needed). This is equivalent to img_preprocess
    # with mask=1 everywhere.
    return img_tensor.unsqueeze(0).float()  # [1, 3, H, W]


# ---------------------------------------------------------------------------
# Caption / positive map construction
# ---------------------------------------------------------------------------

def build_caption_and_positive_map(label_name, tokenizer):
    """Build a caption string and positive_map_label_to_token dict.

    Caption format: "label_name. "
    The positive_map maps label_id -> list of token indices.
    """
    caption = label_name + ". "

    tokenized = tokenizer(caption, return_tensors="pt")
    input_ids = tokenized["input_ids"][0]  # [seq_len]

    # Find token positions for the label (skip [CLS]=101, find up to separator '.'=1012)
    # Token 101 = [CLS], Token 102 = [SEP], Token 1012 = '.'
    label_token_positions = []
    for idx in range(1, len(input_ids)):  # skip [CLS]
        tid = input_ids[idx].item()
        if tid == 1012 or tid == 102:  # '.' or [SEP]
            break
        label_token_positions.append(idx)

    # Label ID starts from 1 (standard in this codebase)
    positive_map_label_to_token = {1: label_token_positions}
    return caption, positive_map_label_to_token


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_results(image_path, predictions, output_path, score_threshold=0.3):
    """Draw bounding boxes on the image and save."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig_h, orig_w = img.shape[:2]

    boxes = predictions.bbox.cpu().numpy()        # [N, 4] in resized image coords
    scores = predictions.get_field("scores").cpu().numpy()
    labels = predictions.get_field("labels").cpu().numpy()

    # predictions are in model input space; scale back to original image space
    pred_w, pred_h = predictions.size  # (width, height) of the prediction space
    scale_x = orig_w / pred_w
    scale_y = orig_h / pred_h

    count = 0
    for i in range(len(boxes)):
        if scores[i] < score_threshold:
            continue
        x1, y1, x2, y2 = boxes[i]
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{scores[i]:.2f}"
        cv2.putText(img, text, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        count += 1

    cv2.imwrite(output_path, img)
    print(f"Saved {count} detections (threshold={score_threshold}) to {output_path}")


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VisTex-OVLM Image-Prompted Inference")
    parser.add_argument("--ref", type=str, required=True,
                        help="Path to reference patch image (object-only)")
    parser.add_argument("--img", type=str, required=True,
                        help="Path to query image to detect similar objects")
    parser.add_argument("--text", type=str, required=True,
                        help="Text description of the object (e.g., 'cat')")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: assets/infer_results/<img_name>)")
    parser.add_argument("--config", type=str,
                        default="configs/pretrain/glip_Swin_L.yaml",
                        help="Model config yaml")
    parser.add_argument("--weight", type=str,
                        default="/gemini/data-1/model/vistex/model_best.pth",
                        help="Path to model weights")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Score threshold for visualization")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    # --- Config ---
    cfg.merge_from_file(os.path.join(PROJECT_ROOT, args.config))

    # Override for inference
    cfg.IMPROMPT.gvl = -1
    cfg.IMPROMPT.input_way = "input_image_itself"
    cfg.IMPROMPT.save_oneshot_imprompts = True
    cfg.IMPROMPT.shot_num = 1
    cfg.IMPROMPT.shot_fusion = "max"
    cfg.IMPROMPT.stage_fusion = "max"
    cfg.MODEL.DEVICE = args.device
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 800

    # Use the saved training config to ensure compatibility
    saved_config = os.path.join(os.path.dirname(args.weight), "config.yml")
    if os.path.exists(saved_config):
        from yacs.config import CfgNode as CN
        import yaml
        with open(saved_config, "r") as f:
            saved_cfg_dict = yaml.safe_load(f)
        # Only keep keys that exist in the default cfg
        known_keys = set(cfg.keys())
        cleaned = {k: v for k, v in saved_cfg_dict.items() if k in known_keys}
        cfg.defrost()
        cfg.merge_from_other_cfg(CN(cleaned))
        # Re-apply our inference overrides
        cfg.IMPROMPT.gvl = -1
        cfg.IMPROMPT.input_way = "input_image_itself"
        cfg.IMPROMPT.save_oneshot_imprompts = True
        cfg.IMPROMPT.shot_num = 1
        cfg.MODEL.DEVICE = args.device

    cfg.freeze()

    device = torch.device(cfg.MODEL.DEVICE)

    # --- Build and load model ---
    print("Building model...")
    model = build_detection_model(cfg)
    model.to(device)

    print(f"Loading weights from {args.weight}")
    checkpointer = DetectronCheckpointer(cfg, model, save_dir="")
    checkpointer.load(args.weight, force=True)
    model.eval()

    # --- Prepare REF patch ---
    print(f"Loading reference patch: {args.ref}")
    ref_tensor = preprocess_ref_patch(args.ref, target_h=800, target_w=800)
    # ref_tensor: [1, 3, 800, 800]

    # Inject pre-computed reference into model's history cache
    # Label 1 is used as the reference label
    ref_label = 1
    model.history_reference_image_tensor[ref_label] = ref_tensor.cpu()

    # --- Prepare query image ---
    print(f"Loading query image: {args.img}")
    img_tensor, orig_size, new_size = load_and_preprocess_image(
        args.img,
        min_size=cfg.INPUT.MIN_SIZE_TEST,
        max_size=cfg.INPUT.MAX_SIZE_TEST
    )
    # img_tensor: [3, H, W]
    images = to_image_list([img_tensor], cfg.DATALOADER.SIZE_DIVISIBILITY)
    images = images.to(device)

    # --- Prepare caption and positive_map ---
    caption, positive_map = build_caption_and_positive_map(args.text, model.tokenizer)
    captions = [caption]
    print(f"Caption: {caption!r}")
    print(f"Positive map: {positive_map}")

    # --- Build a dummy reference_map (BoxList) ---
    # The model needs reference_map[0].bbox and reference_map[0].extra_fields['labels']
    # to know which label's embedding to inject. Since we pre-injected the ref tensor
    # into history, the model will find it there and skip re-processing.
    # We create a dummy box that won't be meaningfully used (model uses cached tensor).
    new_h, new_w = new_size
    dummy_box = torch.tensor([[0, 0, float(new_w), float(new_h)]])  # cover entire image
    dummy_boxlist = BoxList(dummy_box, (new_w, new_h), mode="xyxy")
    dummy_boxlist.add_field("labels", torch.tensor([ref_label]))
    reference_map = [dummy_boxlist.to(device)]

    # --- Run inference ---
    print("Running inference...")
    with torch.no_grad():
        output = model(
            images,
            captions=captions,
            positive_map=positive_map,
            reference_map=reference_map,
        )

    predictions = output[0].to(torch.device("cpu"))
    predictions.size = (new_w, new_h)

    n_total = len(predictions.bbox)
    n_above = (predictions.get_field("scores") >= args.threshold).sum().item()
    print(f"Total predictions: {n_total}, above threshold {args.threshold}: {n_above}")

    # --- Visualize ---
    output_path = args.output
    if output_path is None:
        os.makedirs(os.path.join(PROJECT_ROOT, "assets", "infer_results"), exist_ok=True)
        img_basename = os.path.splitext(os.path.basename(args.img))[0]
        output_path = os.path.join(PROJECT_ROOT, "assets", "infer_results",
                                   f"{img_basename}_result.jpg")

    visualize_results(args.img, predictions, output_path,
                      score_threshold=args.threshold)

    print("Done!")


if __name__ == "__main__":
    main()
