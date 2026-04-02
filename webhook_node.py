"""
WebhookCallback node - POSTs workflow results to a callback URL.
Supports images, text, video output.
"""
import json
import os
import base64
import shutil
import urllib.request
import urllib.error
import torch
import folder_paths


def _empty_image():
    # Comfy IMAGE shape convention: [B, H, W, C], float32 in [0,1]
    return torch.zeros((1, 1, 1, 3), dtype=torch.float32)


def _build_filename(base, ext, index=None, total=1):
    if total <= 1 or index is None:
        return f"{base}{ext}"
    return f"{base}_{index:04d}{ext}"


def _to_items(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class BAInputSlot:
    CATEGORY = "api-bridge"
    FUNCTION = "execute"
    RETURN_TYPES = ()
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variable_name": ("STRING", {"default": "input_1", "multiline": False}),
                "input_type": (["text", "image", "video"], {"default": "text"}),
            },
            "optional": {
                "text_in": ("STRING", {"forceInput": True}),
                "image_in": ("IMAGE",),
                "video_in": ("VIDEO", {"forceInput": True}),
            },
        }

    def execute(self, variable_name, input_type, text_in=None, image_in=None, video_in=None):
        # Definition node only: used for variable declaration and binding via incoming links.
        return {}


class OpenClawOutput:
    """
    Output marker node for OpenClaw polling.
    This node acts as a passthrough OUTPUT_NODE - it passes through
    the connected image/video/text and marks them as final outputs.

    ComfyUI's history API records outputs from OUTPUT_NODEs.
    By connecting your final outputs to this node, OpenClaw can filter
    history results to only this node's outputs.

    Usage: Connect any combination of images/video/text to this node.
    All connected outputs will be recorded and returned by OpenClaw.
    """
    CATEGORY = "api-bridge"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "VIDEO", "STRING")
    RETURN_NAMES = ("images", "videos", "text")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "images": ("IMAGE",),
                "video": ("VIDEO",),
                "text": ("STRING", {"forceInput": True}),
            },
        }

    def execute(self, images=None, video=None, text=None):
        # Passthrough: return inputs as-is.
        # ComfyUI records these in /history/{prompt_id} under this node's ID.
        # OpenClaw filters by class_type="OpenClawOutput" to find this node.
        img = images if images is not None else _empty_image()
        vid = video if video is not None else ""
        txt = text if text is not None else ""
        return (img, vid, txt)


class WebhookCallback:
    CATEGORY = "api-bridge"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "webhook_url": ("STRING", {"default": "", "multiline": False}),
                "task_id": ("STRING", {"default": "", "multiline": False}),
                "output_basename": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "images": ("IMAGE",),
                "text": ("STRING", {"forceInput": True}),
                "video": ("VIDEO", {"forceInput": True}),
            },
        }

    def execute(self, webhook_url, task_id, output_basename="", images=None, text=None, video=None):
        if not webhook_url:
            print("[WebhookCallback] No webhook_url, skipping")
            return {}

        payload = {"task_id": task_id, "status": "completed", "outputs": {}}
        base = output_basename.strip() if isinstance(output_basename, str) else ""
        if not base:
            base = f"webhook_{task_id}"

        if images is not None:
            import numpy as np
            from PIL import Image
            import io
            results = []
            output_dir = folder_paths.get_output_directory()
            total_images = len(images)
            for i, image in enumerate(images):
                arr = (image.cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(arr)
                fname = _build_filename(base, ".png", i, total_images)
                pil_img.save(os.path.join(output_dir, fname))
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                results.append({"filename": fname, "type": "image/png",
                                "base64": base64.b64encode(buf.getvalue()).decode()})
            payload["outputs"]["images"] = results

        text_items = _to_items(text)
        if text_items:
            output_dir = folder_paths.get_output_directory()
            text_results = []
            total_text = len(text_items)
            for i, text_item in enumerate(text_items):
                fname = _build_filename(base, ".txt", i, total_text)
                fpath = os.path.join(output_dir, fname)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(text_item if isinstance(text_item, str) else str(text_item))
                text_results.append({"filename": fname, "type": "text/plain"})
            payload["outputs"]["text"] = text_results

        video_items = [v for v in _to_items(video) if str(v).strip()]
        if video_items:
            try:
                output_dir = folder_paths.get_output_directory()
                video_results = []
                total_video = len(video_items)
                for i, video_item in enumerate(video_items):
                    src = str(video_item).strip()
                    if not os.path.isabs(src):
                        src_candidate = os.path.join(output_dir, src)
                        if os.path.exists(src_candidate):
                            src = src_candidate
                    if os.path.exists(src):
                        ext = os.path.splitext(src)[1] or ".mp4"
                        fname = _build_filename(base, ext, i, total_video)
                        dst = os.path.join(output_dir, fname)
                        if os.path.abspath(src) != os.path.abspath(dst):
                            shutil.copyfile(src, dst)
                        with open(dst, "rb") as f:
                            video_results.append({
                                "filename": fname,
                                "type": "video/mp4",
                                "base64": base64.b64encode(f.read()).decode(),
                            })
                    else:
                        print(f"[WebhookCallback] Video path not found: {src}")
                if video_results:
                    payload["outputs"]["videos"] = video_results
            except Exception as e:
                print(f"[WebhookCallback] Video error: {e}")

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(webhook_url, data=data,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                print(f"[WebhookCallback] Sent -> {resp.status}")
        except Exception as e:
            print(f"[WebhookCallback] Failed: {e}")

        return {}


NODE_CLASS_MAPPINGS = {
    "WebhookCallback": WebhookCallback,
    "BAInputSlot": BAInputSlot,
    "OpenClawOutput": OpenClawOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WebhookCallback": "Webhook Callback (API Bridge)",
    "BAInputSlot": "BA Input Slot (Variable)",
    "OpenClawOutput": "OpenClaw Output (Marker)",
}
