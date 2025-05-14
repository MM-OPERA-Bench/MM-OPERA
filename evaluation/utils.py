# MM-OPERA/evaluation/utils.py

import json
import logging
import base64
import io
from pathlib import Path
from PIL import (
    Image,
    UnidentifiedImageError,
)  # Assuming PIL.Image objects from Hugging Face dataset
import tqdm

# Dynamically determine project root assuming utils.py is in MM-OPERA/evaluation/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logger(name: str, log_file: Path, level=logging.INFO) -> logging.Logger:
    """
    Sets up a logger that outputs to both a file and the console.
    """
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if already configured
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def save_json(
    data: dict | list, file_path: Path, indent: int = 4, ensure_ascii: bool = False
):
    """
    Saves data to a JSON file.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    except IOError as e:
        # It's better to log this with a logger if available
        print(f"Error saving JSON to {file_path}: {e}")
        # Or raise e if the caller should handle it


def load_json(file_path: Path) -> dict | list | None:
    """
    Loads data from a JSON file. Returns None if file not found or error.
    """
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return None  # Or an empty dict/list depending on expected return


def encode_pil_image_to_base64(
    pil_image: Image.Image,
) -> tuple[str, str] | tuple[None, None]:
    """
    Encodes a PIL Image object to a base64 string and determines its MIME type.

    Args:
        pil_image (Image.Image): The PIL Image object.

    Returns:
        tuple[str, str] | tuple[None, None]: (base64_string, mime_type) or (None, None) on error.
                                             MIME type will be 'image/jpeg' or 'image/png'.
                                             Defaults to 'image/png' if format is unknown but encodable.
    """
    if not pil_image:
        return None, None
    try:
        buffered = io.BytesIO()
        img_format = pil_image.format

        if img_format == "JPEG":
            mime_type = "image/jpeg"
            # Ensure image is RGB before saving as JPEG
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            pil_image.save(buffered, format="JPEG")
        elif img_format == "PNG":
            mime_type = "image/png"
            pil_image.save(buffered, format="PNG")
        else:
            # Fallback for other types, try saving as PNG
            # print(f"Warning: Unknown image format '{img_format}'. Attempting to encode as PNG.")
            mime_type = "image/png"
            pil_image.save(buffered, format="PNG")

        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return encoded_string, mime_type
    except Exception as e:
        print(f"Error encoding PIL image: {e}")
        return None, None


# In your utils.py or equivalent
from tqdm import tqdm
from PIL import (
    UnidentifiedImageError,
)  # Ensure this (or relevant PIL error) is imported if type checking

# It's good practice to also import 'logging' if you type hint the logger
import logging


def safe_dataset_iterator_with_error_info(
    dataset_split, logger: logging.Logger, desc="Iterating safely"
):
    """
    Iterates over a Hugging Face dataset split, skipping items that cause specific errors.
    Yields (original_index, item) for successful items.
    Logs skipped items using the provided logger.
    Returns a list of skipped items' information.
    """
    total_len = len(dataset_split)
    skipped_items_log = []  # To store info about skipped items

    # position=1 and leave=False are good for nested tqdm bars if the main loop also has one
    # mininterval can prevent overly frequent updates
    pbar_iterator = tqdm(
        range(total_len),
        desc=desc,
        total=total_len,
        leave=False,
        position=1,
        mininterval=0.5,
    )

    for idx in pbar_iterator:
        # Update description for the inner progress bar
        pbar_iterator.set_description_str(f"{desc} (Checking idx {idx})")
        try:
            item = dataset_split[idx]
            yield idx, item  # Yield original index and the item
        except (
            SyntaxError,
            UnidentifiedImageError,
        ) as e:  # Add other PIL specific errors if necessary
            error_message = str(e)
            error_type_name = type(e).__name__
            if (
                "not a TIFF file" in error_message
                or ("header" in error_message and "not valid" in error_message)
                or "cannot identify image file" in error_message
            ):
                logger.warning(  # Use logger.warning or logger.error as appropriate
                    f"[Safe Iterator] Skipping item at original index {idx} due to known image error: {error_type_name} - {error_message}"
                )
                skipped_items_log.append(
                    {"index": idx, "reason": error_message, "type": error_type_name}
                )
                continue
            else:  # Other SyntaxError/UnidentifiedImageError
                logger.error(  # Potentially more severe if it's an unexpected variant
                    f"[Safe Iterator] Skipping item at original index {idx} due to other {error_type_name}: {error_message}"
                )
                skipped_items_log.append(
                    {
                        "index": idx,
                        "reason": error_message,
                        "type": f"Other {error_type_name}",
                    }
                )
                continue
        except Exception as e:  # Other generic errors
            error_message = str(e)
            error_type_name = type(e).__name__
            logger.error(
                f"[Safe Iterator] Skipping item at original index {idx} due to generic error: {error_type_name} - {error_message}"
            )
            skipped_items_log.append(
                {
                    "index": idx,
                    "reason": error_message,
                    "type": f"Generic {error_type_name}",
                }
            )
            continue

    if skipped_items_log:
        logger.info(
            f"[Safe Iterator] Completed. Skipped a total of {len(skipped_items_log)} items during iteration for '{desc}'."
        )
    else:
        logger.info(
            f"[Safe Iterator] Completed. No items were skipped during iteration for '{desc}'."
        )

    return skipped_items_log


if __name__ == "__main__":
    # Example usage:
    # Create a dummy log file path for testing
    test_log_path = PROJECT_ROOT / "logs" / "test_utils.log"
    logger = setup_logger("UtilsTest", test_log_path)
    logger.info("Utils logger setup complete.")

    test_data = {"key": "value", "numbers": [1, 2, 3]}
    test_json_path = PROJECT_ROOT / "temp_results" / "test.json"  # temp dir for test
    save_json(test_data, test_json_path)
    logger.info(f"Test JSON saved to {test_json_path}")

    loaded_data = load_json(test_json_path)
    if loaded_data:
        logger.info(f"Loaded test JSON: {loaded_data}")

    # To test image encoding, you'd need a PIL.Image object
    # from PIL import Image
    # try:
    #     # Create a dummy image (requires Pillow to be installed)
    #     img = Image.new('RGB', (60, 30), color = 'red')
    #     img.format = 'PNG' # Manually set format for this dummy
    #     b64_str, mime = encode_pil_image_to_base64(img)
    #     if b64_str:
    #         logger.info(f"Dummy image encoded. Mime: {mime}, Base64 (first 20 chars): {b64_str[:20]}...")
    # except ImportError:
    #     logger.warning("Pillow not installed, skipping PIL image encoding test.")
    # except Exception as e:
    #     logger.error(f"Error in PIL image test: {e}")
