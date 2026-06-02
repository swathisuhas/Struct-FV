
from absl import app
from absl import flags
import distance
import embedding
import io_util
import numpy as np
import os


_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "Batch size for embedding generation.")
_MAX_COUNT = flags.DEFINE_integer("max_count", -1, "Maximum number of images to read from each directory.")
_REF_EMBED_FILE = flags.DEFINE_string(
    "ref_embed_file", None, "Path to the pre-computed embedding file for the reference images."
)

def list_images_recursive(directory, exts=("png","jpg","jpeg","bmp")):
    image_paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(exts):
                image_paths.append(os.path.join(root, f))
    return sorted(image_paths)

def compute_cmmd(ref_dir, eval_dir, ref_embed_file=None, batch_size=32, max_count=-1):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    if ref_dir and ref_embed_file:
        raise ValueError("`ref_dir` and `ref_embed_file` both cannot be set at the same time.")
    embedding_model = embedding.ClipEmbeddingModel()
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_images = list_images_recursive(ref_dir)
        print("Computing reference embeddings...")
        ref_embs = io_util.compute_embeddings_for_dir(ref_images, embedding_model, batch_size, max_count).astype(
            "float32"
        )
        print(f"Reference embeddings shape: {ref_embs.shape}")

    eval_images = list_images_recursive(eval_dir)
    print("Computing evaluation embeddings...")
    eval_embs = io_util.compute_embeddings_for_dir(eval_images, embedding_model, batch_size, max_count).astype("float32")
    print(f"Evaluation embeddings shape: {eval_embs.shape}")
    val = distance.mmd(ref_embs, eval_embs)
    return val.numpy()


def main(argv):
    if len(argv) != 3:
        raise app.UsageError("Too few/too many command-line arguments.")
    _, dir1, dir2 = argv
    print(
        "The CMMD value is: "
        f" {compute_cmmd(dir1, dir2, _REF_EMBED_FILE.value, _BATCH_SIZE.value, _MAX_COUNT.value):.3f}"
    )


if __name__ == "__main__":
    app.run(main)