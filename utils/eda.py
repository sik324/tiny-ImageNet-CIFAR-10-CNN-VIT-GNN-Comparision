
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# ── 1. Count images ──────────────────────────────────────────────────
def count_images(path, split_name, dataset_name, has_images_subfolder=False):
    total = 0
    classes = sorted(os.listdir(path))
    print(f"\n{dataset_name} — {split_name}")
    print("-" * 40)
    for cls in classes:
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            if has_images_subfolder:
                images_path = os.path.join(cls_path, "images")
                count = len(os.listdir(images_path)) if os.path.exists(images_path) else 0
            else:
                count = len(os.listdir(cls_path))
            total += count
            print(f"  {cls}: {count} images")
    print(f"  TOTAL: {total} images")

# ── 2. Show sample images ─────────────────────────────────────────────
def show_sample_images(dataset_path, dataset_name,
                       has_images_subfolder=False,
                       save_dir="results/eda"):
    classes = sorted(os.listdir(dataset_path))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f"Sample Images — {dataset_name}", fontsize=16)
    for i, (ax, cls) in enumerate(zip(axes.flat, classes)):
        cls_path = os.path.join(dataset_path, cls)
        if has_images_subfolder:
            cls_path = os.path.join(cls_path, "images")
        images   = os.listdir(cls_path)
        img_path = os.path.join(cls_path, images[0])
        img      = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(cls, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{dataset_name}_samples.png")
    plt.show()
    print(f"Saved to {save_dir}/{dataset_name}_samples.png")

# ── 3. Plot class distribution ────────────────────────────────────────
def plot_class_distribution(dataset_path, dataset_name,
                             has_images_subfolder=False,
                             class_mapping=None,
                             save_dir="results/eda"):
    classes = sorted(os.listdir(dataset_path))
    counts  = []
    labels  = []
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        if not os.path.isdir(cls_path):
            continue
        img_path = os.path.join(cls_path, "images") if has_images_subfolder else cls_path
        count    = len(os.listdir(img_path))
        counts.append(count)
        label = class_mapping.get(cls, cls) if class_mapping else cls
        labels.append(label)
    plt.figure(figsize=(12, 5))
    bars = plt.bar(labels, counts, color="steelblue", edgecolor="white")
    plt.title(f"Class Distribution — {dataset_name}", fontsize=14)
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 str(count), ha="center", fontsize=9)
    if len(set(counts)) == 1:
        plt.figtext(0.5, 0.01,
                    "Perfectly balanced — all classes have equal images",
                    ha="center", fontsize=10, color="green")
    else:
        plt.figtext(0.5, 0.01,
                    "Imbalanced — classes have different number of images",
                    ha="center", fontsize=10, color="red")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{dataset_name}_class_distribution.png")
    plt.show()
    print(f"Balanced: {len(set(counts)) == 1}")

# ── 4. Compare datasets ───────────────────────────────────────────────
def compare_datasets(cifar_train, cifar_val, cifar_test,
                     tiny_train,  tiny_val,  tiny_test,
                     save_dir="results/eda"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("CIFAR-10 vs Tiny ImageNet — Comparison", fontsize=16)
    datasets = [
        {"name": "CIFAR-10",      "train": cifar_train,
         "val":  cifar_val,       "test":  cifar_test,  "subfolder": False},
        {"name": "Tiny ImageNet", "train": tiny_train,
         "val":  tiny_val,        "test":  tiny_test,   "subfolder": True},
    ]
    for ax, ds in zip(axes, datasets):
        splits = ["Train", "Val", "Test"]
        colors = ["steelblue", "orange", "green"]
        totals = []
        for path in [ds["train"], ds["val"], ds["test"]]:
            classes = sorted(os.listdir(path))
            total   = 0
            for cls in classes:
                cls_path = os.path.join(path, cls)
                if os.path.isdir(cls_path):
                    img_path = os.path.join(cls_path, "images") if ds["subfolder"] else cls_path
                    total   += len(os.listdir(img_path))
            totals.append(total)
        bars = ax.bar(splits, totals, color=colors, edgecolor="white", width=0.5)
        ax.set_title(ds["name"], fontsize=13)
        ax.set_ylabel("Number of Images")
        ax.set_xlabel("Split")
        for bar, count in zip(bars, totals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 50,
                    str(count), ha="center", fontsize=11)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/dataset_comparison.png")
    plt.show()

# ── 5. Full comparison summary ────────────────────────────────────────
def full_comparison_summary(cifar_train, tiny_train):
    cifar_cls  = os.listdir(cifar_train)[0]
    cifar_img  = os.listdir(os.path.join(cifar_train, cifar_cls))[0]
    cifar_size = Image.open(os.path.join(cifar_train, cifar_cls, cifar_img)).size
    tiny_cls   = os.listdir(tiny_train)[0]
    tiny_img   = os.listdir(os.path.join(tiny_train, tiny_cls, "images"))[0]
    tiny_size  = Image.open(os.path.join(tiny_train, tiny_cls, "images", tiny_img)).size
    print("=" * 55)
    print(f"{'':>5} {'CIFAR-10':>20} {'Tiny ImageNet':>20}")
    print("=" * 55)
    print(f"{'Classes':<20} {'10':>15} {'10':>20}")
    print(f"{'Image size':<20} {str(cifar_size):>15} {str(tiny_size):>20}")
    print(f"{'Train images':<20} {'10,000':>15} {'5,000':>20}")
    print(f"{'Val images':<20} {'1,000':>15} {'250':>20}")
    print(f"{'Test images':<20} {'10,000':>15} {'250':>20}")
    print(f"{'Total images':<20} {'21,000':>15} {'5,500':>20}")
    print(f"{'Balanced':<20} {'Yes':>15} {'Yes':>20}")
    print(f"{'Label format':<20} {'readable':>15} {'WordNet ID':>20}")
    print("=" * 55)
    print("\nKey differences:")
    print("  1. Image size   — CIFAR-10 32x32  Tiny ImageNet 64x64")
    print("  2. Dataset size — CIFAR-10 has 4x more images overall")
    print("  3. Label names  — CIFAR-10 readable  Tiny uses codes")
    print("  4. Test size    — CIFAR-10 test 40x larger than Tiny")

# ── 6. Compute mean and std ───────────────────────────────────────────
def compute_dataset_stats(dataset_path, dataset_name,
                           has_images_subfolder=False):
    print(f"\nComputing stats for {dataset_name}...")
    all_pixels = []
    classes    = sorted(os.listdir(dataset_path))
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        if not os.path.isdir(cls_path):
            continue
        img_path = os.path.join(cls_path, "images") if has_images_subfolder else cls_path
        for img_file in os.listdir(img_path):
            full_path = os.path.join(img_path, img_file)
            try:
                img       = Image.open(full_path).convert("RGB")
                img_array = np.array(img) / 255.0
                all_pixels.append(img_array)
            except:
                continue
    all_pixels = np.array(all_pixels)
    mean = all_pixels.mean(axis=(0, 1, 2))
    std  = all_pixels.std(axis=(0, 1, 2))
    print(f"  Mean (R,G,B): {mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}")
    print(f"  Std  (R,G,B): {std[0]:.4f},  {std[1]:.4f},  {std[2]:.4f}")
    return mean, std

# ── 7. Pixel heatmap per class ────────────────────────────────────────
def show_class_heatmaps(dataset_path, dataset_name,
                         has_images_subfolder=False,
                         class_mapping=None,
                         save_dir="results/eda"):
    classes = sorted(os.listdir(dataset_path))
    fig, axes = plt.subplots(len(classes), 4,
                              figsize=(16, len(classes) * 3))
    fig.suptitle(f"RGB Channel Heatmaps — {dataset_name}", fontsize=16)
    for row, cls in enumerate(classes):
        cls_path  = os.path.join(dataset_path, cls)
        if not os.path.isdir(cls_path):
            continue
        img_path  = os.path.join(cls_path, "images") if has_images_subfolder else cls_path
        img_file  = os.listdir(img_path)[0]
        img_array = np.array(Image.open(
                    os.path.join(img_path, img_file)).convert("RGB"))
        label     = class_mapping.get(cls, cls) if class_mapping else cls
        axes[row, 0].imshow(img_array)
        axes[row, 0].set_ylabel(label, fontsize=10)
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].axis("off")
        for col, (name, cmap) in enumerate(zip(
                ["Red","Green","Blue"], ["Reds","Greens","Blues"])):
            axes[row, col+1].imshow(img_array[:,:,col], cmap=cmap)
            axes[row, col+1].set_title(name if row == 0 else "")
            axes[row, col+1].axis("off")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{dataset_name}_class_heatmaps.png")
    plt.show()

# ── 8. Average pixel histogram ───────────────────────────────────────
def show_average_pixel_histogram(dataset_path, dataset_name,
                                  has_images_subfolder=False,
                                  sample_size=200,
                                  cifar_mean=None, cifar_std=None,
                                  tiny_mean=None,  tiny_std=None,
                                  save_dir="results/eda"):
    all_r, all_g, all_b = [], [], []
    classes = sorted(os.listdir(dataset_path))
    count   = 0
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        if not os.path.isdir(cls_path):
            continue
        img_path = os.path.join(cls_path, "images") if has_images_subfolder else cls_path
        images   = os.listdir(img_path)
        sampled  = random.sample(images, min(sample_size//len(classes), len(images)))
        for img_file in sampled:
            full_path = os.path.join(img_path, img_file)
            try:
                img       = Image.open(full_path).convert("RGB")
                img_array = np.array(img)
                all_r.extend(img_array[:,:,0].flatten().tolist())
                all_g.extend(img_array[:,:,1].flatten().tolist())
                all_b.extend(img_array[:,:,2].flatten().tolist())
                count += 1
            except:
                continue
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Pixel Distribution — {dataset_name} ({count} images)",
                 fontsize=14)
    axes[0].hist(all_r, bins=50, alpha=0.5, color="red",   label="Red",   density=True)
    axes[0].hist(all_g, bins=50, alpha=0.5, color="green", label="Green", density=True)
    axes[0].hist(all_b, bins=50, alpha=0.5, color="blue",  label="Blue",  density=True)
    axes[0].set_title("Before normalization (0-255)")
    axes[0].set_xlabel("Pixel value")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    mean    = np.array(cifar_mean if "CIFAR" in dataset_name else tiny_mean)
    std     = np.array(cifar_std  if "CIFAR" in dataset_name else tiny_std)
    all_r_n = ((np.array(all_r) / 255.0) - mean[0]) / std[0]
    all_g_n = ((np.array(all_g) / 255.0) - mean[1]) / std[1]
    all_b_n = ((np.array(all_b) / 255.0) - mean[2]) / std[2]
    axes[1].hist(all_r_n, bins=50, alpha=0.5, color="red",   label="Red",   density=True)
    axes[1].hist(all_g_n, bins=50, alpha=0.5, color="green", label="Green", density=True)
    axes[1].hist(all_b_n, bins=50, alpha=0.5, color="blue",  label="Blue",  density=True)
    axes[1].set_title("After normalization (centered at 0)")
    axes[1].set_xlabel("Pixel value")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{dataset_name}_avg_histogram.png")
    plt.show()

# ── 9. Image corruption check ─────────────────────────────────────────
def check_image_corruption_fast(dataset_path, dataset_name,
                                 has_images_subfolder=False,
                                 sample_size=20):
    classes     = sorted(os.listdir(dataset_path))
    total       = 0
    corrupted   = []
    wrong_mode  = []
    wrong_size  = []
    print(f"\nChecking {dataset_name} (sample of {sample_size} per class)...")
    print("-" * 40)
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        if not os.path.isdir(cls_path):
            continue
        img_path = os.path.join(cls_path, "images") if has_images_subfolder else cls_path
        all_imgs = os.listdir(img_path)
        sampled  = random.sample(all_imgs, min(sample_size, len(all_imgs)))
        for img_file in sampled:
            full_path = os.path.join(img_path, img_file)
            total += 1
            try:
                img = Image.open(full_path)
                img.verify()
                img = Image.open(full_path).convert("RGB")
                if img.mode != "RGB":
                    wrong_mode.append(full_path)
                expected = (32, 32) if "CIFAR" in dataset_name else (64, 64)
                if img.size != expected:
                    wrong_size.append({"file": full_path, "size": img.size})
            except Exception as e:
                corrupted.append({"file": full_path, "error": str(e)})
    print(f"  Total checked  : {total}")
    print(f"  Corrupted      : {len(corrupted)}")
    print(f"  Wrong mode     : {len(wrong_mode)}")
    print(f"  Unexpected size: {len(wrong_size)}")
    if len(corrupted) == 0 and len(wrong_mode) == 0 and len(wrong_size) == 0:
        print(f"  All clean!")
    print("-" * 40)
    return corrupted, wrong_mode, wrong_size

# ── 10. Average heatmap ───────────────────────────────────────────────
def show_average_heatmap(dataset_path, dataset_name,
                          has_images_subfolder=False,
                          sample_size=100,
                          save_dir="results/eda"):
    classes    = sorted(os.listdir(dataset_path))
    all_images = []
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        if not os.path.isdir(cls_path):
            continue
        img_path = os.path.join(cls_path, "images") if has_images_subfolder else cls_path
        images   = os.listdir(img_path)
        sampled  = random.sample(images, min(sample_size//len(classes), len(images)))
        for img_file in sampled:
            full_path = os.path.join(img_path, img_file)
            try:
                img = Image.open(full_path).convert("RGB")
                img = img.resize((64, 64))
                all_images.append(np.array(img))
            except:
                continue
    avg_image = np.mean(all_images, axis=0).astype(np.uint8)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Average Heatmap — {dataset_name} ({len(all_images)} images)",
                 fontsize=14)
    axes[0].imshow(avg_image)
    axes[0].set_title("Average image")
    axes[0].axis("off")
    for i, (name, cmap) in enumerate(zip(
            ["Red","Green","Blue"], ["Reds","Greens","Blues"])):
        axes[i+1].imshow(avg_image[:,:,i], cmap=cmap)
        axes[i+1].set_title(f"Avg {name}")
        axes[i+1].axis("off")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{dataset_name}_avg_heatmap.png")
    plt.show()
    print(f"  Red   mean: {avg_image[:,:,0].mean():.1f}")
    print(f"  Green mean: {avg_image[:,:,1].mean():.1f}")
    print(f"  Blue  mean: {avg_image[:,:,2].mean():.1f}")
