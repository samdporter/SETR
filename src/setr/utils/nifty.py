import os


def load_zoom_factors(spect_dir):
    """
    Load previously saved zoom factors from file.

    Args:
        spect_dir: Directory containing the zoom factors file

    Returns:
        tuple: Zoom factors (z, y, x)
    """
    zoom_file_path = os.path.join(spect_dir, "spect_to_pet_zoom_factors.txt")

    if not os.path.exists(zoom_file_path):
        raise FileNotFoundError(f"Zoom factors file not found: {zoom_file_path}")

    with open(zoom_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#") and line:
                zoom_values = line.split()
                return (float(zoom_values[0]), float(zoom_values[1]), float(zoom_values[2]))

    raise ValueError("No zoom factors found in file")
