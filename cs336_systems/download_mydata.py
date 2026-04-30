from pathlib import Path
from cs336_systems.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image
import modal


@app.function(image=build_image(), volumes=VOLUME_MOUNTS)
def fetch_file(filename: str) -> bytes:
    path = DATA_PATH / filename
    return path.read_bytes()


@app.local_entrypoint()
def download_from_modal():
    filename ="profile_leaderboard.nsys-rep"

    data = fetch_file.remote(filename)

    local_path = Path("data/profile_leaderboard_change_kernle.nsys-rep")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    local_path.write_bytes(data)

    print(f"Downloaded to {local_path}")