from pathlib import Path
import zipfile
import retro

games_path = Path("./environments")
retro_path = Path(retro.__file__).parent / "data/stable" 

for game_path in games_path.iterdir():
    to_unzip = retro_path / game_path.stem
    with zipfile.ZipFile(str(game_path), 'r') as zip_ref: # str bc python 3.5 cannot support PosixPath class
        zip_ref.extractall(str(to_unzip))