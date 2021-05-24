import pickle
import yaml
import sys
from utils.set_seed import set_seed


def make_games(dataset, number_of_images, number_of_games, game_type, seed=None):
	# LOAD DATASET
	from utils.dataprep import load_emb_pickled
	metadata, embeddings = load_emb_pickled(dataset)
	filenames = metadata.get("fnames")
	categories = metadata.get("categories")

	# CREATE GAME
	game_settings = {
		"images": embeddings,
		"categories": categories,
		"images_filenames": filenames
	}

	from game import Game
	game = Game(**game_settings)

	if seed is not None:
		set_seed(seed)

	return game.generate_games(number_of_games, number_of_images, game_type)


def main(out_path, **kwargs):
	games = make_games(**kwargs)

	with open(out_path, "wb") as f:
		pickle.dump(games, f)
	print(f"Saved to {out_path}")


if __name__ == "__main__":
	if len(sys.argv) == 2:
		config = sys.argv[1]
	else:
		config = "settings-make-test.yml"
	with open(config, "r") as f:
		settings = yaml.load(f)
	main(**settings)
