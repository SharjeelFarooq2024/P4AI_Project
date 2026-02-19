"""Small helper to peek at the UNSW-NB15 CSVs without loading everything."""

import argparse
import sys
from pathlib import Path

import pandas as pd


# Make project modules importable when running this file directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from src.config import CSV_FILES, DATA_PATH, MULTICLASS_COLUMN, TARGET_COLUMN  # noqa: E402
from src.data_loader import load_feature_names  # noqa: E402


def list_available_files():
	files = [Path(DATA_PATH) / name for name in CSV_FILES]
	feature_file = Path(DATA_PATH) / "NUSW-NB15_features.csv"
	print("Data files:")
	for f in files:
		print(f" - {f.name} ({f})")
	print(f"Features file:\n - {feature_file.name} ({feature_file})")


def view_feature_rows(nrows: int) -> None:
	feature_path = Path(DATA_PATH) / "NUSW-NB15_features.csv"
	df = pd.read_csv(feature_path, nrows=nrows)
	with pd.option_context("display.max_rows", None, "display.max_columns", None):
		print(df)


def view_data_rows(file_name: str, start: int, nrows: int) -> None:
	csv_path = Path(DATA_PATH) / file_name
	feature_names = load_feature_names()
	skip = range(start) if start > 0 else None
	df = pd.read_csv(
		csv_path,
		header=None,
		names=feature_names,
		low_memory=False,
		skiprows=skip,
		nrows=nrows,
	)
	with pd.option_context("display.max_rows", None, "display.max_columns", None):
		print(df)


def list_unique_categories(column: str) -> None:
	feature_names = load_feature_names()
	if column not in feature_names:
		print(f"Column '{column}' not found. Available columns: {len(feature_names)} total")
		return

	uniques = set()
	for file_name in CSV_FILES:
		csv_path = Path(DATA_PATH) / file_name
		for chunk in pd.read_csv(
			csv_path,
			header=None,
			names=feature_names,
			usecols=[column],
			chunksize=100_000,
			low_memory=False,
		):
			uniques.update(chunk[column].dropna().unique())

	print(f"Unique values for '{column}' across all files ({len(uniques)}):")
	for val in sorted(uniques):
		print(f" - {val}")


def count_attack_categories() -> None:
	"""Count occurrences of each attack category across all raw CSV files."""
	feature_names = load_feature_names()

	if MULTICLASS_COLUMN not in feature_names:
		print(
			f"Column '{MULTICLASS_COLUMN}' not found. Available columns: {len(feature_names)} total"
		)
		return

	counts = pd.Series(dtype="int64")
	for file_name in CSV_FILES:
		csv_path = Path(DATA_PATH) / file_name
		for chunk in pd.read_csv(
			csv_path,
			header=None,
			names=feature_names,
			usecols=[MULTICLASS_COLUMN],
			chunksize=100_000,
			low_memory=False,
		):
			counts = counts.add(chunk[MULTICLASS_COLUMN].value_counts(dropna=False), fill_value=0)

	counts = counts.sort_values(ascending=False)
	print(
		f"Attack category counts across all files ({int(counts.sum())} rows):"
	)
	for category, count in counts.items():
		label = "<NA>" if pd.isna(category) else category
		print(f" - {label}: {int(count)}")
	print(f"Unique categories: {counts.count()}")


def _count_column(files, column: str) -> pd.Series:
	"""Stream-count a single column across given CSV files."""
	feature_names = load_feature_names()

	if column not in feature_names:
		raise ValueError(
			f"Column '{column}' not found. Available columns: {len(feature_names)} total"
		)

	counts = pd.Series(dtype="int64")
	for file_name in files:
		csv_path = Path(DATA_PATH) / file_name
		for chunk in pd.read_csv(
			csv_path,
			header=None,
			names=feature_names,
			usecols=[column],
			chunksize=100_000,
			low_memory=False,
		):
			counts = counts.add(chunk[column].value_counts(dropna=False), fill_value=0)
	return counts.sort_values(ascending=False)


def _print_section(title: str, counts: pd.Series) -> None:
	total = int(counts.sum())
	print(title)
	print(f" Total rows: {total}")
	for cat, cnt in counts.items():
		label = "<NA>" if pd.isna(cat) else cat
		pct = (cnt / total) * 100 if total else 0
		print(f"  - {label}: {int(cnt)} ({pct:.2f}%)")
	print(f" Unique categories: {counts.count()}")
	print(" Imbalance (max/min): ", end="")
	if counts.empty or counts.min() == 0:
		print("n/a")
	else:
		print(f"{counts.max():.0f}:{counts.min():.0f} (~{counts.max()/counts.min():.1f}x)")
	print()


def compare_attack_and_label_counts() -> None:
	"""Compare attack_cat and label distribution for first 2 files vs all 4 files."""
	first_two = CSV_FILES[:2]
	all_files = CSV_FILES

	attack_first = _count_column(first_two, MULTICLASS_COLUMN)
	attack_all = _count_column(all_files, MULTICLASS_COLUMN)
	label_first = _count_column(first_two, TARGET_COLUMN)
	label_all = _count_column(all_files, TARGET_COLUMN)

	print("=== attack_cat ===")
	_print_section("First two files", attack_first)
	_print_section("All four files", attack_all)

	print("=== Label (intrusion vs normal) ===")
	_print_section("First two files", label_first)
	_print_section("All four files", label_all)


def compare_last_two_vs_all() -> None:
	"""Compare attack_cat and label distribution for files 3+4 vs all 4 files."""
	last_two = CSV_FILES[2:]
	all_files = CSV_FILES

	attack_last = _count_column(last_two, MULTICLASS_COLUMN)
	attack_all = _count_column(all_files, MULTICLASS_COLUMN)
	label_last = _count_column(last_two, TARGET_COLUMN)
	label_all = _count_column(all_files, TARGET_COLUMN)

	print("=== attack_cat (files 3+4 vs all) ===")
	_print_section("Files 3 and 4", attack_last)
	_print_section("All four files", attack_all)

	print("=== Label (intrusion vs normal) ===")
	_print_section("Files 3 and 4", label_last)
	_print_section("All four files", label_all)


def compare_pairs_vs_all() -> None:
	"""Compare selected file pairs vs all four for attack_cat and label."""
	all_files = CSV_FILES
	pairs = {
		"Files 1 and 2": CSV_FILES[0:2],
		"Files 2 and 3": CSV_FILES[1:3],
		"Files 3 and 4": CSV_FILES[2:4],
		"Files 1 and 4": [CSV_FILES[0], CSV_FILES[3]],
	}

	attack_all = _count_column(all_files, MULTICLASS_COLUMN)
	label_all = _count_column(all_files, TARGET_COLUMN)

	for title, subset in pairs.items():
		attack_subset = _count_column(subset, MULTICLASS_COLUMN)
		label_subset = _count_column(subset, TARGET_COLUMN)

		print(f"=== attack_cat ({title} vs all) ===")
		_print_section(title, attack_subset)
		_print_section("All four files", attack_all)

		print(f"=== Label (intrusion vs normal) ({title} vs all) ===")
		_print_section(title, label_subset)
		_print_section("All four files", label_all)

def main():
	parser = argparse.ArgumentParser(description="View slices of UNSW-NB15 data.")
	parser.add_argument("--file", choices=CSV_FILES, default=CSV_FILES[0], help="Which data file to read")
	parser.add_argument("--rows", type=int, default=5, help="Number of rows to display")
	parser.add_argument("--start", type=int, default=0, help="Row offset (0-based)")
	parser.add_argument("--features", action="store_true", help="Show the features file instead of data")
	parser.add_argument("--list", action="store_true", help="List available files and exit")
	parser.add_argument("--unique", metavar="COLUMN", help="List unique values for the given column across all files")
	parser.add_argument(
		"--count-attack-cat",
		action="store_true",
		help="Count occurrences of each attack category across all files",
	)
	parser.add_argument(
		"--compare-attack",
		action="store_true",
		help="Compare attack_cat and label counts (first two files vs all four files)",
	)
	parser.add_argument(
		"--compare-last-two",
		action="store_true",
		help="Compare attack_cat and label counts (files 3+4 vs all four files)",
	)
	parser.add_argument(
		"--compare-pairs",
		action="store_true",
		help="Compare selected pairs (1&2, 2&3, 3&4, 1&4) vs all four files",
	)

	args = parser.parse_args()

	if args.list:
		list_available_files()
		return

	if args.unique:
		list_unique_categories(args.unique)
		return

	if args.count_attack_cat:
		count_attack_categories()
		return

	if args.compare_attack:
		compare_attack_and_label_counts()
		return

	if args.compare_last_two:
		compare_last_two_vs_all()
		return

	if args.compare_pairs:
		compare_pairs_vs_all()
		return

	if args.features:
		view_feature_rows(args.rows)
	else:
		view_data_rows(args.file, args.start, args.rows)


if __name__ == "__main__":
	main()
