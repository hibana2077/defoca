"""Generate an experiments list file.

Input (set.txt) format:
  seeds 41,42,43
  archs resnet18,resnet50
  ratios 0.1,0.2

Output (experiments.txt) format:
  # Format: <seed> <arch> <ratio>
  # Lines starting with # are ignored.

  41 resnet18 0.1
  ...

Usage:
  python gen.py --set set.txt --out experiments.txt
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar


@dataclass(frozen=True)
class Settings:
	seeds: list[int]
	archs: list[str]
	ratios: list[str]


def _split_csv(value: str) -> list[str]:
	return [part.strip() for part in value.split(",") if part.strip()]


def load_settings(path: Path) -> Settings:
	seeds: list[int] | None = None
	archs: list[str] | None = None
	ratios: list[str] | None = None

	for raw_line in path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#"):
			continue
		parts = line.split(None, 1)
		if len(parts) != 2:
			raise ValueError(f"Invalid line in {path}: {raw_line!r}")
		key, value = parts[0].strip(), parts[1].strip()

		if key == "seeds":
			seeds = [int(x) for x in _split_csv(value)]
		elif key == "archs":
			archs = _split_csv(value)
		elif key == "ratios":
			# keep original string formatting (e.g., 0.1 not 0.10)
			ratios = _split_csv(value)
		else:
			raise ValueError(f"Unknown key {key!r} in {path}")

	missing = [name for name, v in [("seeds", seeds), ("archs", archs), ("ratios", ratios)] if v is None]
	if missing:
		raise ValueError(f"Missing keys in {path}: {', '.join(missing)}")

	return Settings(seeds=seeds or [], archs=archs or [], ratios=ratios or [])


def generate_rows(settings: Settings) -> Iterable[tuple[int, str, str]]:
	return itertools.product(settings.seeds, settings.archs, settings.ratios)


T = TypeVar("T")


def filter_values(
	values: Sequence[T],
	allow: set[str] | None,
	*,
	to_str: Callable[[T], str] = str,
) -> list[T]:
	if not allow:
		return list(values)
	return [v for v in values if to_str(v) in allow]


def write_experiments(path: Path, rows: Iterable[tuple[int, str, str]]) -> None:
	header = (
		"# Format: <seed> <arch> <ratio>\n"
		"# Lines starting with # are ignored.\n\n"
	)
	lines = [header]
	for seed, arch, ratio in rows:
		lines.append(f"{seed} {arch} {ratio}\n")
	path.write_text("".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Generate experiments.txt from set.txt")
	p.add_argument("--set", dest="set_path", default="set.txt", help="Path to set.txt")
	p.add_argument("--out", dest="out_path", default="experiments.txt", help="Output experiments file")
	p.add_argument("--seeds", default=None, help="Optional CSV to restrict seeds (e.g. 41,42)")
	p.add_argument("--archs", default=None, help="Optional CSV to restrict archs (e.g. resnet18)")
	p.add_argument("--ratios", default=None, help="Optional CSV to restrict ratios (e.g. 0.1,0.2)")
	return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)
	set_path = Path(args.set_path)
	out_path = Path(args.out_path)

	settings = load_settings(set_path)

	seed_allow = set(_split_csv(args.seeds)) if args.seeds else None
	arch_allow = set(_split_csv(args.archs)) if args.archs else None
	ratio_allow = set(_split_csv(args.ratios)) if args.ratios else None

	filtered = Settings(
		seeds=filter_values(settings.seeds, seed_allow, to_str=lambda x: str(x)),
		archs=filter_values(settings.archs, arch_allow, to_str=lambda x: x),
		ratios=filter_values(settings.ratios, ratio_allow, to_str=lambda x: x),
	)

	rows = generate_rows(filtered)
	write_experiments(out_path, rows)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
