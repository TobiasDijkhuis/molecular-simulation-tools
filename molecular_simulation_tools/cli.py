"""Command-line interface."""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from ase.db import connect
from ase.visualize import view

from molecular_simulation_tools._version import version
from molecular_simulation_tools.benchmark import (
    BenchmarkConfig,
    plot_energies,
    run_benchmark,
    setup_benchmark,
)
from molecular_simulation_tools.io import read

if TYPE_CHECKING:
    from ase import Atoms


def _gui(args: argparse.Namespace) -> None:
    images: list[Atoms] = []
    for path in args.paths:
        atoms = read(path, index=args.index)
        if not isinstance(atoms, list):
            atoms = [atoms]
        images.extend(atoms)
    view(images, block=args.block)


def _benchmark(args: argparse.Namespace) -> None:
    config = BenchmarkConfig.from_yaml(args.path)
    setup_benchmark(config, verbose=args.verbose)
    print(args.dry_run)
    if args.dry_run:
        return
    run_benchmark(config, max_workers=args.max_workers, verbose=args.verbose)


def _plot(args: argparse.Namespace) -> None:
    if not Path(args.path).is_file():
        raise FileNotFoundError
    db = connect(args.path)
    plot_energies(db)


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlt", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version}")
    subparsers = parser.add_subparsers()

    parser_gui = subparsers.add_parser(
        "gui",
        help="View a molecular geometry. Reads more filetypes than 'ase gui'.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_gui.add_argument(
        "paths",
        nargs="+",
        type=str,
        help="Path to the structure, or to multiple structures.",
    )
    parser_gui.add_argument(
        "--block", action="store_true", help="Whether to block while the file is shown."
    )
    parser_gui.add_argument(
        "--index",
        type=str,
        help="Which index to read from the file",
        default=":",
        required=False,
    )
    parser_gui.set_defaults(func=_gui)

    parser_benchmark = subparsers.add_parser(
        "benchmark",
        help="Benchmark multiple ways of calculating structures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_benchmark.add_argument("path", type=str, help="Path to the .yaml file")
    parser_benchmark.add_argument(
        "-j",
        "--max_workers",
        type=int,
        help="Maximum number of items to run at once.",
        default=-1,
        required=False,
    )
    parser_benchmark.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write the inputs, without running the benchmark.",
    )
    parser_benchmark.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print information during benchmarking.",
    )
    parser_benchmark.set_defaults(func=_benchmark)

    parser_plot = subparsers.add_parser(
        "plot",
        help="Plot benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_plot.add_argument("path", type=str, help="Path to the benchmark db")
    parser_plot.set_defaults(func=_plot)

    return parser


def main() -> None:
    parser = get_argument_parser()

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
