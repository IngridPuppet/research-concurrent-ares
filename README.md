ARES with concurrency
---------------------

This repository investigates faster CPU inference on the ARES neural network.

## Installation

You need to install `e3nn_ares` before setting up `ares_release`.

Please follow their respective instructions:
- [e3nn_ares/README.md](e3nn_ares/README.md)
- [ares_release/README.md](ares_release/README.md)

If you opt for a venv, you might find useful the `requirements.txt` and
`requirements-cpu.txt` files at the root of the repository.

## Usage

The same instructions for inferencing the network still apply, except that
you would now pass two additional options:
- `--torch_threads` to set the number of PyTorch threads for intra-op parallelization
- `--async_threads` to set the number of concurrent threads to use in thread pools during forward pass

If the latter is not passed or set to a value less than 2, the original version of ARES is run.
