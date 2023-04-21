#!/bin/bash
set -ex

pip install -U wheel packaging requests opt_einsum
pip install -U --user keras_preprocessing --no-deps
brew install bazelisk

conda install pytorch torchvision torchaudio -c pytorch-nightly

conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal

pushd ..
git clone git@github.com:google-research/google-research.git
popd

pushd ../google-research/scann
git checkout ea7fbce
python configure.py --no-deps
# bazel clean --expunge
CC=clang-8 bazel build -c opt --features=thin_lto --copt=-mavx --copt=-mfma --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w :build_pip_pkg
./bazel-bin/build_pip_pkg
popd
