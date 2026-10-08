"""Start vLLM with direct attachment installed before the engine is spawned."""

import sys

import stator_vllm
import vllm_stator_direct

stator_vllm.accept_device_names()
vllm_stator_direct.register()


def main():
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = ["vllm", "serve", *sys.argv[1:]]
    return vllm_main()


if __name__ == "__main__":
    sys.exit(main())
