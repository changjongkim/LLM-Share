"""vLLM's server with the plugin in place from the first import.

  python -m stator_vllm.serve MODEL [the arguments of `vllm serve`]

vLLM 0.20.0 reads the device of a process as an integer while it imports
its layers, which is before it loads its plugins, and a MIG instance is
named by a UUID. A process that starts here accepts such a name first. The
engine of vLLM is a spawned child; a spawned child imports the main module
of its parent before it reads its arguments, so the same holds there.
"""
import sys

import stator_vllm

stator_vllm.accept_device_names()


def main():
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = ["vllm", "serve", *sys.argv[1:]]
    return vllm_main()


if __name__ == "__main__":
    sys.exit(main())
