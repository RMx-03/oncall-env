"""Server entrypoint shim forwarding to the main environment app."""

from oncall_env.server.app import app
from oncall_env.server.app import main as _real_main


def main() -> None:
    _real_main()


if __name__ == "__main__":
    main()
