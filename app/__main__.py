"""Launch the local DNAscape NiceGUI prototype with ``python -m app``."""

from app._launch_progress import LaunchProgress


def main() -> None:
    """Show startup feedback before importing the interface and its dependencies."""
    with LaunchProgress() as progress:
        from app.main import main as start_app

        start_app(launch_progress=progress)

if __name__ == "__main__":
    main()
