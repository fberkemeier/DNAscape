"""Launch the local DNAscape NiceGUI prototype with ``python -m app``."""

from app._launch_progress import LaunchProgress, ShutdownProgress


def main() -> None:
    """Show startup feedback and treat Ctrl+C as a normal app shutdown."""
    with ShutdownProgress() as shutdown:
        try:
            with LaunchProgress() as progress:
                from app.main import main as start_app

                start_app(launch_progress=progress, shutdown_progress=shutdown)
        except KeyboardInterrupt:
            # Uvicorn finishes cleanup before re-raising the interrupt. This
            # also handles Ctrl+C during imports, before its handlers exist.
            pass
        shutdown.finish()

if __name__ == "__main__":
    main()
