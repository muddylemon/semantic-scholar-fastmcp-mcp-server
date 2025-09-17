"""
Main server module for the Semantic Scholar MCP Server.
Runs FastMCP as an ASGI app on Cloud Run (HTTP on 0.0.0.0:$PORT).
"""

import asyncio
import logging
import os
import signal
from typing import Any, Dict

import uvicorn
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount

# Import tool registration side-effects (keep these if your tools register on import)
try:
    from .api import papers, authors, recommendations  # noqa: F401
except Exception:
    # If your API modules register tools conditionally, failure to import
    # shouldn't crash the server start—log and proceed.
    pass

# Centralized FastMCP instance
from .mcp import mcp  # type: ignore

# Optional: if you have a client init/teardown for upstream APIs
try:
    from .http_client import initialize_client, cleanup_client  # type: ignore
except Exception:
    async def initialize_client() -> None:
        return

    async def cleanup_client() -> None:
        return


# -----------------
# Logging
# -----------------
logger = logging.getLogger("semantic_scholar.server")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(levelname)s:%(name)s:%(message)s",
)


# -----------------
# Event-loop helpers
# -----------------
async def handle_exception(_loop: asyncio.AbstractEventLoop, context: Dict[str, Any]) -> None:
    msg = context.get("exception") or context.get("message")
    logger.error(f"Caught exception: {msg}")
    # Try graceful shutdown rather than hard exit
    asyncio.create_task(shutdown())


def init_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    logger.info("Signal handlers initialized")

    def _graceful(*_args: Any) -> None:
        loop.create_task(shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _graceful)
        except NotImplementedError:
            # Windows / environments without signal support
            pass


# -----------------
# Shutdown
# -----------------
async def shutdown() -> None:
    """Gracefully shut down the server."""
    logger.info("Initiating graceful shutdown...")

    # Cancel all other tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Cleanup resources (FastMCP has no 'cleanup()')
    try:
        await cleanup_client()
    except Exception as e:
        logger.warning(f"Ignoring cleanup error: {e}")

    logger.info(f"Cancelled {len(tasks)} tasks")
    logger.info("Shutdown complete")


# -----------------
# Server
# -----------------
async def run_server() -> None:
    """Serve MCP as an ASGI app under Uvicorn on Cloud Run."""
    try:
        # Initialize any upstream clients/secrets
        await initialize_client()

        # Build the MCP ASGI app (HTTP transport)
        core_app = mcp.http_app()  # or: mcp.streamable_http_app() if you need streaming
        app = Starlette(routes=[Mount("/", app=core_app)])

        # Health endpoint for Cloud Run's TCP/HTTP probes
        @app.route("/health")
        async def health(_request):
            return PlainTextResponse("ok", status_code=200)

        port = int(os.getenv("PORT", "8080"))
        logger.info("Starting Semantic Scholar Server on 0.0.0.0:%d", port)

        config = uvicorn.Config(app, host="0.0.0.0",
                                port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        await shutdown()


def main() -> None:
    """Main entry point."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(handle_exception)
        init_signal_handlers(loop)
        loop.run_until_complete(run_server())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        try:
            # Let pending tasks finish
            loop.run_until_complete(asyncio.sleep(0))
            loop.close()
        except Exception as e:
            logger.error(f"Error during final cleanup: {e}")
        logger.info("Server stopped")


if __name__ == "__main__":
    main()
