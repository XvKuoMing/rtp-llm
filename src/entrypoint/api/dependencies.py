from fastapi import Request

from ..server_manager import ServerManager


def set_app_server_manager(app, manager: ServerManager) -> None:
    app.state.server_manager = manager


def get_server_manager(request: Request) -> ServerManager:
    manager = getattr(request.app.state, "server_manager", None)
    if manager is None:
        raise RuntimeError("ServerManager is not initialized on app.state")
    return manager


