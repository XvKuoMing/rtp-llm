from fastapi import Request

from ..exceptions import ConfigurationError

from ..server_manager import ServerManager


def set_app_server_manager(app, manager: ServerManager) -> None:
    app.state.server_manager = manager


def get_server_manager(request: Request) -> ServerManager:
    manager = getattr(request.app.state, "server_manager", None)
    if manager is None:
        raise ConfigurationError("ServerManager is not initialized on app.state")
    return manager


