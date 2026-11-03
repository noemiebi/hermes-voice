import uvicorn

from . import config

if __name__ == "__main__":
    uvicorn.run("server.app:app", host=config.HOST, port=config.PORT, log_level="info", ws_max_size=4 * 1024 * 1024)
