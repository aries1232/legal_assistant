import os
from urllib.parse import urlparse

import chromadb


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _parse_host(host: str | None) -> tuple[str, int | None, bool]:
    if not host:
        return "api.trychroma.com", None, True

    normalized = host.strip()
    if "://" not in normalized:
        normalized = f"https://{normalized}"

    parsed = urlparse(normalized)
    parsed_host = parsed.hostname or "api.trychroma.com"
    parsed_port = parsed.port
    use_ssl = parsed.scheme != "http"
    return parsed_host, parsed_port, use_ssl


def get_chroma_client() -> chromadb.ClientAPI: # type: ignore
    chroma_host = _env("CHROMA_HOST")
    chroma_tenant = _env("CHROMA_TENANT")
    chroma_database = _env("CHROMA_DATABASE")
    chroma_api_key = _env("CHROMA_API_KEY")

    if chroma_api_key:
        cloud_host, cloud_port, enable_ssl = _parse_host(chroma_host)
        cloud_kwargs: dict[str, object] = {
            "api_key": chroma_api_key,
            "enable_ssl": enable_ssl,
        }

        if chroma_tenant:
            cloud_kwargs["tenant"] = chroma_tenant
        if chroma_database:
            cloud_kwargs["database"] = chroma_database
        if chroma_host:
            cloud_kwargs["cloud_host"] = cloud_host
        if cloud_port is not None:
            cloud_kwargs["cloud_port"] = cloud_port

        return chromadb.CloudClient(**cloud_kwargs) # type: ignore

    if chroma_host:
        server_host, server_port, use_ssl = _parse_host(chroma_host)
        http_kwargs: dict[str, object] = {
            "host": server_host,
            "ssl": use_ssl,
        }

        if server_port is not None:
            http_kwargs["port"] = server_port
        if chroma_tenant:
            http_kwargs["tenant"] = chroma_tenant
        if chroma_database:
            http_kwargs["database"] = chroma_database

        return chromadb.HttpClient(**http_kwargs) # type: ignore

    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db"))
    return chromadb.PersistentClient(path=db_path)