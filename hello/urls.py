from django.urls import path
from hello import views
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularSwaggerView,
    SpectacularRedocView
)

# Key-based dictionary mapping route keys to path, view, name, and description
ROUTES_MAP = {
    "home": {
        "path": "",
        "view": views.home,
        "name": "home",
        "description": "Root endpoint returning welcome message."
    },
    "getAllLogs": {
        "path": "getAllLogs",
        "view": views.getAllLogs,
        "name": "getAllLogs",
        "description": "Retrieves all application log records from the database."
    },
    "getAllPersons": {
        "path": "getAllPersons",
        "view": views.getAllPersons,
        "name": "getAllPersons",
        "description": "Retrieves all person entities from the database."
    },
    "getAllContactForms": {
        "path": "getAllContactForms",
        "view": views.getAllContactForms,
        "name": "getAllContactForms",
        "description": "Retrieves all submitted contact form records."
    },
    "health": {
        "path": "health",
        "view": views.health,
        "name": "health",
        "description": "Returns server status, runtime information, and API routes listing."
    },
    "getAppVersion": {
        "path": "getAppVersion",
        "view": views.getAppVersion,
        "name": "getAppVersion",
        "description": "Returns the current application release version."
    },
    "getPythonVersion": {
        "path": "getPythonVersion",
        "view": views.getPythonVersion,
        "name": "getPythonVersion",
        "description": "Returns the Python runtime version running on the host server."
    },
    "getPythonWebServerVersion": {
        "path": "getPythonWebServerVersion",
        "view": views.getPythonWebServerVersion,
        "name": "getPythonWebServerVersion",
        "description": "Returns the Django framework web server version."
    },
    "ping": {
        "path": "ping",
        "view": views.ping,
        "name": "ping",
        "description": "Simple latency and heartbeat check endpoint."
    },
    "schema": {
        "path": "api/schema/",
        "view": SpectacularAPIView.as_view(),
        "name": "schema",
        "description": "Generates raw OpenAPI 3 schema definition (JSON/YAML)."
    },
    "swagger-ui": {
        "path": "api/docs/",
        "view": SpectacularSwaggerView.as_view(url_name='schema'),
        "name": "swagger-ui",
        "description": "Interactive Swagger UI for API exploration and testing."
    },
    "redoc": {
        "path": "api/redoc/",
        "view": SpectacularRedocView.as_view(url_name='schema'),
        "name": "redoc",
        "description": "Read-only three-column ReDoc API documentation view."
    }
}

# Generate Django path objects dynamically from the dictionary
urlpatterns = [
    path(route_data["path"], route_data["view"], name=route_data["name"])
    for route_data in ROUTES_MAP.values()
]