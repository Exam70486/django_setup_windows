FROM python:3.12-slim

# Evita que Python escriba archivos .pyc en el contenedor
ENV PYTHONDONTWRITEBYTECODE=1
# Evita que Python guarde en búfer las salidas de la terminal (útil para ver logs en tiempo real en Render)
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema y el ODBC Driver 18 para SQL Server
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    apt-transport-https \
    ca-certificates \
    && mkdir -p /usr/share/keyrings \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl -sSL https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && sed -i 's|deb \[arch=amd64,arm64,armhf\]|deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-prod.gpg]|g' /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 unixodbc-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los requerimientos e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar todo el código fuente del proyecto al contenedor
COPY . .

# Exponer el puerto por defecto que utiliza Render
EXPOSE 10000

# Ejecutar la aplicación usando Gunicorn apuntando a tu módulo web_project
CMD ["gunicorn", "web_project.wsgi:application", "--bind", "0.0.0.0:10000"]