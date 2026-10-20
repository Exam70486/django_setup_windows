
from django.http import JsonResponse
from django.http import FileResponse, JsonResponse,  HttpResponse
from django.shortcuts import render
from django.db import connection
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import serializers
from rest_framework import status
from tensorflow.keras import layers, models
from typing import Tuple, Dict
import django
import socket
import sys
import json
import zipfile
import shutil
import tempfile
import random
import numpy as np
import tensorflow as tf
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings

import logging
import math # Import math module to check for special values

logger = logging.getLogger(__name__)

#############################################
# BEGIN GENERAL FUNCIONALITY
#############################################

@api_view(['GET'])
def home(request):
    return HttpResponse("Hello, Django!")

@api_view(['GET'])
def getAppVersion(request):
    current_version      = '1.0.1'
    return HttpResponse(f"{current_version}")

@api_view(['GET'])
def getPythonVersion(request):
    current_version = sys.version
    return HttpResponse(f"{current_version}")

@api_view(['GET'])
def getPythonWebServerVersion(request):
    current_version = django.get_version()
    return HttpResponse(f"{current_version}")

@api_view(['GET'])
def ping(request):
    return Response(status=204)

@api_view(['GET'])
# views.py

def health(request):
    from hello.urls import ROUTES_MAP

    server_info = {
        "status": "UP",
        "message": "The server is working properly.",
        "server_hostname": socket.gethostname(),
        "django_version": django.get_version(),
        "python_version": sys.version.split(' ')[0]
    }

    endpoints = {
        key: {
            "path": f"/{data['path'].strip('/')}/" if data['path'] else "/",
            "name": data["name"],
            "description": data["description"]
        }
        for key, data in ROUTES_MAP.items()
    }

    return JsonResponse({
        "server": server_info,
        "endpoints": endpoints
    }, status=200)
    
#############################################
# END GENERAL FUNCIONALITY
#############################################


# #######################################
# BEGIN DATABASE END POINTS
# #######################################



class RawDataSerializer(serializers.Serializer):
    def to_representation(self, instance):
        # Convert row tuple to dictionary
        fields = [field[0] for field in self.context['cursor'].description]
        return dict(zip(fields, instance))


@api_view(['GET'])
def getAllLogs(request):
    try:
        with connection.cursor() as cursor:
            # Your raw SQL query
            sql = """

                 SELECT TOP 100
                       AL.[ID_column]  id_Column
                      ,AL.[PageName]   pageName
                      ,AL.[AccessDate] accessDate
                      ,AL.[IpValue]    ipValue
                 FROM
                       [dbo].[accessLogs] AL
                    WHERE
                       AL.[LogType] = 1
                    AND
                        (AL.PAGENAME LIKE '%DEMO%'
                    and
                        AL.PAGENAME LIKE '%PAGE%')
                    AND
                        AL.PAGENAME NOT LIKE '%ERROR%'
                    AND
                        AL.PAGENAME  NOT LIKE '%PAGE_DEMO_INDEX%'
                    AND
                        UPPER(AL.PAGENAME) NOT LIKE '%CACHE%'
                    AND
                        AL.IPVALUE <> '::1'
                 order by
                       AL.[ID_column] desc
                  """
            cursor.execute(sql)
            rows = cursor.fetchall()
            serializer = RawDataSerializer(rows, many=True, context={
                                           'cursor': cursor})   # Pass cursor for field names
            return Response(serializer.data)
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def getAllPersons(request):
    try:
        with connection.cursor() as cursor:
            # Your raw SQL query
            sql = """

                  SELECT
                     [Id_Column]        id_Column
                    ,[NombreCompleto]   nombreCompleto
                    ,[ProfesionOficio]  profesionOficio
                    ,[Ciudad]           ciudad
                FROM
                    [dbo].[Persona]
                ORDER BY
                    Id_Column

                """
            cursor.execute(sql)
            rows = cursor.fetchall()
            serializer = RawDataSerializer(rows, many=True, context={
                                           'cursor': cursor})  # Pass cursor for field names
            return Response(serializer.data)
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def getAllContactForms(request):
    try:
        with connection.cursor() as cursor:
            # Your raw SQL query
            sql = """

                  SELECT
                        id         id
                        ,Name      name
                        ,Email     field_1
                        ,Message   field_2
                        ,CreatedAt field_3
                FROM
                    ContactForm
                ORDER BY
                    id desc
                """
            cursor.execute(sql)
            rows = cursor.fetchall()
            serializer = RawDataSerializer(rows, many=True, context={
                                           'cursor': cursor})  # Pass cursor for field names
            return Response(serializer.data)
    except Exception as e:
        return Response({'error': str(e)}, status=500)


# ##########################
# END DATABASE END POINTS
# ##########################

