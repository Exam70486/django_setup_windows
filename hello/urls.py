from django.urls import path
from hello import views
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularSwaggerView,
    SpectacularRedocView
)

urlpatterns = [
    path(""                            , views.home                                       , name="home"),
    path('getAllLogs'                  , views.getAllLogs                                 , name='getAllLogs'),
    path('getAllPersons'               , views.getAllPersons                              , name='getAllPersons'),
    path('getAllContactForms'          , views.getAllContactForms                         , name='getAllContactForms'),
    path('health/'                     , views.health_check                               , name='health_check'),  # Optional
    path('getAppVersion'               , views.getAppVersion                              , name='getAppVersion'),
    path('getPythonVersion'            , views.getPythonVersion                           , name='getPythonVersion'),
    path('getPythonWebServerVersion'   , views.getPythonWebServerVersion                  , name='getPythonWebServerVersion'),    
    path('ping'                        , views.ping                                       , name='ping'),
    path('api/schema/'                 , SpectacularAPIView.as_view()                     , name='schema'),
    path('api/docs/'                   , SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/redoc/'                  , SpectacularRedocView.as_view(url_name='schema')  , name='redoc')           
]