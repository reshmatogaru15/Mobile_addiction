"""Mobile_addiction URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from application import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.home,name='home'),
    path('register',views.register,name='register'),
    path('login',views.login_view,name='login'),
    path('upload',views.Upload_data,name='upload'),
    path('dtc',views.DTC,name='dtc'),
    path('rfc',views.RFC,name='rfc'),
    path('catboost',views.catboost_view,name='catboost'),
    path('logout',views.logout_view),
    path('prediction',views.prediction,name='prediction')
]
