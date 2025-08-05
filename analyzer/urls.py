from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('compare/', views.compare_villages, name='compare_villages'),
    path('get-villages/', views.get_villages, name='get_villages'),
] 