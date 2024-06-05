from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.inventory_list, name='inventory_list'),
    path('edit/<int:id>/', views.edit_item, name='edit_item'),
    path('delete/<int:id>/', views.delete_item, name='delete_item'),
    path('visualization/', views.inventory_visualization, name='inventory_visualization'),
    path('add/', views.add_item, name='add_item'),  # Add the new URL pattern here
    path('export_csv/', views.export_inventory_csv, name='export_inventory_csv'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
