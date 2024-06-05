from django.shortcuts import get_object_or_404, redirect, render
from django.http import HttpResponse
import csv
import os
import django
from .models import Inventory
from .forms import InventoryForm 
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

def inventory_list(request):
    inventory_items = Inventory.objects.all()
    return render(request, 'inventory_list.html', {'inventory_items': inventory_items})

def edit_item(request, id):
    item = get_object_or_404(Inventory, id=id)
    if request.method == "POST":
        form = InventoryForm(request.POST, instance=item)
        if form.is_valid():
            form.save()
            return redirect('inventory_list')
    else:
        form = InventoryForm(instance=item)
    return render(request, 'edit_item.html', {'form': form})

def delete_item(request, id):
    item = get_object_or_404(Inventory, id=id)
    if request.method == "POST":
        item.delete()
        return redirect('inventory_list')
    return render(request, 'confirm_delete.html', {'item': item})

def add_item(request):
    if request.method == 'POST':
        form = InventoryForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('inventory_list')  # Redirect to inventory list view after successful addition
    else:
        form = InventoryForm()
    return render(request, 'add_item.html', {'form': form})

def inventory_visualization(request):
    return render(request, 'gradio.html')

def export_inventory_csv(request):
    # Create the HttpResponse object with the appropriate CSV header.
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="inventory.csv"'

    writer = csv.writer(response)
    # Write the header row
    writer.writerow(['Product Name', 'Quantity in Stock', 'Cost per Item', 'Quantity Sold', 'Sales', 'Stock Date'])

    # Write data rows
    for item in Inventory.objects.all():
        writer.writerow([item.product_name, item.quantity_in_stock, item.cost_per_item, item.quantity_sold, item.sales, item.stock_date])

    return response
