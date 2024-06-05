from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse
from django import forms
from .models import Inventory
import csv
import os
import sqlite3
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

# Define InventoryForm class
class InventoryForm(forms.ModelForm):
    class Meta:
        model = Inventory  # Specify the model your form is based on
        fields = ['product_name', 'quantity_in_stock', 'cost_per_item', 'quantity_sold', 'sales', 'stock_date', 'photos']
        # Specify the fields you want to include in your form
        widgets = {
            'stock_date': forms.DateInput(attrs={'type': 'date'}),
        }

# Define QueryForm class
class QueryForm(forms.Form):
    query = forms.CharField(label='Enter your natural language query', max_length=200)

# Load the ML model and tokenizer
model_path = 'gaussalgo/T5-LM-Large-text2sql-spider'
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def query_database(db_path, query):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return results
    except sqlite3.Error as e:
        return str(e)

def get_sql_query(question, schema):
    input_text = f"Question: {question} Schema: {schema}"
    model_inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**model_inputs, max_length=512)
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_text[0]

def query_view(request):
    if request.method == 'POST':
        form = QueryForm(request.POST)
        if form.is_valid():
            question = form.cleaned_data['query']
            schema = '''"myapps_inventory" "DATABASE" 
                        "product_ID" int, 
                        "product_name" text, 
                        "quantity_in_stock" int, 
                        "cost_per_item" float, 
                        primary key: "product_ID"'''
            
            sql_query = get_sql_query(question, schema)
            db_path = os.path.join('ims', 'db.sqlite3')
            results = query_database(db_path, sql_query)
            
            return render(request, 'results.html', {'results': results, 'query': sql_query})
    else:
        form = QueryForm()
    return render(request, 'index.html', {'form': form})

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
