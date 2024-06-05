from django import forms
from .models import Inventory

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
