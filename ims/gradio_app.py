import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import os
import django

# Set the DJANGO_SETTINGS_MODULE environment variable
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ims.settings")

# Initialize Django
django.setup()

from myapps.models import Inventory  # Import models after setting up Django

def plot_sales():
    # Fetch data from the Inventory model
    inventory_items = Inventory.objects.all().values()

    data = {
        "product_name": [item['product_name'] for item in inventory_items],
        "quantity_in_stock": [item['quantity_in_stock'] for item in inventory_items],
        "cost_per_item": [item['cost_per_item'] for item in inventory_items],
        "quantity_sold": [item['quantity_sold'] for item in inventory_items],
        "sales": [item['sales'] for item in inventory_items],
        "stock_date": [item['stock_date'] for item in inventory_items],
    }

    df = pd.DataFrame(data)
    df['sales'] = pd.to_numeric(df['sales'], errors='coerce')

    fig, axes = plt.subplots(3, 1, figsize=(10, 18))

    # Plot sales by product
    df.set_index('product_name', inplace=True)
    df['sales'].plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Sales by Product')
    axes[0].set_xlabel('Product Name')
    axes[0].set_ylabel('Sales')

    # Plot quantities in stock
    df['quantity_in_stock'].plot(kind='bar', ax=axes[1], color='green')
    axes[1].set_title('Quantities in Stock by Product')
    axes[1].set_xlabel('Product Name')
    axes[1].set_ylabel('Quantity in Stock')

    # Plot most sold products
    df['quantity_sold'].plot(kind='bar', ax=axes[2], color='orange')
    axes[2].set_title('Most Sold Products')
    axes[2].set_xlabel('Product Name')
    axes[2].set_ylabel('Quantity Sold')

    plt.tight_layout()
    return fig

demo = gr.Interface(
    fn=plot_sales,
    inputs=[],
    outputs=gr.Plot()
)

if __name__ == "__main__":
    demo.launch(share=True)
