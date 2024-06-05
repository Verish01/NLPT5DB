from django.db import models

# Create your models here.
class Inventory(models.Model):
    product_name=models.CharField(max_length=30,null=False,blank=False)
    cost_per_item=models.DecimalField(max_digits=12,decimal_places=2,null=False,blank=False)
    quantity_in_stock=models.IntegerField(null=False,blank=False)
    quantity_sold=models.IntegerField(null=False,blank=False)
    sales=models.DecimalField(max_digits=12,decimal_places=2,null=False,blank=False)
    stock_date=models.DateField()
    photos=models.ImageField(upload_to="Inventph/")

    def __str__(self):
        return self.product_name

