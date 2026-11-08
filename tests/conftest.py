import numpy as np
import pytest

from searchrel.catalog import CatalogArrays, Product
from searchrel.engine import Engine, EngineConfig
from searchrel.index import InvertedIndex
from searchrel.retrieval import Retriever

SMALL = EngineConfig(n_products=2500, n_searches=6000, n_judged=90, folds=3, seed=3)


def make_product(i, title, leaf="rain_jacket", mid="Jackets", top="Clothing", brand="Quillback", color="blue",
                 sizes=("S", "M", "L"), material="nylon", gender="men", waterproof_mm=10000, price=150.0,
                 rating=4.2, reviews=30, inventory=10, units=20, description="", refers_to=""):
    return Product(id=f"p{i:05d}", title=title, brand=brand, leaf=leaf, mid=mid, top=top, color=color,
                   sizes=tuple(sizes), material=material, gender=gender, waterproof_mm=waterproof_mm,
                   weight_g=400, price=price, rating=rating, reviews=reviews, inventory=inventory, margin=0.4,
                   units_90d=units, description=description, refers_to=refers_to)


@pytest.fixture()
def tiny_catalog():
    return [
        make_product(0, "Quillback Ridge Men's Rain Jacket", color="blue", price=180),
        make_product(1, "Halvard Crest Women's Rain Jacket", brand="Halvard", gender="women", color="red", price=220),
        make_product(2, "Quillback Rain Jacket Repair Tape", leaf="gear_care", mid="Gear Care", top="Accessories",
                     sizes=(), material="plastic", gender="unisex", waterproof_mm=0, price=12, refers_to="rain_jacket"),
        make_product(3, "Verglas Summit Men's Down Jacket", brand="Verglas", leaf="insulated_jacket",
                     material="down", color="black", waterproof_mm=0, price=260),
        make_product(4, "Owlbrook Trek Backpacking Tent", brand="Owlbrook", leaf="tents", mid="Shelter",
                     top="Camping", sizes=(), gender="unisex", color="green", price=320,
                     description="A nylon tent for backpacking trips. Pairs well with our rain jackets."),
        make_product(5, "Quillback Vista Men's Rain Jacket, Navy", color="navy", sizes=("XL",), price=140,
                     inventory=0),
    ]


def build_retriever(products, reparse=None):
    idx = InvertedIndex.from_products(products)
    arr = CatalogArrays(products)
    return Retriever(idx, arr, np.log1p([p.units_90d for p in products]), reparse=reparse), idx, arr


@pytest.fixture(scope="session")
def engine():
    return Engine(SMALL)
