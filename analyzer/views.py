import os
import pandas as pd
import logging
import json
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import tensorflow as tf
import joblib
from fuzzywuzzy import process
import tiktoken
import re

# print("settings.OPENAI_API_KEY..",settings.OPENAI_API_KEY)

# Suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set Pandas future behavior
pd.set_option('future.no_silent_downcasting', True)

# Category mapping
CATEGORY_MAPPING = {
    "demand": [
        "total sold - igr", "flat_sold - igr", "office_sold - igr", "others_sold - igr", "shop_sold - igr",
        "commercial_sold - igr", "other_sold - igr", "residential_sold - igr",
        "1.5bhk_sold - igr", "1bhk_sold - igr", "2.25bhk_sold - igr", "2.5bhk_sold - igr",
        "2.75bhk_sold - igr", "2bhk_sold - igr", "3bhk_sold - igr", "<1bhk_sold - igr",
        ">3bhk_sold - igr", "total carpet area consumed (sqft) - igr",
        "flat_carpet_area_consumed(in sqft) - igr", "office_carpet_area_consumed(in sqft) - igr",
        "others_carpet_area_consumed(in sqft) - igr", "shop_carpet_area_consumed(in sqft) - igr",
        "1.5bhk_carpet_area_consumed(in sqft) - igr", "1bhk_carpet_area_consumed(in sqft) - igr",
        "2.25bhk_carpet_area_consumed(in sqft) - igr", "2.5bhk_carpet_area_consumed(in sqft) - igr",
        "2.75bhk_carpet_area_consumed(in sqft) - igr", "2bhk_carpet_area_consumed(in sqft) - igr",
        "3bhk_carpet_area_consumed(in sqft) - igr", "<1bhk_carpet_area_consumed(in sqft) - igr",
        ">3bhk_carpet_area_consumed(in sqft) - igr", "residential_carpet_area_consumed(in sqft) - igr",
        "commercial_carpet_area_consumed(in sqft) - igr", "other_carpet_area_consumed(in sqft) - igr",
        "flat - avg carpet area (in sqft)", "others - avg carpet area (in sqft)",
        "office - avg carpet area (in sqft)", "shop - avg carpet area (in sqft)",
        "<1bhk - avg carpet area (in sqft)", "1bhk - avg carpet area (in sqft)",
        "1.5bhk - avg carpet area (in sqft)", "2bhk - avg carpet area (in sqft)",
        "2.25bhk - avg carpet area (in sqft)", "2.5bhk - avg carpet area (in sqft)",
        "2.75bhk - avg carpet area (in sqft)", "3bhk - avg carpet area (in sqft)",
        ">3bhk - avg carpet area (in sqft)",
        "loc_lat", "loc_lng", "residential share (in %)", "commercial share (in %)",
        "others share (in %)", "residential+commercial share (in %)",
        "total projects commenced", "total no.of rera entry",
        "total no. of blocks (based of projects)", "no. of residential projects",
        "no. of commercial projects", "no. of others projects", "residential+commercial",
        "no. of indusrial projects", "no. of d.a. registered","total area conveyed for developement"
    ],
    "supply": [
        "total units", "flat total", "shop total", "office total", "others total",
        "<1bhk total", "1bhk total", "1.5bhk total", "2bhk total", "2.25bhk total",
        "2.5bhk total", "2.75bhk total", "3bhk total", ">3bhk total",
        "total carpet area supplied (sqft)", "flat_carpet_area_supplied_rera_sqft",
        "shop_carpet_area_supplied_rera_sqft", "others_carpet_area_supplied_rera_sqft",
        "office_carpet_area_supplied_rera_sqft", "undefined flats_carpet_area_supplied_rera_sqft",
        "<1bhk_carpet_area_supplied_rera_sqft", "1bhk_carpet_area_supplied_rera_sqft",
        "1.5bhk_carpet_area_supplied_rera_sqft", "2bhk_carpet_area_supplied_rera_sqft",
        "2.25bhk_carpet_area_supplied_rera_sqft", "2.5bhk_carpet_area_supplied_rera_sqft",
        "2.75bhk_carpet_area_supplied_rera_sqft", "3bhk_carpet_area_supplied_rera_sqft",
        ">3bhk_carpet_area_supplied_rera_sqft",
        "loc_lat", "loc_lng", "residential share (in %)", "commercial share (in %)",
        "others share (in %)", "residential+commercial share (in %)",
        "total projects commenced", "total no.of rera entry",
        "total no. of blocks (based of projects)", "no. of residential projects",
        "no. of commercial projects", "no. of others projects", "residential+commercial",
        "no. of indusrial projects"
    ],
    "price": [
        "total_sales - igr", "flat- total agreement price", "office- total agreement price",
        "others- total agreement price", "shop- total agreement price",
        "1.5bhk- total agreement price", "1bhk- total agreement price",
        "2.25bhk- total agreement price", "2.5bhk- total agreement price",
        "2.75bhk- total agreement price", "2bhk- total agreement price",
        "3bhk- total agreement price", "<1bhk- total agreement price",
        ">3bhk- total agreement price", "commercial- total agreement price",
        "other- total agreement price", "residential- total agreement price",
        "flat - 50th percentile rate", "flat - 75th percentile rate", "flat - 90th percentile rate",
        "office - 50th percentile rate", "others - 50th percentile rate", "shop - 50th percentile rate",
        "office - 75th percentile rate", "others - 75th percentile rate", "shop - 75th percentile rate",
        "office - 90th percentile rate", "others - 90th percentile rate", "shop - 90th percentile rate",
        "commercial- avg agreement price", "other- avg agreement price", "residential- avg agreement price",
        "flat- avg agreement price", "office- avg agreement price", "others- avg agreement price",
        "shop- avg agreement price", "1.5bhk- avg agreement price", "1bhk- avg agreement price",
        "2.25bhk- avg agreement price", "2.5bhk- avg agreement price", "2.75bhk- avg agreement price",
        "2bhk- avg agreement price", "3bhk- avg agreement price", "<1bhk- avg agreement price",
        ">3bhk- avg agreement price", "flat - weighted average rate", "office - weighted average rate",
        "others - weighted average rate", "shop - weighted average rate",
        "flat - most prevailing rate - range", "office - most prevailing rate - range",
        "others - most prevailing rate - range", "shop - most prevailing rate - range",
        "loc_lat", "loc_lng", "residential share (in %)", "commercial share (in %)",
        "others share (in %)", "residential+commercial share (in %)",
        "total projects commenced", "total no.of rera entry",
        "total no. of blocks (based of projects)", "no. of residential projects",
        "no. of commercial projects", "no. of others projects", "residential+commercial",
        "no. of indusrial projects","total sales","rate_range_per_sqft"
    ],
    "demography": [
        "flat - pincode wise unit sold", "others - pincode wise unit sold",
        "office - pincode wise unit sold", "shop - pincode wise unit sold",
        "<1bhk - pincode wise unit sold", "1bhk - pincode wise unit sold",
        "1.5bhk - pincode wise unit sold", "2bhk - pincode wise unit sold",
        "2.25bhk - pincode wise unit sold", "2.5bhk - pincode wise unit sold",
        "2.75bhk - pincode wise unit sold", "3bhk - pincode wise unit sold",
        ">3bhk - pincode wise unit sold", "flat - age range wise unit sold",
        "others - age range wise unit sold", "office - age range wise unit sold",
        "shop - age range wise unit sold", "<1bhk - age range wise unit sold",
        "1bhk - age range wise unit sold", "1.5bhk - age range wise unit sold",
        "2bhk - age range wise unit sold", "2.25bhk - age range wise unit sold",
        "2.5bhk - age range wise unit sold", "2.75bhk - age range wise unit sold",
        "3bhk - age range wise unit sold", ">3bhk - age range wise unit sold",
        "flat - pincode wise total sales", "others - pincode wise total sales",
        "office - pincode wise total sales", "shop - pincode wise total sales",
        "<1bhk - pincode wise total sales", "1bhk - pincode wise total sales",
        "1.5bhk - pincode wise total sales", "2bhk - pincode wise total sales",
        "2.25bhk - pincode wise total sales", "2.5bhk - pincode wise total sales",
        "2.75bhk - pincode wise total sales", "3bhk - pincode wise total sales",
        ">3bhk - pincode wise total sales", "flat - pincode wise carpet area consumed in sqft",
        "others - pincode wise carpet area consumed in sqft",
        "office - pincode wise carpet area consumed in sqft",
        "shop - pincode wise carpet area consumed in sqft",
        "<1bhk - pincode wise carpet area consumed in sqft",
        "1bhk - pincode wise carpet area consumed in sqft",
        "1.5bhk - pincode wise carpet area consumed in sqft",
        "2bhk - pincode wise carpet area consumed in sqft",
        "2.25bhk - pincode wise carpet area consumed in sqft",
        "2.5bhk - pincode wise carpet area consumed in sqft",
        "2.75bhk - pincode wise carpet area consumed in sqft",
        "3bhk - pincode wise carpet area consumed in sqft",
        ">3bhk - pincode wise carpet area consumed in sqft",
        "flat - age range wise carpet area consumed in sqft",
        "others - age range wise carpet area consumed in sqft",
        "office - age range wise carpet area consumed in sqft",
        "shop - age range wise carpet area consumed in sqft", "top_buyer_pincode", "top10_buyer_in_locality",
        "loc_lat", "loc_lng", "residential share (in %)", "commercial share (in %)",
        "others share (in %)", "residential+commercial share (in %)",
        "total projects commenced", "total no.of rera entry",
        "total no. of blocks (based of projects)", "no. of residential projects",
        "no. of commercial projects", "no. of others projects", "residential+commercial",
        "no. of indusrial projects"
    ]
}

# User-provided column mapping for unstructured data
COLUMN_MAPPING = {
    "Location": ["final location"],
    "Segment Share (%)": ["residential share (in %)", "commercial share (in %)", "others share (in %)", "residential+commercial share (in %)"],
    "ROI from 2021 to 2024, computed as (2024 rate − 2021 rate) ÷ 2021 rate.": ["roi"],
    "Project commenced according to RERA": ["total projects commenced"],
    "total no of rera entry (phases) according to RERA": ["total no.of rera entry"],
    "no of blocks [tower, buildings] present according to RERA": ["total no. of blocks (based of projects)"],
    "No of rera entry commeneced in each project type according to RERA": ["no. of residential projects", "no. of commercial projects", "no. of others projects", "residential+commercial", "no. of indusrial projects"],
    "Total units sold": ["total sold - igr"],
    "Property type wise Units Sold": ["flat_sold - igr", "office_sold - igr", "others_sold - igr", "shop_sold - igr"],
    "Property category wise units sold": ["commercial_sold - igr", "other_sold - igr", "residential_sold - igr"],
    "BHK wise units sold": ["1.5bhk_sold - igr", "1bhk_sold - igr", "2.25bhk_sold - igr", "2.5bhk_sold - igr", "2.75bhk_sold - igr", "2bhk_sold - igr", "3bhk_sold - igr", "<1bhk_sold - igr", ">3bhk_sold - igr"],
    "Total Carpet Area consumed/Absorbed": ["total carpet area consumed (sqft) - igr"],
    "Property Type wise Carpet area consumed/absorbed in Sq ft": ["flat_carpet_area_consumed(in sqft) - igr", "office_carpet_area_consumed(in sqft) - igr", "others_carpet_area_consumed(in sqft) - igr", "shop_carpet_area_consumed(in sqft) - igr"],
    "BHK wise Carpet Area consumed (Square feet)": ["1.5bhk_carpet_area_consumed(in sqft) - igr", "1bhk_carpet_area_consumed(in sqft) - igr", "2.25bhk_carpet_area_consumed(in sqft) - igr", "2.5bhk_carpet_area_consumed(in sqft) - igr", "2.75bhk_carpet_area_consumed(in sqft) - igr", "2bhk_carpet_area_consumed(in sqft) - igr", "3bhk_carpet_area_consumed(in sqft) - igr", "<1bhk_carpet_area_consumed(in sqft) - igr", ">3bhk_carpet_area_consumed(in sqft) - igr"],
    "Property category wise Carpet area consumed/absorbed in Sq ft": ["residential_carpet_area_consumed(in sqft) - igr", "commercial_carpet_area_consumed(in sqft) - igr", "other_carpet_area_consumed(in sqft) - igr"],
    "Property type wise Average Carpet Area": ["flat - avg carpet area (in sqft)", "others - avg carpet area (in sqft)", "office - avg carpet area (in sqft)", "shop - avg carpet area (in sqft)"],
    "BHK wise Average Carpet Area": ["<1bhk - avg carpet area (in sqft)", "1bhk - avg carpet area (in sqft)", "1.5bhk - avg carpet area (in sqft)", "2bhk - avg carpet area (in sqft)", "2.25bhk - avg carpet area (in sqft)", "2.5bhk - avg carpet area (in sqft)", "2.75bhk - avg carpet area (in sqft)", "3bhk - avg carpet area (in sqft)", ">3bhk - avg carpet area (in sqft)"],
    "Property type wise Area range wise Units sold": ["others - area range unit sold", "flat - area range unit sold", "office - area range unit sold", "shop - area range unit sold"],
    "BHK wise Area range wise Units sold": ["1.5bhk - area range unit sold", "1bhk - area range unit sold", "2.5bhk - area range unit sold", "2bhk - area range unit sold", "3bhk - area range unit sold", "<1bhk - area range unit sold", ">3bhk - area range unit sold", "2.25bhk - area range unit sold", "2.75bhk - area range unit sold"],
    "Property type wise area range wise unit sold": ["others - area range total sales", "flat - area range total sales", "office - area range total sales", "shop - area range total sales"],
    "BHK wise area range wise unit sold": ["1.5bhk - area range total sales", "1bhk - area range total sales", "2.5bhk - area range total sales", "2bhk - area range total sales", "3bhk - area range total sales", "<1bhk - area range total sales", ">3bhk - area range total sales", "2.25bhk - area range total sales", "2.75bhk - area range total sales"],
    "Property type wise area range wise Carpet area consumed": ["others - area range carpet area consumed in sqft", "flat - area range carpet area consumed in sqft", "office - area range carpet area consumed in sqft", "shop - area range carpet area consumed in sqft"],
    "BHK wise area range wise carpet area consumed": ["1.5bhk - area range carpet area consumed in sqft", "1bhk - area range carpet area consumed in sqft", "2.5bhk - area range carpet area consumed in sqft", "2bhk - area range carpet area consumed in sqft", "3bhk - area range carpet area consumed in sqft", "<1bhk - area range carpet area consumed in sqft", ">3bhk - area range carpet area consumed in sqft", "2.25bhk - area range carpet area consumed in sqft", "2.75bhk - area range carpet area consumed in sqft"],
    "Property type wise Top 10 Buyer Pincode units bought": ["flat - pincode wise unit sold", "others - pincode wise unit sold", "office - pincode wise unit sold", "shop - pincode wise unit sold"],
    "BHK wise Top 10 Buyer Pincode units bought": ["<1bhk - pincode wise unit sold", "1bhk - pincode wise unit sold", "1.5bhk - pincode wise unit sold", "2bhk - pincode wise unit sold", "2.25bhk - pincode wise unit sold", "2.5bhk - pincode wise unit sold", "2.75bhk - pincode wise unit sold", "3bhk - pincode wise unit sold", ">3bhk - pincode wise unit sold"],
    "Property type wise Age Range wise units bought": ["flat - age range wise unit sold", "others - age range wise unit sold", "office - age range wise unit sold", "shop - age range wise unit sold"],
    "BHK wise Age Range wise units bought": ["<1bhk - age range wise unit sold", "1bhk - age range wise unit sold", "1.5bhk - age range wise unit sold", "2bhk - age range wise unit sold", "2.25bhk - age range wise unit sold", "2.5bhk - age range wise unit sold", "2.75bhk - age range wise unit sold", "3bhk - age range wise unit sold", ">3bhk - age range wise unit sold"],
    "Property type wise Top 10 Buyer Pincode Total sales": ["flat - pincode wise total sales", "others - pincode wise total sales", "office - pincode wise total sales", "shop - pincode wise total sales"],
    "BHK wise Top 10 Buyer Pincode Total sales": ["<1bhk - pincode wise total sales", "1bhk - pincode wise total sales", "1.5bhk - pincode wise total sales", "2bhk - pincode wise total sales", "2.25bhk - pincode wise total sales", "2.5bhk - pincode wise total sales", "2.75bhk - pincode wise total sales", "3bhk - pincode wise total sales", ">3bhk - pincode wise total sales"],
    "Property type wise Top 10 Buyer Pincode Carpet Area Consumed": ["flat - pincode wise carpet area consumed in sqft", "others - pincode wise carpet area consumed in sqft", "office - pincode wise carpet area consumed in sqft", "shop - pincode wise carpet area consumed in sqft"],
    "BHK wise Top 10 Buyer Pincode Carpet Area Consumed": ["<1bhk - pincode wise carpet area consumed in sqft", "1bhk - pincode wise carpet area consumed in sqft", "1.5bhk - pincode wise carpet area consumed in sqft", "2bhk - pincode wise carpet area consumed in sqft", "2.25bhk - pincode wise carpet area consumed in sqft", "2.5bhk - pincode wise carpet area consumed in sqft", "2.75bhk - pincode wise carpet area consumed in sqft", "3bhk - pincode wise carpet area consumed in sqft", ">3bhk - pincode wise carpet area consumed in sqft"],
    "Property type wise Age Range wise Carpet Area Consumed": ["flat - age range wise carpet area consumed in sqft", "others - age range wise carpet area consumed in sqft", "office - age range wise carpet area consumed in sqft", "shop - age range wise carpet area consumed in sqft"],
    "Top buyer pincode": ["top_buyer_pincode"],
    "Top 10 Buyer Pincode units bought": ["top10_buyer_in_locality"],
    "BHK wise Age Range wise Total sales": ["<1bhk - age range wise total sales", "1bhk - age range wise total sales", "1.5bhk - age range wise total sales", "2bhk - age range wise total sales", "2.25bhk - age range wise total sales", "2.5bhk - age range wise total sales", "2.75bhk - age range wise total sales", "3bhk - age range wise total sales", ">3bhk - age range wise total sales"],
    "Property type wise Age Range wise Total sales": ["flat - age range wise total sales", "others - age range wise total sales", "office - age range wise total sales", "shop - age range wise total sales"],
    "BHK wise Age Range wise Carpet Area Consumed": ["<1bhk - age range wise carpet area consumed in sqft", "1bhk - age range wise carpet area consumed in sqft", "1.5bhk - age range wise carpet area consumed in sqft", "2bhk - age range wise carpet area consumed in sqft", "2.25bhk - age range wise carpet area consumed in sqft", "2.5bhk - age range wise carpet area consumed in sqft", "2.75bhk - age range wise carpet area consumed in sqft", "3bhk - age range wise carpet area consumed in sqft", ">3bhk - age range wise carpet area consumed in sqft"],
    "Property Type wise Average Agreement Price": ["flat- avg agreement price", "office- avg agreement price", "others- avg agreement price", "shop- avg agreement price"],
    "Total sales (Agreement Price)": ["total_sales - igr"],
    "Property Type wise Total Sales (Agreement Price)": ["flat- total agreement price", "office- total agreement price", "others- total agreement price", "shop- total agreement price"],
    "BHK wise Total Sales (Agreement Price)": ["1.5bhk- total agreement price", "1bhk- total agreement price", "2.25bhk- total agreement price", "2.5bhk- total agreement price", "2.75bhk- total agreement price", "2bhk- total agreement price", "3bhk- total agreement price", "<1bhk- total agreement price", ">3bhk- total agreement price"],
    "BHK wise Average Agreement Price": ["1.5bhk- avg agreement price", "1bhk- avg agreement price", "2.25bhk- avg agreement price", "2.5bhk- avg agreement price", "2.75bhk- avg agreement price", "2bhk- avg agreement price", "3bhk- avg agreement price", "<1bhk- avg agreement price", ">3bhk- avg agreement price"],
    "Property category wise Total sales (Agreement Price)": ["commercial- total agreement price", "other- total agreement price", "residential- total agreement price"],
    "Property category wise Average Agreement Price": ["commercial- avg agreement price", "other- avg agreement price", "residential- avg agreement price"],
    "Property type wise Weighted Average Rate (Rupees per sq feet)": ["flat - weighted average rate", "office - weighted average rate", "others - weighted average rate", "shop - weighted average rate"],
    "Property Type wise Most Prevailing Rate Range (Rupees per sq foot)": ["flat - most prevailing rate - range", "office - most prevailing rate - range", "others - most prevailing rate - range", "shop - most prevailing rate - range"],
    "Property type wise Rate range wise Units sold": ["office - rate range unit sold", "shop - rate range unit sold", "flat - rate range unit sold", "others - rate range unit sold"],
    "Property type wise Agreement range wise unit sold": ["flat - agreement price range unit sold", "others - agreement price range unit sold", "office - agreement price range unit sold", "shop - agreement price range unit sold"],
    "BHK wise rate range wise Units sold": ["1.5bhk - rate range unit sold", "1bhk - rate range unit sold", "2.5bhk - rate range unit sold", "2bhk - rate range unit sold", "3bhk - rate range unit sold", "<1bhk - rate range unit sold", ">3bhk - rate range unit sold", "2.25bhk - rate range unit sold", "2.75bhk - rate range unit sold"],
    "BHK wise agreement price range wise Units sold": ["1.5bhk - agreement price range unit sold", "1bhk - agreement price range unit sold", "2.5bhk - agreement price range unit sold", "2bhk - agreement price range unit sold", "3bhk - agreement price range unit sold", "<1bhk - agreement price range unit sold", ">3bhk - agreement price range unit sold", "2.25bhk - agreement price range unit sold", "2.75bhk - agreement price range unit sold"],
    "Property Type wise Rate Range wise Total sales": ["flat - rate range total sales", "others - rate range total sales", "office - rate range total sales", "shop - rate range total sales"],
    "Property type wise Agreement Price Range wise Total sales": ["others - agreement price range total sales", "flat - agreement price range total sales", "office - agreement price range total sales", "shop - agreement price range total sales"],
    "BHK wise Rate Range wise Total sales": ["1.5bhk - rate range total sales", "1bhk - rate range total sales", "2.5bhk - rate range total sales", "2bhk - rate range total sales", "3bhk - rate range total sales", "<1bhk - rate range total sales", ">3bhk - rate range total sales", "2.25bhk - rate range total sales", "2.75bhk - rate range total sales"],
    "BHK wise agreement range wise Total Sales": ["1.5bhk - agreement price range total sales", "1bhk - agreement price range total sales", "2.5bhk - agreement price range total sales", "2bhk - agreement price range total sales", "3bhk - agreement price range total sales", "<1bhk - agreement price range total sales", ">3bhk - agreement price range total sales", "2.25bhk - agreement price range total sales", "2.75bhk - agreement price range total sales"],
    "Property Type wise Rate Range wise Carpet Area consumed (Square feet)": ["others - rate range carpet area consumed in sqft", "flat - rate range carpet area consumed in sqft", "office - rate range carpet area consumed in sqft", "shop - rate range carpet area consumed in sqft"],
    "Agreement price wise range Carpet area consumed": ["others - agreement price range carpet area consumed in sqft", "flat - agreement price range carpet area consumed in sqft", "office - agreement price range carpet area consumed in sqft", "shop - agreement price range carpet area consumed in sqft"],
    "BHK wise rate range wise carpet area consumed": ["1bhk - rate range carpet area consumed in sqft", "2bhk - rate range carpet area consumed in sqft", "3bhk - rate range carpet area consumed in sqft", "<1bhk - rate range carpet area consumed in sqft", ">3bhk - rate range carpet area consumed in sqft", "1.5bhk - rate range carpet area consumed in sqft", "2.5bhk - rate range carpet area consumed in sqft", "2.25bhk - rate range carpet area consumed in sqft", "2.75bhk - rate range carpet area consumed in sqft"],
    "bhk wise Agreement price range wise Carpet area consumed": ["1bhk - agreement price range carpet area consumed in sqft", "2bhk - agreement price range carpet area consumed in sqft", "3bhk - agreement price range carpet area consumed in sqft", "<1bhk - agreement price range carpet area consumed in sqft", ">3bhk - agreement price range carpet area consumed in sqft", "1.5bhk - agreement price range carpet area consumed in sqft", "2.5bhk - agreement price range carpet area consumed in sqft", "2.25bhk - agreement price range carpet area consumed in sqft", "2.75bhk - agreement price range carpet area consumed in sqft"],
    "total units supplied according to RERA": ["total units"],
    "Property Type wise total units supplied": ["flat total", "shop total", "office total", "others total"],
    "BHK wise total units supplied": ["<1bhk total", "1bhk total", "1.5bhk total", "2bhk total", "2.25bhk total", "2.5bhk total", "2.75bhk total", "3bhk total", ">3bhk total"],
    "Total Carpet Area supplied(In sq ft)": ["total carpet area supplied (sqft)"],
    "Property type wise Total Carpet Area supplied(in sq ft)": ["flat_carpet_area_supplied_rera_sqft", "shop_carpet_area_supplied_rera_sqft", "others_carpet_area_supplied_rera_sqft", "office_carpet_area_supplied_rera_sqft", "undefined flats_carpet_area_supplied_rera_sqft"],
    "BHK wise carpet area supplied in sqft according to RERA": ["<1bhk_carpet_area_supplied_rera_sqft", "1bhk_carpet_area_supplied_rera_sqft", "1.5bhk_carpet_area_supplied_rera_sqft", "2bhk_carpet_area_supplied_rera_sqft", "2.25bhk_carpet_area_supplied_rera_sqft", "2.5bhk_carpet_area_supplied_rera_sqft", "2.75bhk_carpet_area_supplied_rera_sqft", "3bhk_carpet_area_supplied_rera_sqft", ">3bhk_carpet_area_supplied_rera_sqft", "undefined flats_carpet_area_supplied_rera_sqft"],
    "Property Type wise Percentile Rate (Rupees per sq foot)": ["flat - 50th percentile rate", "flat - 75th percentile rate", "flat - 90th percentile rate", "office - 50th percentile rate", "others - 50th percentile rate", "shop - 50th percentile rate", "office - 75th percentile rate", "others - 75th percentile rate", "shop - 75th percentile rate", "office - 90th percentile rate", "others - 90th percentile rate", "shop - 90th percentile rate"],
    "No of Development agreement (DA) Regiesterd and total area conveyed": ["no of da registered", "total area conveyed"],
    "total sales in developement agreement (DA)": ["total sales in da"],
    "Most prevailing rate for Development Agreement (DA)": ["most prevailing rate for da"],
}

def normalize_colname(name):
    # Lowercase, strip, replace multiple spaces/dashes with single space, remove leading/trailing spaces
    return re.sub(r'[-\s]+', ' ', name.strip().lower())

def get_mapping_key(column_name):
    norm_col = normalize_colname(column_name)
    for key, values in COLUMN_MAPPING.items():
        for v in values:
            if normalize_colname(v) == norm_col:
                return key
    return "UnknownMapping"

# File paths
try:
    excel_path = os.path.join(settings.BASE_DIR, "SampleR.xlsx")
    pickle_path = os.path.join(settings.BASE_DIR, "SampleR.pkl")
except Exception as e:
    logger.error(f"Failed to locate data files: {e}")

# Load and clean data
def load_and_clean_data(excel_path, pickle_path, villages=None, years=None, category=None):
    try:
        if Path(pickle_path).exists():
            df = joblib.load(pickle_path)
            logger.info(f"Pickle file loaded. Shape: {df.shape}")
        else:
            df = pd.read_excel(excel_path)
            joblib.dump(df, pickle_path, compress=3)
            logger.info(f"Excel file loaded and saved as pickle. Shape: {df.shape}")
        
        df["final location"] = df["final location"].str.strip().str.lower()
        available_villages = df["final location"].unique()
        logger.info(f"Available villages: {available_villages}")
        
        if villages:
            df = df[df["final location"].isin([v.lower() for v in villages])]
            if df.empty:
                logger.error(f"No data for villages {villages}")
                return None, None
            logger.info(f"Filtered data for villages {villages}. Shape: {df.shape}")
        
        years = [y for y in (years or [2020, 2021, 2022, 2023, 2024]) if 2020 <= y <= 2024]
        if years:
            df = df[df["year"].isin(years)]
            logger.info(f"Filtered data for years {years}. Shape: {df.shape}")
        
        df = df.sort_values(by=["final location", "year"])
        
        if category and category != "general":
            relevant_columns = ["final location", "year"]
            category_keywords = CATEGORY_MAPPING.get(category.lower(), [])
            for col in df.columns:
                if any(keyword in col.lower() for keyword in category_keywords):
                    relevant_columns.append(col)
            relevant_columns = list(dict.fromkeys(relevant_columns))
            df = df[[col for col in relevant_columns if col in df.columns]]
            logger.info(f"Filtered columns for category '{category}'. Shape: {df.shape}")
        
        defaults = {
            "year": 2020,
            "total sold - igr": 0,
            "1bhk_sold - igr": 0,
            "flat total": 0,
            "shop total": 0,
            "office total": 0,
            "others total": 0,
            "1bhk total": 0,
            "<1bhk total": 0
        }
        
        df = df.infer_objects(copy=False).fillna({col: defaults.get(col, 0) for col in df.columns})
        logger.info(f"Final data shape: {df.shape}, columns: {df.columns.tolist()}")
        return df, defaults
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None

# Get village names
def get_village_names():
    try:
        if Path(pickle_path).exists():
            df = joblib.load(pickle_path)
        else:
            df = pd.read_excel(excel_path)
            joblib.dump(df, pickle_path, compress=3)
        villages = sorted(df["final location"].str.strip().str.lower().unique())
        if not villages:
            logger.error("No villages found in SampleR.xlsx")
            return []
        logger.info(f"Available villages: {villages}")
        return villages
    except Exception as e:
        logger.error(f"Failed to load village names: {e}")
        return []

# Format data for a single column
def format_column_data(df, column, village1, village2, defaults, years=[2020, 2021, 2022, 2023, 2024]):
    lines = []
    mapping_key = get_mapping_key(column)
    for village in [village1.lower(), village2.lower()]:
        village_df = df[df["final location"] == village]
        for year in years:
            year_df = village_df[village_df["year"] == year]
            value = year_df[column].iloc[0] if not year_df.empty and column in year_df.columns else defaults.get(column, 'N/A')
            # New format: location_year_mappingkey_columnname: value
            lines.append(f"{village}_{year}_{mapping_key}_{column}: {value}")
    return "\n".join(lines)

# Create documents organized by column
def create_documents(df, village1, village2, defaults, include_columns=None):
    documents = []
    include_columns = include_columns if include_columns else [col for col in df.columns if col not in ['final location', 'year']]
    years = [2020, 2021, 2022, 2023, 2024]
    
    for column in include_columns:
        if column in df.columns:
            content = format_column_data(df, column, village1, village2, defaults, years)
            documents.append(
                Document(
                    page_content=content,
                    metadata={'column': column, 'village1': village1.lower(), 'village2': village2.lower()}
                )
            )
    
    logger.info(f"Created {len(documents)} documents for villages {village1} and {village2}")
    return documents

# Token counting
def count_tokens(text, model="gpt-4o-mini"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return 0

# LLM setup
try:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=8000,
        max_retries=3,
        api_key="your_openai_api_key"
    )
    logger.info("OpenAI gpt-4o-mini LLM initialized")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI: {e}")
    llm = None

# Prompt template
prompt_template = """You are a real estate analyst. The user has requested: "{question}" for {village1} and {village2} (2020–2024). The data from SampleR.xlsx contains metrics for 
the '{category}' category, prefixed by village and year (e.g., 'Aundh 2020 total units'). Use the data to compare {village1} and {village2}. If direct metrics are missing, infer trends or note limitations. Provide a concise text-based analysis.

Data:
{context}

Provide the analysis focusing on trends and insights.
"""

def home(request):
    """Home page view"""
    villages = get_village_names()
    context = {
        'villages': villages,
        'categories': ['Demand', 'Supply', 'Price', 'Demography']
    }
    return render(request, 'analyzer/home.html', context)

@csrf_exempt
def get_villages(request):
    """API endpoint to get village names"""
    villages = get_village_names()
    return JsonResponse({'villages': villages})

@csrf_exempt
def compare_villages(request):
    """API endpoint to compare villages"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            village1 = data.get('village1', '').strip()
            village2 = data.get('village2', '').strip()
            categories = data.get('categories', [])
            query = data.get('query', '').strip()
            stream = request.GET.get('stream', 'false').lower() == 'true'
            # Validation
            if not village1 or not village2:
                return JsonResponse({'error': 'Both villages are required'}, status=400)
            if village1.lower() == village2.lower():
                return JsonResponse({'error': 'Villages must be different'}, status=400)
            if not categories:
                return JsonResponse({'error': 'At least one category is required'}, status=400)
            if not llm:
                return JsonResponse({'error': 'OpenAI API not configured'}, status=500)
            # Set default query if not provided
            if not query:
                query = f"Compare {', '.join(categories).lower()} metrics for {village1} and {village2}"
            logger.info(f"Query: '{query}'")
            # Load and clean data for all categories together
            all_categories = [cat.lower() for cat in categories]
            combined_include_columns = set()
            for category in all_categories:
                combined_include_columns.update(CATEGORY_MAPPING.get(category, []))
            combined_include_columns = list(combined_include_columns)
            df, defaults = load_and_clean_data(excel_path, pickle_path, villages=[village1, village2], years=[2020, 2021, 2022, 2023, 2024], category=None)
            if df is None or df.empty:
                return JsonResponse({'error': f'No data for {village1} or {village2} in 2020-2024'}, status=400)
            # Filter columns based on combined categories
            relevant_columns = ["final location", "year"]
            for col in df.columns:
                if any(keyword in col.lower() for keyword in combined_include_columns):
                    relevant_columns.append(col)
            relevant_columns = list(dict.fromkeys(relevant_columns))
            df = df[[col for col in relevant_columns if col in df.columns]]
            logger.info(f"Filtered data for combined categories. Shape: {df.shape}")
            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("HuggingFace embeddings initialized")
            vector_store_path = f"faiss_index_{village1}_{village2}_combined"
            if os.path.exists(vector_store_path):
                import shutil
                shutil.rmtree(vector_store_path)
                logger.info(f"Cleared FAISS index at {vector_store_path}")
            documents = create_documents(df, village1, village2, defaults, combined_include_columns)
            if not documents:
                return JsonResponse({'error': 'Failed to create documents'}, status=500)
            vector_store = FAISS.from_documents(documents, embeddings)
            vector_store.save_local(vector_store_path)
            logger.info(f"FAISS index created with {len(documents)} documents for combined categories")
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": len(combined_include_columns)})
            # Get relevant documents
            docs = retriever.invoke(query.strip())
            if not docs:
                return JsonResponse({'error': 'No relevant data found'}, status=400)
            context = "\n\n".join([doc.page_content.strip() for doc in docs])
            logger.info(f"Retrieved {len(docs)} documents")
            formatted_prompt = prompt_template.format(
                question=query.strip(),
                village1=village1,
                village2=village2,
                category=", ".join(categories).lower(),
                context=context
            )
            input_tokens = count_tokens(formatted_prompt)
            logger.info(f"Input tokens: {input_tokens}")
            # Streaming response
            if stream:
                def stream_llm():
                    for chunk in llm.stream(formatted_prompt):
                        chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                        yield chunk_text
                return StreamingHttpResponse(stream_llm(), content_type='text/plain')
            # Non-streaming (default)
            response = llm.invoke(formatted_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            output_tokens = count_tokens(response_text)
            return JsonResponse({
                'analysis': response_text,
                'tokens': {
                    'input': input_tokens,
                    'output': output_tokens,
                    'total': input_tokens + output_tokens
                },
                'sources': [doc.page_content for doc in docs]
            })
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logger.error(f"Error in comparison: {e}")
            return JsonResponse({'error': f'Failed to process request: {str(e)}'}, status=500)
    return JsonResponse({'error': 'Only POST method allowed'}, status=405) 