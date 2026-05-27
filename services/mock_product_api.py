# services/mock_product_api.py

import logging
from typing import Optional
from sqlalchemy import or_
from database.connection import SessionLocal
from database.models import Product

logger = logging.getLogger(__name__)


def search_products(filters: dict) -> list:
    """
    Searches products table using filters extracted
    by the LLM in extract_preferences node.
    Called by search_products tool node and
    broaden_search node.
    """
    db = SessionLocal()
    try:
        query = db.query(Product)

        if filters.get("category"):
            query = query.filter(
                Product.category.ilike(
                    f"%{filters['category']}%"
                )
            )

        if filters.get("max_price"):
            query = query.filter(
                Product.price <= float(filters["max_price"])
            )

        if filters.get("min_price"):
            query = query.filter(
                Product.price >= float(filters["min_price"])
            )

        if filters.get("brand"):
            brand = filters["brand"]
            if isinstance(brand, list):
                brand_filters = [
                    Product.brand.ilike(f"%{b}%")
                    for b in brand
                ]
                query = query.filter(or_(*brand_filters))
            else:
                query = query.filter(
                    Product.brand.ilike(f"%{brand}%")
                )

        if filters.get("min_rating"):
            query = query.filter(
                Product.rating >= float(filters["min_rating"])
            )

        if filters.get("in_stock") is True:
            query = query.filter(Product.in_stock == True)

        # product_type search — try exact match first, fall back to name keyword
        product_type = filters.get("product_type")
        if product_type and product_type != "other":
            type_query = query.filter(Product.product_type == product_type)
            type_results = (
                type_query
                .order_by(Product.rating.desc().nullslast())
                .limit(20)
                .all()
            )

            if type_results:
                return [
                    {
                        "product_id":       p.product_id,
                        "name":             p.name,
                        "category":         p.category,
                        "price":            float(p.price),
                        "actual_price":     float(p.actual_price) if p.actual_price else None,
                        "discount_percent": float(p.discount_percent) if p.discount_percent else 0,
                        "brand":            p.brand,
                        "rating":           float(p.rating) if p.rating else 0,
                        "rating_count":     p.rating_count or 0,
                        "in_stock":         p.in_stock,
                        "product_type":     p.product_type
                    }
                    for p in type_results
                ]

            # No exact product_type matches — return empty so broaden_search
            # can widen the filters (e.g. remove max_price) and retry
            return []

        elif filters.get("keyword"):
            query = query.filter(
                Product.name.ilike(f"%{filters['keyword']}%")
            )

        # order by rating descending — best products first
        query = query.order_by(
            Product.rating.desc().nullslast()
        )

        products = query.limit(20).all()

        return [
            {
                "product_id":       p.product_id,
                "name":             p.name,
                "category":         p.category,
                "price":            float(p.price),
                "actual_price":     float(p.actual_price) if p.actual_price else None,
                "discount_percent": float(p.discount_percent) if p.discount_percent else 0,
                "brand":            p.brand,
                "rating":           float(p.rating) if p.rating else 0,
                "rating_count":     p.rating_count or 0,
                "in_stock":         p.in_stock,
                "product_type":     p.product_type
            }
            for p in products
        ]

    except Exception as e:
        logger.error(f"Error searching products: {e}")
        return []
    finally:
        db.close()


def get_ratings(product_ids: list) -> dict:
    """
    Returns rating and rating_count for a list
    of product IDs.
    Called by fetch_ratings_tool inside
    product_enrichment subgraph.
    """
    db = SessionLocal()
    try:
        products = db.query(Product).filter(
            Product.product_id.in_(product_ids)
        ).all()

        return {
            p.product_id: {
                "rating":       float(p.rating) if p.rating else 0,
                "rating_count": p.rating_count or 0
            }
            for p in products
        }

    except Exception as e:
        logger.error(f"Error fetching ratings: {e}")
        return {}
    finally:
        db.close()


def get_specs(product_ids: list) -> dict:
    """
    Returns description and features for a list
    of product IDs.
    Called by fetch_specs_tool inside
    product_enrichment subgraph.
    Used by format_recommendations to show
    feature lists in the final response.
    """
    db = SessionLocal()
    try:
        products = db.query(Product).filter(
            Product.product_id.in_(product_ids)
        ).all()

        return {
            p.product_id: {
                "features":    p.features or [],
                "description": p.description or ""
            }
            for p in products
        }

    except Exception as e:
        logger.error(f"Error fetching specs: {e}")
        return {}
    finally:
        db.close()