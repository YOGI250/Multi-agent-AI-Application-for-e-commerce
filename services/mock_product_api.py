# services/mock_product_api.py

import logging
from sqlalchemy import or_
from database.connection import SessionLocal
from database.models import Product
from monitoring.metrics import record_error

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

        # category filter intentionally removed — LLM often assigns wrong category
        # (e.g. keyboards → Electronics instead of Computers and Accessories),
        # causing correct products to be excluded. product_type is precise enough.

        if filters.get("max_price"):
            query = query.filter(Product.price <= float(filters["max_price"]))

        if filters.get("min_price"):
            query = query.filter(Product.price >= float(filters["min_price"]))

        if filters.get("brand"):
            brand = filters["brand"]
            if isinstance(brand, list):
                brand_filters = [Product.brand.ilike(f"%{b}%") for b in brand]
                query = query.filter(or_(*brand_filters))
            else:
                query = query.filter(Product.brand.ilike(f"%{brand}%"))

        if filters.get("min_rating"):
            query = query.filter(Product.rating >= float(filters["min_rating"]))

        if filters.get("in_stock") is True:
            query = query.filter(Product.in_stock == True)

        # product_type search — try exact match first, fall back to name keyword
        product_type = filters.get("product_type")
        if product_type and product_type != "other":
            # normalize plural → singular (keyboards→keyboard, cables→cable)
            singular = product_type.rstrip("s") if product_type.endswith("s") else product_type
            type_query = query.filter(
                or_(
                    Product.product_type.ilike(product_type),
                    Product.product_type.ilike(singular),
                    Product.product_type.ilike(singular + "s"),
                )
            )
            type_results = type_query.order_by(Product.rating.desc().nullslast()).limit(20).all()

            if type_results:
                return [
                    {
                        "product_id": p.product_id,
                        "name": p.name,
                        "category": p.category,
                        "price": float(p.price),
                        "actual_price": float(p.actual_price) if p.actual_price else None,
                        "discount_percent": float(p.discount_percent) if p.discount_percent else 0,
                        "brand": p.brand,
                        "rating": float(p.rating) if p.rating else 0,
                        "rating_count": p.rating_count or 0,
                        "in_stock": p.in_stock,
                        "product_type": p.product_type,
                    }
                    for p in type_results
                ]

            # No exact product_type match — user may have used a colloquial or
            # regional term (e.g. "geyser" for water_heater, "vacuum cleaner"
            # for vacuum). Fall back to searching product names so these queries
            # find results without needing a hardcoded synonym map.
            name_results = (
                query.filter(Product.name.ilike(f"%{singular}%"))
                .order_by(Product.rating.desc().nullslast())
                .limit(20)
                .all()
            )
            if name_results:
                return [
                    {
                        "product_id": p.product_id,
                        "name": p.name,
                        "category": p.category,
                        "price": float(p.price),
                        "actual_price": float(p.actual_price) if p.actual_price else None,
                        "discount_percent": float(p.discount_percent) if p.discount_percent else 0,
                        "brand": p.brand,
                        "rating": float(p.rating) if p.rating else 0,
                        "rating_count": p.rating_count or 0,
                        "in_stock": p.in_stock,
                        "product_type": p.product_type,
                    }
                    for p in name_results
                ]

            return []

        elif filters.get("keyword"):
            query = query.filter(Product.name.ilike(f"%{filters['keyword']}%"))

        # Exclude 'other' from generic searches — these are products we don't sell
        # (phones, TVs, calculators, etc.) intentionally classified out of search.
        if not filters.get("product_type"):
            query = query.filter(Product.product_type != "other")

        # order by rating descending — best products first
        query = query.order_by(Product.rating.desc().nullslast())

        products = query.limit(20).all()

        return [
            {
                "product_id": p.product_id,
                "name": p.name,
                "category": p.category,
                "price": float(p.price),
                "actual_price": float(p.actual_price) if p.actual_price else None,
                "discount_percent": float(p.discount_percent) if p.discount_percent else 0,
                "brand": p.brand,
                "rating": float(p.rating) if p.rating else 0,
                "rating_count": p.rating_count or 0,
                "in_stock": p.in_stock,
                "product_type": p.product_type,
            }
            for p in products
        ]

    except Exception as e:
        logger.error(f"Error searching products: {e}")
        record_error(error_type=type(e).__name__, agent_used="product_agent")
        return []
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
        products = db.query(Product).filter(Product.product_id.in_(product_ids)).all()

        return {p.product_id: {"features": p.features or [], "description": p.description or ""} for p in products}

    except Exception as e:
        logger.error(f"Error fetching specs: {e}")
        record_error(error_type=type(e).__name__, agent_used="product_agent")
        return {}
    finally:
        db.close()
