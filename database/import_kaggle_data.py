# database/import_kaggle_data.py

import pandas as pd
import uuid
import random
import re
import logging
from datetime import datetime, timedelta, date
from database.connection import SessionLocal
from database.models import Product, Order, CarrierTracking

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# CONSTANTS
# ==========================================
CARRIERS = ["BlueDart", "Delhivery", "DTDC"]
CARRIER_PREFIXES = {"BlueDart": "BD", "Delhivery": "DL", "DTDC": "DT"}
DEMO_USER_ID = "demo_user_001"


# ==========================================
# HELPERS
# ==========================================
def clean_price(price_str):
    """Convert '₹1,299' to 1299.00"""
    try:
        cleaned = re.sub(r"[₹,\s]", "", str(price_str))
        return round(float(cleaned), 2)
    except Exception:
        return None


def clean_rating(rating_str):
    """Convert '4.2' to 4.2, handle '|' separated values"""
    try:
        val = str(rating_str).split("|")[0].strip()
        rating = float(val)
        if 0 <= rating <= 5:
            return round(rating, 1)
        return None
    except Exception:
        return None


def clean_rating_count(count_str):
    """Convert '24,269' to 24269"""
    try:
        cleaned = re.sub(r"[,\s]", "", str(count_str))
        return int(cleaned)
    except Exception:
        return 0


def clean_discount(discount_str):
    """Convert '64%' to 64.0"""
    try:
        cleaned = str(discount_str).replace("%", "").strip()
        return round(float(cleaned), 2)
    except Exception:
        return 0.0


def extract_brand(product_name):
    """Extract brand from the first word of product name"""
    try:
        brand = str(product_name).split()[0].strip(",").strip()
        return brand[:50]
    except Exception:
        return "Unknown"


def clean_category(category_str):
    """Take first level of category: 'Computers&Accessories|...' -> 'Computers'"""
    try:
        first = str(category_str).split("|")[0].strip()
        first = first.replace("&", " and ")
        return first[:100]
    except Exception:
        return "General"


def extract_features(about_product):
    """Extract bullet points as feature list"""
    try:
        points = str(about_product).split("|")
        features = []
        for point in points[:5]:
            point = point.strip()
            if len(point) > 10:
                first_sentence = point.split(".")[0][:100]
                features.append(first_sentence)
        return features
    except Exception:
        return []


def generate_tracking_number(carrier):
    """Generate a realistic tracking number"""
    prefix = CARRIER_PREFIXES.get(carrier, "XX")
    number = "".join([str(random.randint(0, 9)) for _ in range(9)])
    return f"{prefix}{number}"


def generate_tracking_events(status, order_date, expected_delivery):
    """Generate realistic tracking events based on order status"""
    events = []
    base_time = order_date

    if status in ["shipped", "delivered", "delayed"]:
        events.append(
            {
                "timestamp": str(base_time + timedelta(hours=2)),
                "location": "Seller Warehouse",
                "status": "Order picked up by carrier",
            }
        )
        events.append(
            {"timestamp": str(base_time + timedelta(hours=14)), "location": "Origin Hub", "status": "In transit"}
        )

    if status == "delivered":
        events.append(
            {
                "timestamp": str(base_time + timedelta(days=2)),
                "location": "Destination City Hub",
                "status": "Arrived at destination city",
            }
        )
        events.append(
            {
                "timestamp": str(base_time + timedelta(days=3)),
                "location": "Local Delivery Centre",
                "status": "Out for delivery",
            }
        )
        events.append(
            {
                "timestamp": str(base_time + timedelta(days=3, hours=5)),
                "location": "Customer Address",
                "status": "Delivered successfully",
            }
        )

    if status == "delayed":
        events.append(
            {
                "timestamp": str(base_time + timedelta(days=2)),
                "location": "Sorting Hub",
                "status": "Package delayed due to high volume",
            }
        )
        events.append(
            {"timestamp": str(base_time + timedelta(days=3)), "location": "Sorting Hub", "status": "Processing resumed"}
        )

    if status == "shipped":
        events.append(
            {
                "timestamp": str(base_time + timedelta(days=1)),
                "location": "Regional Hub",
                "status": "In transit to destination city",
            }
        )

    return events


def get_current_location(status, events):
    """Get the most recent location from events"""
    if events:
        return events[-1]["location"]
    return "Unknown"


def get_current_tracking_status(status):
    """Map order status to tracking status"""
    mapping = {
        "shipped": "in_transit",
        "delivered": "delivered",
        "delayed": "delayed",
        "processing": "not_shipped",
        "cancelled": "cancelled",
    }
    return mapping.get(status, "unknown")


# ==========================================
# IMPORT PRODUCTS
# ==========================================
def import_products(db):
    logger.info("Starting product import from Amazon dataset...")

    existing_count = db.query(Product).count()
    if existing_count > 0:
        logger.info(f"Products already imported ({existing_count} rows). Skipping.")
        return existing_count

    df = pd.read_csv("data/amazon.csv")
    logger.info(f"Loaded {len(df)} rows from amazon.csv")

    # Drop duplicates on product_id
    df = df.drop_duplicates(subset=["product_id"])
    logger.info(f"After dedup: {len(df)} unique products")

    imported = 0
    skipped = 0

    for _, row in df.iterrows():
        try:
            price = clean_price(row.get("discounted_price"))
            actual_price = clean_price(row.get("actual_price"))
            rating = clean_rating(row.get("rating"))
            rating_count = clean_rating_count(row.get("rating_count"))
            discount = clean_discount(row.get("discount_percentage"))

            if price is None or price <= 0:
                skipped += 1
                continue

            product = Product(
                product_id=str(row["product_id"])[:50],
                name=str(row["product_name"])[:500],
                category=clean_category(row.get("category", "General")),
                price=price,
                actual_price=actual_price,
                discount_percent=discount,
                brand=extract_brand(row.get("product_name", "Unknown")),
                rating=rating,
                rating_count=rating_count,
                description=str(row.get("about_product", ""))[:2000],
                features=extract_features(row.get("about_product", "")),
                in_stock=True,
            )
            db.add(product)
            imported += 1

            # Commit every 100 rows
            if imported % 100 == 0:
                db.commit()
                logger.info(f"  Imported {imported} products so far...")

        except Exception as e:
            skipped += 1
            continue

    db.commit()
    logger.info(f"Products import complete. Imported: {imported}, Skipped: {skipped}")
    return imported


# ==========================================
# IMPORT ORDERS
# ==========================================
def import_orders(db):
    logger.info("Starting order import from Olist dataset...")

    existing_count = db.query(Order).count()
    if existing_count > 0:
        logger.info(f"Orders already imported ({existing_count} rows). Skipping.")
        return existing_count

    # Load both CSV files
    orders_df = pd.read_csv("data/olist_orders_dataset.csv")
    items_df = pd.read_csv("data/olist_order_items_dataset.csv")

    logger.info(f"Loaded {len(orders_df)} orders and {len(items_df)} items")

    # Calculate order value per order
    order_values = items_df.groupby("order_id")["price"].sum().reset_index()
    order_values.columns = ["order_id", "order_value"]

    # Merge
    orders_df = orders_df.merge(order_values, on="order_id", how="left")
    orders_df["order_value"] = orders_df["order_value"].fillna(100.0)

    # Filter to get a good mix of statuses for demo
    status_map = {
        "delivered": "delivered",
        "shipped": "shipped",
        "canceled": "cancelled",
        "processing": "processing",
        "invoiced": "processing",
        "approved": "processing",
        "created": "processing",
        "unavailable": "cancelled",
    }

    orders_df["clean_status"] = orders_df["order_status"].map(status_map)
    orders_df = orders_df.dropna(subset=["clean_status"])

    # Pick a balanced sample — 20 of each status
    sampled_frames = []
    for status in ["delivered", "shipped", "cancelled", "processing"]:
        subset = orders_df[orders_df["clean_status"] == status]
        n = min(20, len(subset))
        sampled_frames.append(subset.sample(n=n, random_state=42))

    # Add some delayed orders (modify shipped ones)
    shipped_sample = orders_df[orders_df["clean_status"] == "shipped"].sample(
        n=min(10, len(orders_df[orders_df["clean_status"] == "shipped"])), random_state=99
    )
    shipped_sample = shipped_sample.copy()
    shipped_sample["clean_status"] = "delayed"
    sampled_frames.append(shipped_sample)

    sampled = pd.concat(sampled_frames).drop_duplicates(subset=["order_id"])
    logger.info(f"Selected {len(sampled)} orders for import")

    imported = 0

    for idx, (_, row) in enumerate(sampled.iterrows()):
        try:
            # Parse dates
            order_date = pd.to_datetime(row["order_purchase_timestamp"], errors="coerce")
            if pd.isna(order_date):
                order_date = datetime.utcnow() - timedelta(days=random.randint(5, 30))
            else:
                order_date = order_date.to_pydatetime()

            expected_del = pd.to_datetime(row["order_estimated_delivery_date"], errors="coerce")
            if pd.isna(expected_del):
                expected_delivery = (order_date + timedelta(days=random.randint(5, 15))).date()
            else:
                expected_delivery = expected_del.date()

            status = row["clean_status"]
            carrier = random.choice(CARRIERS)
            tracking_number = generate_tracking_number(carrier)

            # Format order_id nicely
            order_id = f"ORD-{str(idx + 1001)}"

            # Simple items list
            items = [{"name": f"Product {idx + 1}", "qty": 1, "price": round(float(row["order_value"]), 2)}]

            order = Order(
                order_id=order_id,
                user_id=DEMO_USER_ID,
                status=status,
                items=items,
                carrier=carrier if status not in ["processing", "cancelled"] else None,
                tracking_number=tracking_number if status not in ["processing", "cancelled"] else None,
                order_date=order_date,
                expected_delivery=expected_delivery,
                order_value=round(float(row["order_value"]), 2),
            )
            db.add(order)

            # Create matching carrier_tracking record
            if status not in ["processing", "cancelled"]:
                events = generate_tracking_events(status, order_date, expected_delivery)
                tracking = CarrierTracking(
                    tracking_id=str(uuid.uuid4()),
                    tracking_number=tracking_number,
                    carrier_name=carrier,
                    current_status=get_current_tracking_status(status),
                    current_location=get_current_location(status, events),
                    events=events,
                    estimated_delivery=expected_delivery,
                    last_updated=datetime.utcnow(),
                )
                db.add(tracking)

            imported += 1

        except Exception as e:
            logger.error(f"Error importing order row {idx}: {e}")
            continue

    db.commit()
    logger.info(f"Orders import complete. Imported: {imported}")
    return imported


# ==========================================
# SEED POLICIES
# ==========================================
def seed_policies(db):
    from database.models import Policy

    existing = db.query(Policy).count()
    if existing > 0:
        logger.info(f"Policies already seeded ({existing} rows). Skipping.")
        return

    policies = [
        Policy(
            issue_type="damaged_product",
            policy_text="""Damaged Product Policy:
If you receive a damaged product, you are eligible for a full replacement or refund.
Steps to claim:
1. Report the damage within 48 hours of delivery with photo evidence.
2. Our support team will verify within 24 hours.
3. Once verified, a replacement will be shipped within 3-5 business days OR a full refund processed within 5-7 business days.
4. The damaged item must be kept until our logistics team arranges pickup.
Note: Damage reported after 48 hours may not be eligible for replacement but may qualify for partial refund.""",
        ),
        Policy(
            issue_type="wrong_item",
            policy_text="""Wrong Item Delivered Policy:
If you received a wrong item, you are eligible for a full replacement or refund.
Steps to resolve:
1. Report within 72 hours of delivery with a photo of the item received.
2. The correct item will be shipped within 2-3 business days after verification.
3. The wrong item will be picked up by our logistics partner at no cost to you.
4. If the correct item is out of stock, a full refund will be processed within 5-7 business days.""",
        ),
        Policy(
            issue_type="refund",
            policy_text="""Refund Policy:
Refunds are available under the following conditions:
- Product not delivered within 15 days of expected delivery date.
- Product received in damaged condition.
- Wrong item delivered.
- Product cancelled before shipment.
Refund process:
1. Submit a refund request through our support team.
2. Refund processed within 5-7 business days after approval.
3. Refund credited to the original payment method.
4. For cash on delivery orders, refund transferred to bank account within 7-10 business days.
Note: Refunds are not available for change of mind after delivery.""",
        ),
        Policy(
            issue_type="cancellation",
            policy_text="""Cancellation Policy:
Order cancellation is available under these conditions:
- Orders can be cancelled within 24 hours of placement for a full refund.
- Orders in processing status can be cancelled with full refund.
- Orders that have been shipped CANNOT be cancelled.
How to cancel:
1. Contact support with your order ID within 24 hours.
2. Cancellation confirmed within 2 hours.
3. Refund processed within 5-7 business days.
Note: For prepaid orders cancelled after 24 hours but before shipment, a 2% cancellation fee may apply.""",
        ),
    ]

    for policy in policies:
        db.add(policy)

    db.commit()
    logger.info(f"Seeded {len(policies)} policies")


# ==========================================
# SEED DEMO SUPPORT TICKETS
# ==========================================
def seed_demo_tickets(db):
    from database.models import SupportTicket

    existing = db.query(SupportTicket).filter(SupportTicket.user_id == DEMO_USER_ID).count()

    if existing > 0:
        logger.info(f"Demo tickets already exist ({existing}). Skipping.")
        return

    tickets = [
        SupportTicket(
            user_id=DEMO_USER_ID,
            order_id="ORD-1001",
            issue_type="damaged_product",
            priority="HIGH",
            status="resolved",
        ),
        SupportTicket(
            user_id=DEMO_USER_ID,
            order_id="ORD-1002",
            issue_type="wrong_item",
            priority="HIGH",
            status="resolved",
        ),
        SupportTicket(
            user_id=DEMO_USER_ID,
            order_id="ORD-1003",
            issue_type="refund",
            priority="MEDIUM",
            status="open",
        ),
    ]

    for ticket in tickets:
        db.add(ticket)

    db.commit()
    logger.info(f"Seeded {len(tickets)} demo support tickets for {DEMO_USER_ID}")


# ==========================================
# MAIN
# ==========================================
def run_import():
    logger.info("=" * 50)
    logger.info("Starting full database import")
    logger.info("=" * 50)

    db = SessionLocal()
    try:
        products_count = import_products(db)
        orders_count = import_orders(db)
        seed_policies(db)
        seed_demo_tickets(db)

        logger.info("=" * 50)
        logger.info("Import complete. Summary:")
        logger.info(f"  Products : {products_count}")
        logger.info(f"  Orders   : {orders_count}")
        logger.info(f"  Policies : 4")
        logger.info(f"  Demo tickets: 3")
        logger.info("=" * 50)

        print("\n✅ Products imported")
        print("✅ Orders imported")
        print("✅ Policies seeded")
        print("✅ Demo tickets seeded")
        print("\nDatabase is ready.")

    except Exception as e:
        db.rollback()
        logger.error(f"Import failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_import()
