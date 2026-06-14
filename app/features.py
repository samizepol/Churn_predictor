import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import os

RAW_DIR = Path(__file__).parent / "data" / "raw"

def build_features_from_raw(users, carts, observation_cutoff: datetime = None):
    if observation_cutoff is None:
        
        observation_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    elif observation_cutoff.tzinfo is None:
        observation_cutoff = observation_cutoff.replace(tzinfo=timezone.utc)

    rows = []

    for user in users:
        uid = user["id"]

        # 1. Extract only carts BEFORE the historical cutoff date
        ucarts = [
            c for c in carts
            if c["userId"] == uid
            and datetime.fromisoformat(c["date"]).replace(tzinfo=timezone.utc) <= observation_cutoff
        ]

        registered = datetime.fromisoformat(user["registered_at"]).replace(tzinfo=timezone.utc)
        account_age_days = (observation_cutoff - registered).days  # Relative to historical snapshot

        # Extract sorted timestamps for behavioral calculations
        ucart_dates = sorted(
            datetime.fromisoformat(c["date"]).replace(tzinfo=timezone.utc)
            for c in ucarts
        )

        # 2. Activity Velocity 
        if len(ucart_dates) >= 2 and account_age_days > 0:
            midpoint_date = registered + timedelta(days=account_age_days / 2)
            early_carts = sum(1 for d in ucart_dates if d <= midpoint_date)
            recent_carts = sum(1 for d in ucart_dates if d > midpoint_date)
            activity_velocity = recent_carts - early_carts
        else:
            activity_velocity = 0 

        # 3. Engagement Ratios
        if account_age_days > 0:
            cart_frequency_ratio = len(ucarts) / account_age_days
        else:
            cart_frequency_ratio = 0

        total_spent = 0
        total_items = 0
        expensive = 0
        cheap = 0
        categories = set()

        # 4. Aggregations 
        for c in ucarts:
            for p in c["products"]:
                value = p["price"] * p["quantity"]
                total_spent += value
                total_items += p["quantity"]
                categories.add(p["category"])
                if p["price"] > 100:
                    expensive += p["quantity"]
                else:
                    cheap += p["quantity"]

        avg_items_per_cart = total_items / max(len(ucarts), 1)
        expensive_ratio = expensive / max(expensive + cheap, 1)

        
        rows.append({
            "user_id": uid,
            "account_age_days": max(account_age_days, 0),
            "activity_velocity": activity_velocity,
            "cart_frequency_ratio": cart_frequency_ratio,
            "total_spent": total_spent,
            "avg_items_per_cart": avg_items_per_cart,
            "distinct_categories": len(categories),
            "expensive_ratio": expensive_ratio,
            "bought_electronics": int("electronics" in categories),
        })

    return pd.DataFrame(rows)

def load_training_dataset(observation_cutoff: datetime = None):
    users = json.load(open(RAW_DIR / "users.json"))
    carts = json.load(open(RAW_DIR / "carts.json"))
    labels = pd.read_csv(RAW_DIR / "labels.csv")
    
    X = build_features_from_raw(users, carts, observation_cutoff=observation_cutoff)
    return X.merge(labels, on="user_id")

def single_user_features(payload):
    return build_features_from_raw([payload["user"]], payload["carts"], observation_cutoff=datetime.now(timezone.utc))