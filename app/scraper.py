import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import random
import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent / "data" / "raw"

def generate_dataset(n_users=300, churn_months=6, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    categories = ["electronics","jewelery","men's clothing","women's clothing"]
    users, carts = [], []
    now = datetime.now(timezone.utc)

    for uid in range(1, n_users + 1):
        reg_days = random.randint(30, 1000)
        registered = now - timedelta(days=reg_days)

        users.append({
            "id": uid,
            "email": f"user{uid}@mail.com",
            "address": {"city": random.choice(["NY","LA","CHI"]), "zipcode": str(random.randint(10000,99999))},
            "registered_at": registered.isoformat()
        })

        n_carts = random.randint(0, 20)
        last_cart_date = None

        for cid in range(n_carts):
            cart_date = registered + timedelta(days=random.randint(1, reg_days))
            if cart_date > now:
                cart_date = now - timedelta(days=random.randint(0,10))

            last_cart_date = max(last_cart_date, cart_date) if last_cart_date else cart_date

            products = []
            for _ in range(random.randint(1,5)):
                products.append({
                    "productId": random.randint(1,100),
                    "category": random.choice(categories),
                    "price": round(random.uniform(5,500),2),
                    "quantity": random.randint(1,4)
                })

            carts.append({
                "id": len(carts)+1,
                "userId": uid,
                "date": cart_date.isoformat(),
                "products": products
            })

    labels = []
    for u in users:
        ucarts = [c for c in carts if c["userId"] == u["id"]]
        if not ucarts:
            churn = 1
        else:
            last = max(datetime.fromisoformat(c["date"]) for c in ucarts)
            churn = int((now - last).days > churn_months * 30)

        labels.append({"user_id": u["id"], "churn": churn})

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(users, open(RAW_DIR/"users.json","w"))
    json.dump(carts, open(RAW_DIR/"carts.json","w"))
    pd.DataFrame(labels).to_csv(RAW_DIR/"labels.csv", index=False)

if __name__ == "__main__":
    generate_dataset()
