PRAGMA foreign_keys = ON;

WITH RECURSIVE seq(n) AS (
    SELECT 1
    UNION ALL
    SELECT n + 1 FROM seq WHERE n < 24
)
INSERT INTO categories (category_id, category_name, parent_category_id)
SELECT
    n,
    CASE
        WHEN n = 1 THEN 'Electronics'
        WHEN n = 2 THEN 'Home'
        WHEN n = 3 THEN 'Fashion'
        WHEN n = 4 THEN 'Sports'
        WHEN n = 5 THEN 'Beauty'
        WHEN n = 6 THEN 'Books'
        ELSE 'Category ' || n
    END,
    CASE
        WHEN n <= 6 THEN NULL
        WHEN n <= 12 THEN ((n - 1) % 6) + 1
        WHEN n <= 18 THEN ((n - 7) % 6) + 1
        ELSE ((n - 13) % 6) + 1
    END
FROM seq;

WITH RECURSIVE seq(n) AS (
    SELECT 1
    UNION ALL
    SELECT n + 1 FROM seq WHERE n < 420
)
INSERT INTO customers (
    customer_id,
    first_name,
    last_name,
    email,
    city,
    state,
    signup_date,
    customer_segment
)
SELECT
    n,
    'First' || n,
    'Last' || n,
    'customer' || n || '@example.com',
    CASE n % 10
        WHEN 0 THEN 'Seattle'
        WHEN 1 THEN 'San Francisco'
        WHEN 2 THEN 'Austin'
        WHEN 3 THEN 'Chicago'
        WHEN 4 THEN 'New York'
        WHEN 5 THEN 'Boston'
        WHEN 6 THEN 'Denver'
        WHEN 7 THEN 'Miami'
        WHEN 8 THEN CASE WHEN n % 40 = 8 THEN NULL ELSE 'Portland' END
        ELSE 'Atlanta'
    END,
    CASE n % 9
        WHEN 0 THEN 'CA'
        WHEN 1 THEN 'WA'
        WHEN 2 THEN 'TX'
        WHEN 3 THEN 'IL'
        WHEN 4 THEN 'NY'
        WHEN 5 THEN 'MA'
        WHEN 6 THEN 'CO'
        WHEN 7 THEN 'FL'
        ELSE NULL
    END,
    date('2022-01-01', '+' || (n * 3) || ' day'),
    CASE n % 5
        WHEN 0 THEN 'VIP'
        WHEN 1 THEN 'Loyal'
        WHEN 2 THEN 'Growth'
        WHEN 3 THEN 'Occasional'
        ELSE 'New'
    END
FROM seq;

WITH RECURSIVE seq(n) AS (
    SELECT 1
    UNION ALL
    SELECT n + 1 FROM seq WHERE n < 220
)
INSERT INTO products (
    product_id,
    product_name,
    category_id,
    brand,
    unit_price,
    is_active,
    launched_at
)
SELECT
    n,
    'Product ' || n,
    ((n - 1) % 18) + 1,
    CASE n % 8
        WHEN 0 THEN 'Acme'
        WHEN 1 THEN 'Northwind'
        WHEN 2 THEN 'Globex'
        WHEN 3 THEN 'Initech'
        WHEN 4 THEN 'Umbrella'
        WHEN 5 THEN 'Soylent'
        WHEN 6 THEN CASE WHEN n % 32 = 6 THEN 'Brand-X/Legacy' ELSE 'Stark' END
        ELSE 'Wayne'
    END,
    round(8 + ((n * 17) % 240) + ((n % 7) * 0.95), 2),
    CASE WHEN n % 11 = 0 THEN 0 ELSE 1 END,
    CASE WHEN n % 14 = 0 THEN NULL ELSE date('2021-01-01', '+' || (n * 6) || ' day') END
FROM seq;

WITH RECURSIVE seq(n) AS (
    SELECT 1
    UNION ALL
    SELECT n + 1 FROM seq WHERE n < 720
)
INSERT INTO orders (
    order_id,
    customer_id,
    order_date,
    order_status,
    payment_method,
    shipping_country,
    shipping_state,
    total_amount
)
SELECT
    n,
    ((n - 1) % 330) + 1,
    datetime('2023-01-01', '+' || n || ' hour'),
    CASE n % 7
        WHEN 0 THEN 'cancelled'
        WHEN 1 THEN 'processing'
        WHEN 2 THEN 'shipped'
        WHEN 3 THEN 'delivered'
        WHEN 4 THEN 'returned'
        WHEN 5 THEN 'pending'
        ELSE 'shipped'
    END,
    CASE n % 4
        WHEN 0 THEN 'credit_card'
        WHEN 1 THEN 'paypal'
        WHEN 2 THEN CASE WHEN n % 20 = 2 THEN 'crypto' ELSE 'apple_pay' END
        ELSE 'gift_card'
    END,
    CASE n % 6
        WHEN 0 THEN 'USA'
        WHEN 1 THEN 'USA'
        WHEN 2 THEN 'USA'
        WHEN 3 THEN 'Canada'
        WHEN 4 THEN 'USA'
        ELSE 'UK'
    END,
    CASE
        WHEN n % 6 IN (3, 5) THEN NULL
        WHEN n % 8 = 0 THEN 'CA'
        WHEN n % 8 = 1 THEN 'TX'
        WHEN n % 8 = 2 THEN 'WA'
        WHEN n % 8 = 3 THEN 'NY'
        WHEN n % 8 = 4 THEN 'FL'
        WHEN n % 8 = 5 THEN 'IL'
        WHEN n % 8 = 6 THEN 'CO'
        ELSE 'MA'
    END,
    CASE
        WHEN n % 90 = 0 THEN 0
        ELSE round(25 + ((n * 29) % 900) + ((n % 5) * 1.5), 2)
    END
FROM seq;

WITH RECURSIVE seq(n) AS (
    SELECT 1
    UNION ALL
    SELECT n + 1 FROM seq WHERE n < 1440
)
INSERT INTO order_items (
    order_item_id,
    order_id,
    product_id,
    quantity,
    unit_price,
    discount_amount
)
SELECT
    n,
    ((n - 1) % 720) + 1,
    ((n * 7 - 1) % 210) + 1,
    (n % 4) + 1,
    round(6 + ((n * 13) % 220) + ((n % 6) * 0.75), 2),
    CASE
        WHEN n % 10 = 0 THEN round(((n % 5) + 1) * 1.25, 2)
        WHEN n % 7 = 0 THEN round(((n % 4) + 1) * 0.75, 2)
        WHEN n % 18 = 0 THEN NULL
        ELSE 0
    END
FROM seq;

WITH RECURSIVE seq(n) AS (
    SELECT 1
    UNION ALL
    SELECT n + 1 FROM seq WHERE n < 180
)
INSERT INTO returns (
    return_id,
    order_item_id,
    return_date,
    return_reason,
    refund_amount,
    return_status
)
SELECT
    n,
    ((n * 5 - 1) % 900) + 1,
    date('2024-01-01', '+' || (n % 180) || ' day'),
    CASE
        WHEN n % 8 = 0 THEN NULL
        WHEN n % 11 = 0 THEN ''
        WHEN n % 5 = 0 THEN 'damaged'
        WHEN n % 5 = 1 THEN 'wrong_size'
        WHEN n % 5 = 2 THEN 'late_delivery'
        WHEN n % 5 = 3 THEN 'changed_mind'
        ELSE 'quality_issue'
    END,
    round(4 + ((n * 11) % 160) + ((n % 3) * 0.6), 2),
    CASE n % 3
        WHEN 0 THEN 'approved'
        WHEN 1 THEN 'pending'
        ELSE 'rejected'
    END
FROM seq;
