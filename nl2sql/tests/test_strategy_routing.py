from nl2sql.src.strategy_bench.routing import CatalogEntry, MatchRule, RouteDecision, RoutingCatalog, RuleBasedRouter


def test_router_prefers_higher_priority_and_reuse() -> None:
    catalog = RoutingCatalog(
        [
            CatalogEntry(
                id="adapt-sales",
                route_type="adapt",
                match_rules=[MatchRule(type="keyword", keywords=["sales"], priority=10)],
                template="SELECT * FROM sales WHERE region = '{region}'",
                placeholders=["region"],
            ),
            CatalogEntry(
                id="reuse-sales",
                route_type="reuse",
                match_rules=[MatchRule(type="keyword", keywords=["sales"], priority=5)],
                sql="SELECT * FROM sales",
            ),
        ]
    )

    decision = RuleBasedRouter(catalog).route("show sales")

    assert isinstance(decision, RouteDecision)
    assert decision.strategy == "reuse"
    assert decision.catalog_entry_id == "reuse-sales"


def test_router_adapts_template_when_placeholder_is_extracted() -> None:
    catalog = RoutingCatalog(
        [
            CatalogEntry(
                id="orders-by-customer",
                route_type="adapt",
                match_rules=[
                    MatchRule(type="keyword", keywords=["orders"]),
                    MatchRule(type="regex", pattern=r"for (?P<customer>[A-Za-z]+)", priority=20),
                ],
                template="SELECT amount FROM orders WHERE customer = '{customer}'",
                placeholders=["customer"],
            )
        ]
    )

    decision = RuleBasedRouter(catalog).route("Show orders for Alice")

    assert decision.strategy == "adapt"
    assert decision.sql == "SELECT amount FROM orders WHERE customer = 'Alice'"


def test_router_degrades_to_generate_when_placeholder_is_missing() -> None:
    catalog = RoutingCatalog(
        [
            CatalogEntry(
                id="orders-by-customer",
                route_type="adapt",
                match_rules=[MatchRule(type="keyword", keywords=["orders"])],
                template="SELECT amount FROM orders WHERE customer = '{customer}'",
                placeholders=["customer"],
            )
        ]
    )

    decision = RuleBasedRouter(catalog).route("Show orders")

    assert decision.strategy == "generate"
    assert decision.reason == "adaptation left unresolved placeholders"
