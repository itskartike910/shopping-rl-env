"""
Playwright Chromium Browser Interface for Shopping RL Environment.

Controls a local Chromium browser instance via Playwright to interact
with the Flask store visually, demonstrating real browser-agent capabilities.
"""

import time
from typing import Dict, Any, Optional

class BrowserShoppingAgent:
    """
    Drives Chromium to browse the local shopping website, search, add products,
    and checkout.
    """
    def __init__(self, base_url: str = "http://localhost:5000", headless: bool = True):
        self.base_url = base_url
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.page = None

    def start(self):
        """Launch Playwright and Chromium."""
        try:
            from playwright.sync_api import sync_playwright
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            self.page = self.browser.new_page()
            self.page.goto(self.base_url)
            return True
        except Exception as e:
            print(f"[BrowserAgent] Failed to launch Chromium: {e}")
            return False

    def search(self, query: str):
        """Type search query and submit."""
        if not self.page:
            return False
        self.page.fill("#search-input", query)
        self.page.click("#search-btn")
        self.page.wait_for_load_state("networkidle")
        return True

    def sort_by(self, sort_option: str):
        """Select a sort dropdown option (price_asc, price_desc, rating, name)."""
        if not self.page:
            return False
        self.page.select_option("#sort-select", sort_option)
        self.page.wait_for_load_state("networkidle")
        return True

    def add_product_to_cart(self, product_id: str):
        """Click 'Add to Cart' for a specific product ID."""
        if not self.page:
            return False
        btn_selector = f"#btn-add-{product_id.upper()}"
        if self.page.locator(btn_selector).count() > 0:
            self.page.click(btn_selector)
            self.page.wait_for_load_state("networkidle")
            return True
        else:
            # Try navigating directly to product page
            self.page.goto(f"{self.base_url}/product/{product_id.upper()}")
            self.page.click("#detail-add-to-cart-btn")
            self.page.wait_for_load_state("networkidle")
            return True

    def view_cart(self):
        """Navigate to cart page."""
        if not self.page:
            return False
        self.page.goto(f"{self.base_url}/cart")
        self.page.wait_for_load_state("networkidle")
        return True

    def checkout(self) -> Optional[str]:
        """Click checkout and extract confirmed order ID."""
        if not self.page:
            return None
        self.view_cart()
        btn_selector = "#checkout-button"
        if self.page.locator(btn_selector).count() > 0:
            self.page.click(btn_selector)
            self.page.wait_for_load_state("networkidle")
            # Extract confirmed order id
            if self.page.locator("#confirmed-order-id").count() > 0:
                order_id = self.page.locator("#confirmed-order-id").inner_text()
                return order_id.strip()
        return None

    def close(self):
        """Clean up browser resources."""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
