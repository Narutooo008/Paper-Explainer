import re
import sys
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient

from backend.main import app
import backend.config as config

PUBLIC_PAGES = ["/", "/how-it-works", "/about", "/privacy"]


class HeadingAndLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.headings = []
        self.links = []
        self.images = []

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        if re.match(r"^h[1-6]$", tag):
            self.headings.append((tag, int(tag[1])))
        if tag == "a" and "href" in attr_dict:
            self.links.append(attr_dict["href"])
        if tag == "img":
            self.images.append(attr_dict)


class ProductionSEOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, follow_redirects=False)

    def test_public_pages_status_and_metadata(self):
        site_url = config.site_url()
        titles = set()
        descriptions = set()

        for path in PUBLIC_PAGES:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200, f"Page {path} returned status {res.status_code}")
            html = res.text

            # 1. Title tag (< 60 chars, unique)
            title_match = re.search(r"<title>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
            self.assertIsNotNone(title_match, f"Missing <title> in {path}")
            title = title_match.group(1).strip()
            self.assertTrue(len(title) > 0, f"Empty <title> in {path}")
            self.assertLess(len(title), 60, f"<title> in {path} exceeds 60 chars ({len(title)}): {title}")
            self.assertNotIn(title, titles, f"Duplicate <title> across pages: {title}")
            titles.add(title)

            # 2. Meta description (<= 160 chars, unique)
            desc_match = re.search(r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']', html, re.IGNORECASE)
            self.assertIsNotNone(desc_match, f"Missing meta description in {path}")
            desc = desc_match.group(1).strip()
            self.assertTrue(len(desc) > 0, f"Empty meta description in {path}")
            self.assertLessEqual(len(desc), 160, f"Meta description in {path} exceeds 160 chars ({len(desc)}): {desc}")
            self.assertNotIn(desc, descriptions, f"Duplicate meta description across pages: {desc}")
            descriptions.add(desc)

            # 3. Canonical link (absolute, matching clean URL)
            expected_canonical = f"{site_url}{path}" if path != "/" else f"{site_url}/"
            canon_match = re.search(r'<link\s+rel=["\']canonical["\']\s+href=["\'](.*?)["\']', html, re.IGNORECASE)
            self.assertIsNotNone(canon_match, f"Missing <link rel='canonical'> in {path}")
            canonical = canon_match.group(1).strip()
            self.assertEqual(canonical, expected_canonical, f"Canonical mismatch in {path}: expected {expected_canonical}, got {canonical}")

            # 4. Open Graph & Twitter tags
            self.assertTrue('property="og:title"' in html or "property='og:title'" in html, f"Missing og:title in {path}")
            self.assertTrue('property="og:description"' in html or "property='og:description'" in html, f"Missing og:description in {path}")
            self.assertTrue('property="og:image"' in html or "property='og:image'" in html, f"Missing og:image in {path}")
            self.assertTrue('property="og:url"' in html or "property='og:url'" in html, f"Missing og:url in {path}")
            self.assertIn(f'{site_url}/assets/og-image.png', html, f"Missing absolute og-image URL in {path}")
            self.assertTrue('name="twitter:card"' in html or "name='twitter:card'" in html, f"Missing twitter:card in {path}")

            # 5. Exactly one H1 and heading hierarchy check
            parser = HeadingAndLinkParser()
            parser.feed(html)
            h1_tags = [h for h in parser.headings if h[1] == 1]
            self.assertEqual(len(h1_tags), 1, f"Expected exactly one <h1> in {path}, found {len(h1_tags)}")

            # Check no skipped heading levels (e.g. h1 followed directly by h3)
            current_level = 1
            for tag, level in parser.headings:
                self.assertLessEqual(level, current_level + 1, f"Skipped heading level in {path}: h{current_level} directly to h{level}")
                current_level = max(current_level, level)

    def test_images_attributes(self):
        for path in PUBLIC_PAGES:
            res = self.client.get(path)
            parser = HeadingAndLinkParser()
            parser.feed(res.text)
            for img in parser.images:
                self.assertTrue("alt" in img and len(img["alt"].strip()) > 0, f"Image missing alt attribute in {path}: {img}")
                self.assertIn("width", img, f"Image missing explicit width in {path}: {img}")
                self.assertIn("height", img, f"Image missing explicit height in {path}: {img}")
                self.assertEqual(img.get("loading"), "lazy", f"Image missing loading='lazy' in {path}: {img}")

    def test_internal_links_integrity(self):
        for path in PUBLIC_PAGES:
            res = self.client.get(path)
            parser = HeadingAndLinkParser()
            parser.feed(res.text)

            for link in parser.links:
                if link.startswith("/") and not link.startswith("//"):
                    clean_target = link.split("?")[0].split("#")[0]
                    if not clean_target:
                        clean_target = "/"
                    target_res = self.client.get(clean_target)
                    self.assertIn(target_res.status_code, (200, 301), f"Broken internal link '{link}' on page {path} returned {target_res.status_code}")

    def test_clean_url_redirects(self):
        # 1. /index.html should 301 redirect to /
        res_index = self.client.get("/index.html")
        self.assertEqual(res_index.status_code, 301)
        self.assertEqual(res_index.headers["location"], "/")

        # 2. Trailing slashes should 301 redirect to clean path
        for path in ["/how-it-works/", "/about/", "/privacy/"]:
            res_slash = self.client.get(path)
            self.assertEqual(res_slash.status_code, 301)
            self.assertEqual(res_slash.headers["location"], path.rstrip("/"))

    def test_private_pages_robots_and_headers(self):
        job_id = "a612ed52b01bd0bf"
        res_p = self.client.get(f"/p/{job_id}")
        self.assertEqual(res_p.status_code, 200)
        self.assertIn("X-Robots-Tag", res_p.headers)
        self.assertEqual(res_p.headers["X-Robots-Tag"], "noindex, nofollow")
        self.assertIn('<meta name="robots" content="noindex, nofollow">', res_p.text)

        # Exactly one H1 on result page
        parser = HeadingAndLinkParser()
        parser.feed(res_p.text)
        h1_tags = [h for h in parser.headings if h[1] == 1]
        self.assertEqual(len(h1_tags), 1, f"Expected exactly one <h1> in /p/{job_id}, found {len(h1_tags)}")

        # /history
        res_hist = self.client.get("/history")
        self.assertEqual(res_hist.status_code, 200)
        self.assertEqual(res_hist.headers.get("X-Robots-Tag"), "noindex, nofollow")

        # /api/*
        res_api = self.client.get("/api/jobs")
        self.assertEqual(res_api.status_code, 200)
        self.assertEqual(res_api.headers.get("X-Robots-Tag"), "noindex, nofollow")

        # /media/*
        res_media = self.client.get(f"/media/{job_id}/result.json")
        self.assertEqual(res_media.status_code, 200)
        self.assertEqual(res_media.headers.get("X-Robots-Tag"), "noindex, nofollow")

    def test_robots_txt(self):
        res = self.client.get("/robots.txt")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.headers["content-type"].startswith("text/plain"))
        txt = res.text
        self.assertIn("Allow: /", txt)
        self.assertIn("Disallow: /api/", txt)
        self.assertIn("Disallow: /media/", txt)
        self.assertIn("Disallow: /p/", txt)
        self.assertIn("Disallow: /history", txt)
        site_url = config.site_url()
        self.assertIn(f"Sitemap: {site_url}/sitemap.xml", txt)

    def test_sitemap_xml(self):
        res = self.client.get("/sitemap.xml")
        self.assertEqual(res.status_code, 200)
        self.assertIn("xml", res.headers["content-type"])
        root = ET.fromstring(res.text)
        urls = [elem.text for elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
        site_url = config.site_url()

        expected_urls = [
            f"{site_url}/",
            f"{site_url}/how-it-works",
            f"{site_url}/about",
            f"{site_url}/privacy",
        ]
        self.assertEqual(set(urls), set(expected_urls), f"Sitemap locs mismatch: {urls}")

    def test_llms_txt(self):
        res = self.client.get("/llms.txt")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/markdown", res.headers["content-type"])
        txt = res.text
        self.assertIn("# PaperExplainer", txt)
        self.assertIn("/how-it-works", txt)
        self.assertIn("/about", txt)
        self.assertIn("/privacy", txt)

    def test_404_error_handling(self):
        # 1. Web route 404
        res = self.client.get("/some-nonexistent-page")
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.headers.get("X-Robots-Tag"), "noindex, nofollow")
        self.assertIn('<meta name="robots" content="noindex, nofollow">', res.text)
        self.assertIn("Page Not Found", res.text)
        self.assertIn('<a href="/"', res.text)
        self.assertIn('<a href="/how-it-works"', res.text)

        # Exactly one H1 on 404 page
        parser = HeadingAndLinkParser()
        parser.feed(res.text)
        h1_tags = [h for h in parser.headings if h[1] == 1]
        self.assertEqual(len(h1_tags), 1)

        # 2. API route 404
        res_api = self.client.get("/api/nonexistent-endpoint")
        self.assertEqual(res_api.status_code, 404)
        self.assertEqual(res_api.headers.get("X-Robots-Tag"), "noindex, nofollow")
        self.assertIsNotNone(res_api.json().get("detail"))

    def test_structured_data_schemas(self):
        # Home: WebApplication
        res_home = self.client.get("/")
        self.assertIn('"@type": "WebApplication"', res_home.text)
        self.assertIn('"applicationCategory": "EducationalApplication"', res_home.text)

        # About: Organization
        res_about = self.client.get("/about")
        self.assertIn('"@type": "Organization"', res_about.text)
        self.assertIn(config.contact_email(), res_about.text)

        # Breadcrumbs
        for path in ["/how-it-works", "/about", "/privacy"]:
            res_page = self.client.get(path)
            self.assertIn('"@type": "BreadcrumbList"', res_page.text)


if __name__ == "__main__":
    unittest.main()
